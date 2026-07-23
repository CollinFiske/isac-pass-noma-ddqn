
# =====================================================================================
#  mobility-ddqn-v3  -  UNIFIED JOINT DDQN   (single self-contained file: physics + net + training)
# -------------------------------------------------------------------------------------
#  A single agent learns to (a) GROUP mobile users into NOMA pairs and (b) SPLIT each pair's power
#  (alpha), maximizing a semantic utility under sensing- and rate-QoS, beating an exhaustive-greedy
#  baseline. It is a REAL Double-DQN: ONE network `JointQNet` scores every (pair, alpha) action and
#  is trained PURELY from experienced rewards + Bellman bootstrapping (no analytic labels).
#
#  Pairing one scene is a short MDP: pick a (pair, alpha) -> collect that pair's reward -> remove the
#  two users AND their comm antennas -> repeat until nobody is left. Because an early choice changes
#  what later pairs get, the Q-function's lookahead can beat greedy's myopic "grab the best pair now".
#
#  RUN:   python train.py     # trains -> NEW_isac_pass_noma_ddqn.pth (+ eval_curve.npy, training_log.npy)
#  The demo/eval scripts (inference / compare-ddqn-vs-greedy / evaluate-ddqn) import this file as `tv` for the
#  shared physics/net/policy. Importing does NOT trigger training (guarded by __main__ at the bottom).
# =====================================================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import combinations

device = torch.device("cpu")   # hardcoded CPU; switch to "cuda" if a GPU is available 

# system parameters
F_CARRIER, C_LIGHT = 28e9, 3e8
LAMBDA = C_LIGHT / F_CARRIER
N_EFF = 1.4
LAMBDA_G = LAMBDA / N_EFF
ETA = (LAMBDA / (4 * np.pi)) ** 2
B_SIG = 100e6              # signal bandwidth (Hz)
SIGMA2 = 2.5e-12           # noise power sigma^2 (folds in NF=8 dB)
P_TOTAL, RHO = 1.0, 0.3    # total power, fraction to sensing
P_S, P_C = RHO * P_TOTAL, (1.0 - RHO) * P_TOTAL
RCS = 1.0                  # target radar cross-section
T_INT = 0.04               # radar coherent integration time (s)
G_P = T_INT * B_SIG        # coherent processing gain (eq 14)
GAMMA_MIN_DB = 15.0        # min sensing-SNR QoS threshold (dB, eq 16)
R_MIN = 1.0                # min per-user rate (bit/s/Hz, eq c_rate)
RATE_PENALTY_FACTOR = 10.0 # reward penalty per bit of R_min shortfall
SENSING_PENALTY_FACTOR = 1.0
N_S, N_C = 8, 8            # 8 sensing + 8 comm pinching antennas (PAs)
DMIN = LAMBDA / 2          # min PA spacing
L_PITCH, W_PITCH, H_WG = 105.0, 68.0, 12.0
Y_S, Y_C = W_PITCH / 2.0 - 0.5, W_PITCH / 2.0 + 0.5
ALPHA_MIN, ALPHA_MAX = 0.55, 0.95
FID_A1, FID_A2, FID_C1, FID_C2 = 0.30, 0.98, 0.25, -0.8
WEIGHT_SET = np.array([0.3, 0.6, 0.7, 0.8, 1.0])
ALPHA_V, SIGMA_V, DELTA_T = 0.9, 0.5, 0.04   # velocity memory, walk-noise std, slot length (s)

# discrete NOMA power-split grid the agent chooses from (0.55..0.95)
ALPHA_LEVELS = np.round(np.arange(ALPHA_MIN, ALPHA_MAX + 1e-9, 0.05), 2)
K_ALPHA = len(ALPHA_LEVELS)   # = 9

N_MAX = 12
CHAN_DB_OFFSET = 90.0
USER_FEAT = 7                 # [pos_x, pos_y, vel_x, vel_y, fading, weight, chan_dB]
PAIR_DIM = 2 * USER_FEAT      # 14 - two users glued together
CTX_DIM = 6                   # NEW: remaining-pool context (lets a compact net see the "rest of the scene")
IN_DIM = PAIR_DIM + CTX_DIM   # 20 - JointQNet input width

# trained model path for other files: Trainer saves it, demos/eval load it.
MODEL_PATH = os.environ.get("MODEL_PATH", "NEW_isac_pass_noma_ddqn.pth")


# random per-user signal blockage - some users just have worse luck
def shadowing(size):
    return 10.0 ** (np.random.normal(0.0, 2.0, size) / 20.0)

# comm signal strength at a user, summed over its antennas (eq 4-5)
#   h_{c,k,n} = (sqrt(eta)/d) e^{-j 2pi/lambda d}   free-space loss (~1/d) + air phase (eq 4)
#   g_{c,n}   = e^{-j 2pi/lambda_g x_c,n}           in-waveguide phase before the port (eq 5)
def comm_pa_gain(pt, x_c_sel, fading_k):
    d = np.sqrt((pt[0] - x_c_sel) ** 2 + (pt[1] - Y_C) ** 2 + H_WG ** 2)
    h_c = (np.sqrt(ETA) / d) * np.exp(-1j * (2 * np.pi / LAMBDA) * d)
    g_c = np.exp(-1j * (2 * np.pi / LAMBDA_G) * x_c_sel)
    return np.sum(g_c * h_c) * fading_k

# radar SNR (dB) for detecting a user - two-way echo, squared terms (eq 12, 14, 16)
def sensing_snr_db(pt, x_s):
    d = np.sqrt((pt[0] - x_s) ** 2 + (pt[1] - Y_S) ** 2 + H_WG ** 2)
    h_s = (np.sqrt(ETA) / d) * np.exp(-1j * (2 * np.pi / LAMBDA) * d)
    g_s = np.exp(-1j * (2 * np.pi / LAMBDA_G) * x_s)
    beta = np.sqrt(RCS / (4 * np.pi * ETA)) * np.sum((g_s ** 2) * (h_s ** 2))   # eq (12)
    gamma = (P_S / N_S) * G_P * (np.abs(beta) ** 2) / SIGMA2                    # eq (14)
    return 10.0 * np.log10(max(float(gamma), 1e-12))

# give a pair the m nearest unused comm PAs and remove them from the pool (eq 6/7 exclusivity)
def assign_comm_pas(group_cx, pool, x_c, m):
    picks = sorted(pool, key=lambda n: abs(x_c[n] - group_cx))[:m]
    for n in picks:
        pool.remove(n)
    mm = max(1, len(picks))
    xc_sel = np.clip(group_cx + (np.arange(mm) - mm / 2.0) * DMIN, 0.0, L_PITCH)
    return xc_sel, pool

# semantic quality as a logistic S-curve of SINR (eq 17)
def fidelity(gamma):
    return FID_A1 + (FID_A2 - FID_A1) / (1.0 + np.exp(-(FID_C1 * gamma + FID_C2)))

# NOMA SINRs: weak user (smaller |h|^2) gets share a_w and eats interference; strong cancels weak via SIC
#   gamma_w = a_w |h_w|^2 / (a_s |h_w|^2 + sigma^2)   (eq 10)
#   gamma_s = a_s |h_s|^2 / sigma^2                   (eq 11)
def pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w):
    hu2 = Pm * np.abs(comm_pa_gain(points[u], xc_sel, fading[u])) ** 2
    hv2 = Pm * np.abs(comm_pa_gain(points[v], xc_sel, fading[v])) ** 2
    if hu2 <= hv2:
        w_idx, s_idx, hw2, hs2 = u, v, hu2, hv2
    else:
        w_idx, s_idx, hw2, hs2 = v, u, hv2, hu2
    a_s = 1.0 - a_w
    gamma_w = (a_w * hw2) / (a_s * hw2 + SIGMA2)
    gamma_s = (a_s * hs2) / SIGMA2
    return w_idx, s_idx, gamma_w, gamma_s

# semantic utility of a pair: sum_k w_k * fidelity(gamma_k) * log2(1+gamma_k)  (eq 18)
def pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w):
    w_idx, s_idx, gamma_w, gamma_s = pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w)
    return (weights[w_idx] * fidelity(gamma_w) * np.log2(1.0 + gamma_w)
            + weights[s_idx] * fidelity(gamma_s) * np.log2(1.0 + gamma_s))

# soft R_min penalty: RATE_PENALTY_FACTOR * sum_k max(0, R_MIN - log2(1+gamma_k))  (eq c_rate)
def rate_penalty(u, v, points, xc_sel, Pm, fading, a_w):
    _, _, gamma_w, gamma_s = pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w)
    return RATE_PENALTY_FACTOR * (max(0.0, R_MIN - np.log2(1.0 + gamma_w)) + max(0.0, R_MIN - np.log2(1.0 + gamma_s)))

# analytic best alpha for a pair: (pure utility at the best alpha, that index, objective-per-alpha vector).
#   objective(alpha) = pair_utility_at_alpha(alpha) - rate_penalty(alpha)
# The agent does NOT train against this - it is used only by evaluate-ddqn.py to grade the net's partner
# picks against the true best, and by the "best" alpha_mode in pairs_total_utility.
def best_alpha(u, v, points, xc_sel, Pm, fading, weights):
    utils = np.array([pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a)
                      for a in ALPHA_LEVELS], dtype=np.float32)
    rpen = np.array([rate_penalty(u, v, points, xc_sel, Pm, fading, a)
                     for a in ALPHA_LEVELS], dtype=np.float32)
    obj = utils - rpen
    k = int(np.argmax(obj))
    return float(utils[k]), k, obj

# closed-form power split (greedy baseline): a_w = clip(|h_s|^2/(|h_w|^2+|h_s|^2), .55, .95)  (eq 23)
def closed_form_alpha(u, v, points, xc_sel, Pm, fading):
    hu2 = Pm * np.abs(comm_pa_gain(points[u], xc_sel, fading[u])) ** 2
    hv2 = Pm * np.abs(comm_pa_gain(points[v], xc_sel, fading[v])) ** 2
    hw2, hs2 = (hu2, hv2) if hu2 <= hv2 else (hv2, hu2)
    return float(np.clip(hs2 / (hw2 + hs2 + 1e-30), ALPHA_MIN, ALPHA_MAX))

# slide antennas to follow the crowd: sensing spread over the x-extent, comm bunched at the centroid
def reposition_pas(active_pts):
    xmin, xmax = active_pts[:, 0].min(), active_pts[:, 0].max()
    if xmax - xmin < (N_S - 1) * DMIN:
        c = 0.5 * (xmin + xmax)
        xmin, xmax = c - (N_S - 1) * DMIN / 2, c + (N_S - 1) * DMIN / 2
    x_s = np.clip(np.linspace(xmin, xmax, N_S), 0.0, L_PITCH)
    cx = np.clip(np.mean(active_pts[:, 0]), 0.0, L_PITCH)
    x_c = np.clip(cx + (np.arange(N_C) - N_C / 2.0) * DMIN, 0.0, L_PITCH)
    return x_s, x_c

# equation for user mobility (using Gauss-Markov mobility model)
def step_mobility(points, velocities, mean_vel, n_sub=1):
    for _ in range(n_sub):
        noise = np.random.normal(0, 1, velocities.shape)
        velocities[:] = (ALPHA_V * velocities + (1.0 - ALPHA_V) * mean_vel
                         + SIGMA_V * np.sqrt(1.0 - ALPHA_V ** 2) * noise)
        points[:, 0] += velocities[:, 0] * DELTA_T
        points[:, 1] += velocities[:, 1] * DELTA_T
        for ax, hi in ((0, L_PITCH), (1, W_PITCH)):
            lo_hit, hi_hit = points[:, ax] < 0.0, points[:, ax] > hi
            velocities[lo_hit | hi_hit, ax] *= -1.0
            mean_vel[lo_hit | hi_hit, ax] *= -1.0
            points[:, ax] = np.clip(points[:, ax], 0.0, hi)
    return velocities


# Per-user rows [pos_x, pos_y, vel_x, vel_y, fading, weight, chan_dB]
def build_base_features(points, velocities, weights, fading, x_c):
    N = len(points)
    base = np.zeros((N, USER_FEAT), dtype=np.float32)
    for i in range(N):
        g2 = np.abs(comm_pa_gain(points[i], x_c, fading[i])) ** 2
        base[i] = [points[i, 0], points[i, 1], velocities[i, 0], velocities[i, 1],
                   fading[i], weights[i], 10.0 * np.log10(g2 + 1e-30) + CHAN_DB_OFFSET]
    return base

# put two users' 7 features into a 14-vector
def pair_input(base, i, j):
    return np.concatenate([base[i], base[j]]).astype(np.float32)

# fixed-size summary of the users/PAs still left to pair. Prepended to every pair input so the Q-net can lookahead at the bootstraped future value
def build_context(base, remaining, pool):
    rem = np.asarray(remaining, dtype=bool)
    n_rem = int(rem.sum())
    if n_rem > 0:
        ch, wt, px = base[rem, 6], base[rem, 5], base[rem, 0]
        mean_ch, std_ch, mean_wt, mean_px = ch.mean(), ch.std(), wt.mean(), px.mean()
    else:
        mean_ch = std_ch = mean_wt = mean_px = 0.0
    return np.array([n_rem / N_MAX, len(pool) / N_C, mean_ch / 100.0, std_ch / 20.0, mean_wt, mean_px / L_PITCH], dtype=np.float32)

# rows for a batch of candidate pairs at a given state (same ctx for all pairs in that state)
def cand_rows(base, ctx, cand):
    return np.stack([np.concatenate([pair_input(base, i, j), ctx]) for (i, j) in cand]).astype(np.float32)


# THE NETWORK itself
class JointQNet(nn.Module):
    def __init__(self, in_dim=IN_DIM, k_alpha=K_ALPHA):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k_alpha)) # 3 layer model

    def forward(self, x):
        return self.net(x)

# rolling memory of transitions (s, chosen alpha idx, reward, base, remaining_after, pool_after, done)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *item):
        self.buffer.append(item)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# getting the greedy action for that part of the training
def greedy_action(net, base, ctx, cand):
    rows = torch.tensor(cand_rows(base, ctx, cand))
    with torch.no_grad():
        q = net(rows).numpy()          # (C, K_ALPHA)
    flat = int(np.argmax(q))
    ci, ki = divmod(flat, K_ALPHA)
    return ci, ki, q  # returns (cand index, alpha index, Q matrix)

# max_k Q(state, pair i-j, alpha_k) = the net's grouping value of pairing i with j
def pair_group_value(net, base, ctx, i, j): 
    row = torch.tensor(np.concatenate([pair_input(base, i, j), ctx])[None, :].astype(np.float32))
    with torch.no_grad():
        return float(net(row).max())

# inference policy - using the nn
def ddqn_form_pairs(net, base, remaining_in, x_s=None, x_c=None):
    remaining = list(remaining_in)
    n0 = sum(remaining)
    p_groups = max(1, n0 // 2)
    m = max(1, N_C // p_groups)
    pool = list(range(N_C))
    pairs = []
    while sum(remaining) >= 2:
        ctx = build_context(base, remaining, pool)
        avail = [i for i, r in enumerate(remaining) if r]
        cand = [(avail[a], avail[b]) for a in range(len(avail)) for b in range(a + 1, len(avail))]
        ci, ki, _ = greedy_action(net, base, ctx, cand)
        u1, u2 = cand[ci]
        pairs.append((u1, u2, float(ALPHA_LEVELS[ki])))
        remaining[u1] = remaining[u2] = False
        for _ in range(min(m, len(pool))):      # shrink the free-PA count exactly as training did
            pool.pop()
    return pairs


def make_scene(N, rng=None):
    """Random frozen scene of N users + precomputed per-user sensing penalty."""
    if rng is None:
        pts = np.zeros((N, 3))
        pts[:, 0] = np.random.uniform(0, L_PITCH, N); pts[:, 1] = np.random.uniform(0, W_PITCH, N)
        vel = np.random.normal(0, SIGMA_V, (N, 2)); w = np.random.choice(WEIGHT_SET, N)
        fad = shadowing((N,))
    else:
        pts = np.zeros((N, 3))
        pts[:, 0] = rng.uniform(0, L_PITCH, N); pts[:, 1] = rng.uniform(0, W_PITCH, N)
        vel = rng.normal(0, SIGMA_V, (N, 2)); w = rng.choice(WEIGHT_SET, N)
        fad = 10.0 ** (rng.normal(0, 2.0, N) / 20.0)
    x_s, x_c = reposition_pas(pts)
    base = build_base_features(pts, vel, w, fad, x_c)

    sens_pen = np.array([SENSING_PENALTY_FACTOR * max(0.0, GAMMA_MIN_DB - sensing_snr_db(pts[i], x_s)) for i in range(N)], dtype=np.float32)
    
    return dict(N=N, pts=pts, vel=vel, w=w, fad=fad, x_s=x_s, x_c=x_c, base=base, sens_pen=sens_pen, P_groups=max(1, N // 2))

def pair_reward(scene, u1, u2, k, pool):
    """Immediate reward of forming (u1,u2) at alpha level k, consuming PAs from `pool`.
       reward = pair_utility(alpha) - rate_penalty(alpha) - sensing_penalty(u1,u2)   (QoS kept from v2)
    Returns (reward, pure_semantic_utility, pool_after)."""
    pts, fad, w, x_c, P, sp = scene['pts'], scene['fad'], scene['w'], scene['x_c'], scene['P_groups'], scene['sens_pen']
    a_w = float(ALPHA_LEVELS[k])
    gc = 0.5 * (pts[u1, 0] + pts[u2, 0])
    xc_sel, pool_after = assign_comm_pas(gc, list(pool), x_c, max(1, N_C // P))
    Pm = P_C / (len(xc_sel) * P)
    util = float(pair_utility_at_alpha(u1, u2, pts, xc_sel, Pm, fad, w, a_w))
    rpen = float(rate_penalty(u1, u2, pts, xc_sel, Pm, fad, a_w))
    reward = util - rpen - float(sp[u1] + sp[u2])
    return reward, util, pool_after


# ddqn updating
def ddqn_update(net, target_net, memory, optimizer, batch_size, gamma, mse):
    """One Double-DQN gradient step:
        a* = argmax_a' Q_online(s', a')     (select the next action with the online net)
        y  = r + gamma * Q_target(s', a*)   (evaluate it with the target net); y = r if terminal
        loss = MSE(Q_online(s,a), y)        (fit the prediction to that bootstrapped target)"""
    batch = memory.sample(batch_size)
    s_in = torch.tensor(np.stack([b[0] for b in batch]))                     # (B, IN)
    ks = torch.tensor([b[1] for b in batch], dtype=torch.long)               # chosen alpha idx
    rs = torch.tensor([b[2] for b in batch], dtype=torch.float32)            # reward
    q_sa = net(s_in).gather(1, ks[:, None]).squeeze(1)                       # Q_online(s,a)  (B,)

    # gather every candidate (pair, alpha) row for the non-terminal next states, batched into one tensor
    seg, all_rows = [], []
    for b in batch:
        if b[6] >= 1.0:                       # done -> no bootstrap
            seg.append(None); continue
        base, rem, pool = b[3], b[4], b[5]
        avail = [i for i, r in enumerate(rem) if r]
        cand = [(avail[a], avail[c]) for a in range(len(avail)) for c in range(a + 1, len(avail))]
        ctx = build_context(base, rem, pool)
        rows = cand_rows(base, ctx, cand)
        start = len(all_rows); all_rows.extend(rows); seg.append((start, start + len(rows)))

    y = rs.clone()
    if all_rows:
        R = torch.tensor(np.stack(all_rows).astype(np.float32))
        with torch.no_grad():
            qo = net(R)          # online, for SELECTION
            qt = target_net(R)   # target, for EVALUATION
        for i in range(len(batch)):
            if seg[i] is None:
                continue
            s, e = seg[i]
            flat = int(torch.argmax(qo[s:e].reshape(-1)))
            ci, ki = divmod(flat, K_ALPHA)
            y[i] = rs[i] + gamma * qt[s:e][ci, ki]

    loss = mse(q_sa, y.detach())
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())

# 1 training episode 
"""Each step: build the state (candidate pairs + context); pick a (pair, alpha) either at RANDOM
    with prob eps (explore) or by the net's argmax (exploit); collect that pair's reward; remove the
    two users and consume their comm PAs; store the transition; and take one Double-DQN gradient step.
    eps decays across episodes (explore a lot early, trust the learned policy later).
    Returns (cumulative pure semantic utility of the scene, last loss)."""
def train_one_episode(net, target_net, memory, optimizer, eps, scene, batch_size, gamma, mse):
    base, N, P = scene['base'], scene['N'], scene['P_groups']
    remaining = [True] * N
    pool = list(range(N_C))
    cum_util, loss = 0.0, None
    while sum(remaining) >= 2:
        ctx = build_context(base, remaining, pool)
        avail = [i for i, r in enumerate(remaining) if r]
        cand = [(avail[a], avail[b]) for a in range(len(avail)) for b in range(a + 1, len(avail))]
        if np.random.rand() < eps:                                   # explore
            ci = np.random.randint(len(cand)); ki = np.random.randint(K_ALPHA)
        else:                                                        # exploit
            ci, ki, _ = greedy_action(net, base, ctx, cand)
        u1, u2 = cand[ci]
        s_input = np.concatenate([pair_input(base, u1, u2), ctx]).astype(np.float32)
        reward, util, pool_after = pair_reward(scene, u1, u2, ki, pool)
        cum_util += util
        remaining_after = list(remaining); remaining_after[u1] = False; remaining_after[u2] = False
        done = 1.0 if sum(remaining_after) < 2 else 0.0
        memory.push(s_input, ki, reward, base, remaining_after, pool_after, done)
        remaining, pool = remaining_after, pool_after
        if len(memory) >= batch_size:
            loss = ddqn_update(net, target_net, memory, optimizer, batch_size, gamma, mse)
    return cum_util, loss


# total utility for a pair
def pairs_total_utility(pairs, points, fading, weights, x_c, P_groups, alpha_mode="given"):
    pool, tot = list(range(N_C)), 0.0
    for pr in pairs:
        u, v = pr[0], pr[1]
        gc = 0.5 * (points[u, 0] + points[v, 0])
        xc_sel, pool = assign_comm_pas(gc, pool, x_c, max(1, N_C // P_groups))
        Pm = P_C / (len(xc_sel) * P_groups)
        if alpha_mode == "closed":
            a_w = closed_form_alpha(u, v, points, xc_sel, Pm, fading)
        elif alpha_mode == "best":
            _, k, _ = best_alpha(u, v, points, xc_sel, Pm, fading, weights); a_w = float(ALPHA_LEVELS[k])
        else:
            a_w = pr[2]
        tot += pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w)
    return tot

# average semantic utility PER USER = total utility / served users
def pairs_utility_per_user(pairs, points, fading, weights, x_c, P_groups, alpha_mode="given"):
    n_users = 2 * len(pairs)
    if n_users == 0:
        return 0.0
    return pairs_total_utility(pairs, points, fading, weights, x_c, P_groups, alpha_mode) / n_users


def greedy_pairs(points, fading, weights, x_c, P_groups):
    """Exhaustive greedy grouping with CLOSED-FORM alpha (the strong, short-sighted baseline to beat)."""
    remaining = set(range(len(points)))
    pairs = []
    while len(remaining) >= 2:
        best, bp = -1e18, None
        pool = list(range(N_C))
        for (u, v) in combinations(remaining, 2):
            gc = 0.5 * (points[u, 0] + points[v, 0])
            picks = sorted(pool, key=lambda n: abs(x_c[n] - gc))[:max(1, N_C // P_groups)]
            mm = max(1, len(picks))
            xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * DMIN, 0.0, L_PITCH)
            Pm = P_C / (mm * P_groups)
            a_w = closed_form_alpha(u, v, points, xc_sel, Pm, fading)
            sc = pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w)
            if sc > best:
                best, bp = sc, (u, v)
        pairs.append(bp); remaining.discard(bp[0]); remaining.discard(bp[1])
    return pairs


def periodic_eval(net, n_scenes=50, K=8, seed=2024):
    """Grade the frozen Q-policy vs greedy vs random on fixed scenes. Returns avg semantic utility/user."""
    rng = np.random.default_rng(seed); P = K // 2
    d_tot = g_tot = r_tot = 0.0
    for _ in range(n_scenes):
        scene = make_scene(K, rng)
        base, pts, fad, w, x_c = scene['base'], scene['pts'], scene['fad'], scene['w'], scene['x_c']
        dp = ddqn_form_pairs(net, base, [True] * K, scene['x_s'], x_c)
        gp = greedy_pairs(pts, fad, w, x_c, P)
        perm = rng.permutation(K); rp = [(perm[2 * i], perm[2 * i + 1]) for i in range(K // 2)]
        d_tot += pairs_utility_per_user(dp, pts, fad, w, x_c, P, "given")
        g_tot += pairs_utility_per_user(gp, pts, fad, w, x_c, P, "closed")
        r_tot += pairs_utility_per_user(rp, pts, fad, w, x_c, P, "closed")
    return d_tot / n_scenes, g_tot / n_scenes, r_tot / n_scenes


# ===================================================================================
#  TRAINING ENTRY POINT   (run `python train.py` to train the agent)
# ===================================================================================
# Trains the JointQNet agent with a real Double-DQN, purely from sampled rewards + Bellman
# bootstrapping (no analytic labels, no supervised shortcut). Saves -> NEW_isac_pass_noma_ddqn.pth
#
# Per episode: build a random scene, pair it off end-to-end while collecting transitions and taking a
# DDQN gradient step per pair (tv.train_one_episode -> tv.ddqn_update), decay epsilon, and every
# EVAL_EVERY episodes grade the frozen policy vs greedy/random - keeping the BEST-ratio checkpoint so
# a late unlucky episode can't overwrite a good model. Ends with a 200-scene held-out verification.
# Env vars: EPISODES (6000), EVAL_EVERY (500), LR (3e-4), TARGET_SYNC (250, target-net refresh), GAMMA (0.99)
def train():
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    EPISODES = int(os.environ.get("EPISODES", "20000"))
    EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
    lr = float(os.environ.get("LR", "3e-4"))
    target_sync = int(os.environ.get("TARGET_SYNC", "250"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    batch_size, sel_scenes = 128, 50
    eps, eps_end, eps_decay = 1.0, 0.05, 0.9997

    net = JointQNet()                      # the ONLINE Q-net (the one we train and keep)
    target_net = JointQNet()               # slow copy, used only to form the bootstrap targets
    target_net.load_state_dict(net.state_dict()); target_net.eval()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    memory = ReplayBuffer(50000)           # replay of past transitions, sampled in random minibatches
    mse = nn.MSELoss()

    print("=" * 64)
    print(f" PURE DDQN (from scratch) | {EPISODES} episodes | gamma {gamma} | lr {lr}")
    print(f" alpha grid: {ALPHA_LEVELS.tolist()}")
    print("=" * 64)

    best_ratio, reward_log, eval_log = -1.0, [], []
    for ep in range(1, EPISODES + 1):
        N = random.choice([2, 4, 6, 8, 10, 12])
        scene = make_scene(N)
        cum, _ = train_one_episode(net, target_net, memory, optimizer, eps, scene, batch_size, gamma, mse)
        reward_log.append(cum)
        if ep % target_sync == 0:                       # periodically refresh the target net
            target_net.load_state_dict(net.state_dict())
        eps = max(eps_end, eps * eps_decay)             # explore a little less each episode

        if ep % EVAL_EVERY == 0 or ep == 1:
            net.eval(); d, g, r = periodic_eval(net, sel_scenes); net.train()
            eval_log.append((ep, d, g, r)); flag = ""
            if g > 0 and d / g > best_ratio:            # keep only the best-vs-greedy checkpoint so far
                best_ratio = d / g
                torch.save(net.state_dict(), MODEL_PATH); flag = " *saved"
            print(f"ep {ep:5d}/{EPISODES} | eps {eps:.3f} | EVAL avg util/user  "
                  f"DDQN {d:6.3f}  Greedy {g:6.3f}  Random {r:6.3f}  | DDQN/Greedy {100*d/max(g,1e-9):5.1f}%{flag}")

    np.save("training_log.npy", np.array(reward_log, dtype=np.float32))
    np.save("eval_curve.npy", np.array(eval_log, dtype=np.float32))
    # reload the BEST saved checkpoint and verify it on a large, different-seed held-out sample
    net.load_state_dict(torch.load(MODEL_PATH)); net.eval()
    d, g, r = periodic_eval(net, 200, seed=777)
    print("=" * 64)
    print(f" DONE -> {MODEL_PATH} | best(sel) {100*best_ratio:.1f}%")
    print(f" FINAL 200-scene held-out avg util/user: DDQN {d:.3f}  Greedy {g:.3f}  Random {r:.3f}"
          f"  | DDQN/Greedy {100*d/max(g,1e-9):.1f}%  DDQN/Random {100*d/max(r,1e-9):.1f}%")
    print("=" * 64)


if __name__ == '__main__':
    train()
