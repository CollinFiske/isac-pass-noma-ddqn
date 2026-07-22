import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from itertools import combinations

device = torch.device("cpu")   # hardcoded CPU; switch to "cuda" if a GPU is available (the nets are tiny, CPU is fine)

# sys params (from Table 1 of overleaf paper)
# any params missing form table are not needed here, to be added later for the Sionna implementation (ex. cable length, tower height)
F_CARRIER, C_LIGHT = 28e9, 3e8
LAMBDA = C_LIGHT / F_CARRIER
N_EFF = 1.4
LAMBDA_G = LAMBDA / N_EFF
ETA = (LAMBDA / (4 * np.pi)) ** 2
B_SIG = 100e6              # signal bandwidth (Hz)
SIGMA2 = 2.5e-12           # noise power sigma^2 (already folds in the NF=8 dB receiver noise figure); replaces v1's N0
P_TOTAL, RHO = 1.0, 0.3    # total power budget, and RHO = fraction routed to sensing
P_S, P_C = RHO * P_TOTAL, (1.0 - RHO) * P_TOTAL   # -> sensing power P_s, comm power P_c
RCS = 1.0                  # target radar cross-section sigma_RCS (how strongly a user reflects)
T_INT = 0.04              # radar coherent integration time (s)
G_P = T_INT * B_SIG        # coherent processing gain G_p = T_int * B_sig (used in the sensing SNR, eq 14)
GAMMA_MIN_DB = 15.0        # min sensing-SNR QoS threshold Gamma_min in dB (eq 16)
R_MIN = 1.0                 # min per-user rate (bit/s/Hz), paper eq. c_rate
RATE_PENALTY_FACTOR = 10.0  # reward penalty per bit of R_min shortfall (tunable; retune by violation rate)
N_S, N_C = 8, 8            # 8 sensing + 8 comm pinching antennas (PAs), one bank per waveguide
DMIN = LAMBDA / 2          # minimum PA spacing = lambda/2
L_PITCH, W_PITCH, H_WG = 105.0, 68.0, 12.0   # service area 105 x 68 m; waveguides sit at height H_WG = 12 m
Y_S, Y_C = W_PITCH / 2.0 - 0.5, W_PITCH / 2.0 + 0.5   # y-lines of the sensing / comm waveguides, 1 m apart, mid-pitch
ALPHA_MIN, ALPHA_MAX = 0.55, 0.95   # legal NOMA power-split range for the weak user's share a_w
FID_A1, FID_A2, FID_C1, FID_C2 = 0.30, 0.98, 0.25, -0.8   # semantic-fidelity logistic params a1,a2,c1,c2 (eq 17)
WEIGHT_SET = np.array([0.3, 0.6, 0.7, 0.8, 1.0])   # discrete per-user semantic importance weights w_k
# Gauss-Markov mobility params (ported meaning from v1): ALPHA_V = velocity memory (how much this step's
# velocity depends on the previous step), SIGMA_V = std of the random-walk noise, DELTA_T = slot length (s).
ALPHA_V, SIGMA_V, DELTA_T = 0.9, 0.5, 0.04

# Learned power split: the agent chooses alpha_w from this discrete grid instead of the
# old closed-form eq (23). Greedy keeps the closed-form alpha, so a DDQN that learns the
# utility-optimal alpha (which the channel-ratio heuristic is not) exceeds greedy.
ALPHA_LEVELS = np.round(np.arange(ALPHA_MIN, ALPHA_MAX + 1e-9, 0.05), 2)  # 0.55..0.95
K_ALPHA = len(ALPHA_LEVELS)

N_MAX = 12
CHAN_DB_OFFSET = 90.0
# Per-user feature block used to build the compact two-user network inputs:
#   [pos_x, pos_y, vel_x, vel_y, fading, weight, chan_dB]
USER_FEAT = 7
PAIR_DIM = 2 * USER_FEAT   # the grouping/alpha nets score a pair from its two users' features
MODEL_PATH = "NEW_isac_pass_noma_ddqn.pth"

# ---------------------------------------------------------------------
#  PHYSICS (Table-1 channel model)
# ---------------------------------------------------------------------

# random per-user signal blockage - some users just have worse luck
def shadowing(size):
    return 10.0 ** (np.random.normal(0.0, 2.0, size) / 20.0)


# comm signal strength at a user, summed over its antennas - bigger = better channel = higher rate
#   d          = 3D distance from user to comm PA n (x, y across the pitch, H_WG in height)
#   h_{c,k,n}  = (sqrt(eta)/d) * e^{-j 2pi/lambda * d}   free-space path loss (~1/d) + through-the-air propagation phase (eq 4)
#   g_{c,n}    = e^{-j 2pi/lambda_g * x_c,n}             phase accrued INSIDE the waveguide before exiting port n (eq 5)
#   h_k        = fading_k * sum_n g_{c,n} * h_{c,k,n}    coherent sum over the group's assigned comm PAs
# eta = (lambda/4pi)^2 is the reference free-space gain; lambda_g = lambda/N_EFF is the guided wavelength.
def comm_pa_gain(pt, x_c_sel, fading_k):
    d = np.sqrt((pt[0] - x_c_sel) ** 2 + (pt[1] - Y_C) ** 2 + H_WG ** 2)
    h_c = (np.sqrt(ETA) / d) * np.exp(-1j * (2 * np.pi / LAMBDA) * d)     # eq (4): path loss + through-the-air phase
    g_c = np.exp(-1j * (2 * np.pi / LAMBDA_G) * x_c_sel)                  # eq (5): in-waveguide phase before the port
    return np.sum(g_c * h_c) * fading_k


# radar SNR for detecting a user (signal echoes back) - must stay above GAMMA_MIN_DB or reward is penalized.
# h_s, g_s are the same air/waveguide terms as the comm channel, but SQUARED because the pulse travels the
# path twice (PA -> user -> PA, a two-way echo):
#   beta_k  = sqrt(sigma_RCS/(4 pi eta)) * sum_n g_{s,n}^2 * h_{s,k,n}^2   reflection coefficient (eq 12)
#             (the 1/eta cancels one aperture factor that the two-way squaring would otherwise double-count)
#   Gamma_s = (P_s/N_s) * G_p * |beta_k|^2 / sigma^2                       sensing SNR (eq 14), G_p = processing gain
# Returned in dB; the QoS constraint (eq 16) requires Gamma_s >= GAMMA_MIN_DB (15 dB) or the reward is penalized.
def sensing_snr_db(pt, x_s):
    d = np.sqrt((pt[0] - x_s) ** 2 + (pt[1] - Y_S) ** 2 + H_WG ** 2)
    h_s = (np.sqrt(ETA) / d) * np.exp(-1j * (2 * np.pi / LAMBDA) * d)
    g_s = np.exp(-1j * (2 * np.pi / LAMBDA_G) * x_s)
    beta = np.sqrt(RCS / (4 * np.pi * ETA)) * np.sum((g_s ** 2) * (h_s ** 2))   # eq (12): two-way reflection coefficient
    gamma = (P_S / N_S) * G_P * (np.abs(beta) ** 2) / SIGMA2                    # eq (14): sensing SNR (G_p = processing gain)
    return 10.0 * np.log10(max(float(gamma), 1e-12))


# give a pair the m nearest unused comm antennas, and remove them so no other pair reuses them.
# This enforces the paper's exclusivity constraint that each comm PA serves at most one group (eq 6/7):
# a group gets the m = N_C // P PAs closest to its centroid, laid out at lambda/2 (DMIN) spacing.
def assign_comm_pas(group_cx, pool, x_c, m):
    picks = sorted(pool, key=lambda n: abs(x_c[n] - group_cx))[:m]
    for n in picks:
        pool.remove(n)
    mm = max(1, len(picks))
    xc_sel = np.clip(group_cx + (np.arange(mm) - mm / 2.0) * DMIN, 0.0, L_PITCH)
    return xc_sel, pool


# semantic quality as an S-curve of SINR: too low and the meaning breaks down, too high and it plateaus.
#   xi(gamma) = a1 + (a2 - a1) / (1 + exp(-(c1*gamma + c2)))   logistic semantic fidelity (eq 17)
#   a1 = FID_A1 = 0.30 (floor), a2 = FID_A2 = 0.98 (ceiling), c1 = FID_C1 = 0.25 (steepness), c2 = FID_C2 = -0.8 (offset)
def fidelity(gamma):
    return FID_A1 + (FID_A2 - FID_A1) / (1.0 + np.exp(-(FID_C1 * gamma + FID_C2)))


# NOMA: two users share one channel, split by power. Order them by channel gain |h|^2 -> the WEAK user
# (smaller gain) gets the larger power share a_w and eats the strong user's signal as interference; the
# STRONG user cancels the weak signal first via SIC, so it sees noise only.
#   |h|^2   = Pm * |comm_pa_gain|^2                     received power (per-PA power Pm already applied)
#   gamma_w = a_w |h_w|^2 / (a_s |h_w|^2 + sigma^2)     weak user, strong's power is interference (eq 10)
#   gamma_s = a_s |h_s|^2 / sigma^2                     strong user after SIC -> noise-limited (eq 11)
def pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w):
    hu2 = Pm * np.abs(comm_pa_gain(points[u], xc_sel, fading[u])) ** 2
    hv2 = Pm * np.abs(comm_pa_gain(points[v], xc_sel, fading[v])) ** 2
    if hu2 <= hv2:                                  # order weak/strong by channel gain (weak = smaller |h|^2)
        w_idx, s_idx, hw2, hs2 = u, v, hu2, hv2
    else:
        w_idx, s_idx, hw2, hs2 = v, u, hv2, hu2
    a_s = 1.0 - a_w                                # strong user's power share = 1 - weak's
    gamma_w = (a_w * hw2) / (a_s * hw2 + SIGMA2)   # eq (10): weak SINR (self signal vs strong-as-interference + noise)
    gamma_s = (a_s * hs2) / SIGMA2                 # eq (11): strong SINR (noise only, weak already cancelled by SIC)
    return w_idx, s_idx, gamma_w, gamma_s


# THE SCORE WE CARE ABOUT (eq 18): importance x semantic quality x rate, summed over both users
#   U = sum_{k in pair} w_k * xi(gamma_k) * log2(1 + gamma_k)
#       w_k         = per-user importance weight (WEIGHT_SET)
#       xi(gamma_k) = semantic fidelity (eq 17)
#       log2(1+g)   = Shannon achievable rate (bit/s/Hz)
# This is *semantic* utility (rate scaled by fidelity), not raw throughput - it is what every script reports.
def pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w):
    w_idx, s_idx, gamma_w, gamma_s = pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w)
    return (weights[w_idx] * fidelity(gamma_w) * np.log2(1.0 + gamma_w)
            + weights[s_idx] * fidelity(gamma_s) * np.log2(1.0 + gamma_s))


def rate_penalty(u, v, points, xc_sel, Pm, fading, a_w):
    """Soft penalty for violating the per-user min-rate constraint R_k >= R_MIN (eq. c_rate).
        penalty = RATE_PENALTY_FACTOR * sum_k max(0, R_MIN - log2(1 + gamma_k))   (R_MIN = 1 bit/s/Hz)
    Only shapes the training targets (like the sensing penalty); it is NOT added to the reported utility."""
    _, _, gamma_w, gamma_s = pair_sinrs(u, v, points, xc_sel, Pm, fading, a_w)
    return RATE_PENALTY_FACTOR * (max(0.0, R_MIN - np.log2(1.0 + gamma_w))
                                  + max(0.0, R_MIN - np.log2(1.0 + gamma_s)))


def best_alpha(u, v, points, xc_sel, Pm, fading, weights):
    """Return (pure semantic utility at the chosen alpha, chosen index, objective vector).
        objective(alpha) = pair_utility_at_alpha(alpha) - rate_penalty(alpha);  chosen alpha = argmax over ALPHA_LEVELS
    The per-alpha training objective is semantic utility minus the R_min rate penalty, so the
    chosen alpha (and its supervision) is QoS-aware; the scalar utility returned stays pure semantic.
    This is the analytic ORACLE that AlphaNet regresses toward - v2 has no Bellman/TD target."""
    utils = np.array([pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a)
                      for a in ALPHA_LEVELS], dtype=np.float32)
    rpen = np.array([rate_penalty(u, v, points, xc_sel, Pm, fading, a)
                     for a in ALPHA_LEVELS], dtype=np.float32)
    obj = utils - rpen
    k = int(np.argmax(obj))
    return float(utils[k]), k, obj


# the non-learned way to split power (greedy uses this) - decent, but not utility-optimal, so AlphaNet beats it.
#   a_w = clip( |h_s|^2 / (|h_w|^2 + |h_s|^2), 0.55, 0.95 )   channel-ratio heuristic (eq 23)
# It gives the WEAK user the dominant power share (55-95%); this is exactly the fixed rule v1 baked into its
# reward, so v1's agent could never out-do it. v2's AlphaNet learns a better, QoS-aware split instead.
def closed_form_alpha(u, v, points, xc_sel, Pm, fading):
    hu2 = Pm * np.abs(comm_pa_gain(points[u], xc_sel, fading[u])) ** 2
    hv2 = Pm * np.abs(comm_pa_gain(points[v], xc_sel, fading[v])) ** 2
    hw2, hs2 = (hu2, hv2) if hu2 <= hv2 else (hv2, hu2)
    return float(np.clip(hs2 / (hw2 + hs2 + 1e-30), ALPHA_MIN, ALPHA_MAX))   # eq (23): weak gets the dominant share


# antennas + power for ONE pair, ignoring what other pairs took (used for training targets)
def _pair_pa(points, u, v, x_c, P_groups):
    gc = 0.5 * (points[u, 0] + points[v, 0])
    picks = sorted(range(N_C), key=lambda n: abs(x_c[n] - gc))[:N_C // P_groups]
    mm = max(1, len(picks))
    xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * DMIN, 0.0, L_PITCH)
    return xc_sel, P_C / (mm * P_groups)


# the TRUE answer GroupingNet imitates: brute-forced value of pairing u,v - the net learns to guess it instantly
def pair_value(u, v, points, fading, weights, x_c, P_groups):
    xc_sel, Pm = _pair_pa(points, u, v, x_c, P_groups)
    _, _, obj = best_alpha(u, v, points, xc_sel, Pm, fading, weights)
    return float(obj.max())   # constraint-aware value of pairing u,v (semantic utility - R_min penalty)


# slide antennas along the waveguides to follow the crowd: sensing spread out, comm bunched at the center
def reposition_pas(active_pts):
    xmin, xmax = active_pts[:, 0].min(), active_pts[:, 0].max()
    if xmax - xmin < (N_S - 1) * DMIN:
        c = 0.5 * (xmin + xmax)
        xmin, xmax = c - (N_S - 1) * DMIN / 2, c + (N_S - 1) * DMIN / 2
    x_s = np.clip(np.linspace(xmin, xmax, N_S), 0.0, L_PITCH)
    cx = np.clip(np.mean(active_pts[:, 0]), 0.0, L_PITCH)
    x_c = np.clip(cx + (np.arange(N_C) - N_C / 2.0) * DMIN, 0.0, L_PITCH)
    return x_s, x_c


def step_mobility(points, velocities, mean_vel, n_sub=1):
    """Advance Gauss-Markov mobility (eq 1-2) by n_sub 40 ms slots, reflecting off the pitch edges.
    Per slot:  v(t) = ALPHA_V*v(t-1) + (1-ALPHA_V)*mean_vel + SIGMA_V*sqrt(1-ALPHA_V^2)*n,  n ~ N(0, I)
               p(t) = p(t-1) + v(t)*DELTA_T
    ALPHA_V = velocity memory (correlation to the previous step), SIGMA_V = random-walk noise std,
    DELTA_T = slot length (s). The demo animations call this with n_sub >> 1 so a displayed frame spans
    enough real time for users to visibly move (one slot alone moves them ~vel*0.04 m, imperceptibly)."""
    for _ in range(n_sub):
        noise = np.random.normal(0, 1, velocities.shape)
        velocities[:] = (ALPHA_V * velocities + (1.0 - ALPHA_V) * mean_vel
                         + SIGMA_V * np.sqrt(1.0 - ALPHA_V ** 2) * noise)
        points[:, 0] += velocities[:, 0] * DELTA_T
        points[:, 1] += velocities[:, 1] * DELTA_T
        for ax, hi in ((0, L_PITCH), (1, W_PITCH)):
            lo_hit, hi_hit = points[:, ax] < 0.0, points[:, ax] > hi
            velocities[lo_hit | hi_hit, ax] *= -1.0      # bounce back into the field
            mean_vel[lo_hit | hi_hit, ax] *= -1.0
            points[:, ax] = np.clip(points[:, ax], 0.0, hi)
    return velocities


# ---------------------------------------------------------------------
#  NETWORKS + compact pair encoding
#  GroupingNet: value of pairing two users -> scalar.  AlphaNet: utility of each alpha level.
#  Both take a compact [u1 feats || u2 feats] input so the regression is trivial to learn.
# ---------------------------------------------------------------------

def build_base_features(points, velocities, weights, fading, x_c):
    """Per-user feature rows [pos_x, pos_y, vel_x, vel_y, fading, weight, chan_dB], frozen during a pairing round.
    chan_dB = 10*log10(|comm_pa_gain|^2) + CHAN_DB_OFFSET hands the net each user's comm-channel strength
    DIRECTLY. This precomputed channel feature is what made grouping learnable: v1's flat MLP saw only raw
    positions (its 7th feature was an is_available bit) and could not infer channel quality, so it stayed at
    chance. Feeding chan_dB is the key fix (see memory `ddqn-not-beating-random`)."""
    N = len(points)
    base = np.zeros((N, USER_FEAT), dtype=np.float32)
    for i in range(N):
        g2 = np.abs(comm_pa_gain(points[i], x_c, fading[i])) ** 2
        base[i] = [points[i, 0], points[i, 1], velocities[i, 0], velocities[i, 1],
                   fading[i], weights[i], 10.0 * np.log10(g2 + 1e-30) + CHAN_DB_OFFSET]
    return base


# glue two users' 7 features into the single 14-number vector both nets take as input
def pair_input(base, i, j):
    return np.concatenate([base[i], base[j]]).astype(np.float32)


# WHICH USERS TO PAIR: in = 14 numbers (two users), out = 1 score for how good that pair is
class GroupingNet(nn.Module):
    def __init__(self, in_dim=PAIR_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


# HOW TO SPLIT THE POWER: same 14 numbers in, out = 9 scores (one per ALPHA_LEVELS), argmax wins
class AlphaNet(nn.Module):
    def __init__(self, in_dim=PAIR_DIM, k_alpha=K_ALPHA):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k_alpha))

    def forward(self, x):
        return self.net(x)


# rolling memory of past (input, correct answer) examples - batches are sampled randomly so
# consecutive, near-identical examples don't bias the gradient
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *item):
        self.buffer.append(item)

    def sample(self, batch_size):
        return list(zip(*random.sample(self.buffer, batch_size)))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------
#  POLICY HELPERS (shared by training, periodic eval, and the demo scripts)
# ---------------------------------------------------------------------

def score_candidates(gnet, base, remaining, current, cand, x_s, x_c):
    """Grouping score for pairing the anchor (current[0]) with each candidate j."""
    # one batched forward pass scores every candidate at once (remaining/x_s/x_c unused, kept for callers)
    anchor = current[0]
    inp = np.stack([pair_input(base, anchor, j) for j in cand])
    with torch.no_grad():
        return gnet(torch.tensor(inp, dtype=torch.float32)).squeeze(1).numpy()


# ask AlphaNet for its 9 scores and return the index of the best alpha level
def alpha_pick(anet, base, u, v):
    with torch.no_grad():
        q = anet(torch.tensor(pair_input(base, u, v), dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return int(np.argmax(q))


def ddqn_form_pairs(gnet, anet, base, remaining_in, x_s, x_c):
    """Greedy (eps=0) policy: u1 by best-partner value, u2 by partner score, alpha by AlphaNet."""
    # THE INFERENCE LOOP the demos call - pure forward passes, no training or randomness
    remaining = list(remaining_in)
    pairs = []
    while sum(remaining) >= 2:
        avail = [i for i, r in enumerate(remaining) if r]
        best_v, u1 = -1e18, avail[0]
        for i in avail:   # u1 = whoever has the best available partner (most to gain)
            others = [j for j in avail if j != i]
            v = score_candidates(gnet, base, remaining, [i], others, x_s, x_c).max()
            if v > best_v:
                best_v, u1 = v, i
        others = [j for j in avail if j != u1]
        qs = score_candidates(gnet, base, remaining, [u1], others, x_s, x_c)
        u2 = others[int(np.argmax(qs))]                # u2 = u1's top-scoring partner
        ka = alpha_pick(anet, base, u1, u2)            # then AlphaNet sets their power split
        pairs.append((u1, u2, float(ALPHA_LEVELS[ka])))
        remaining[u1] = remaining[u2] = False          # both are taken, loop on the rest
    return pairs


# score a whole set of pairs; alpha_mode = where the power split comes from:
# "given" = the DDQN's own choice, "closed" = the formula (baselines), "best" = brute-forced upper bound
def pairs_total_utility(pairs, points, fading, weights, x_c, P_groups, alpha_mode="given"):
    pool, tot = list(range(N_C)), 0.0
    for pr in pairs:
        u, v = pr[0], pr[1]
        gc = 0.5 * (points[u, 0] + points[v, 0])
        xc_sel, pool = assign_comm_pas(gc, pool, x_c, N_C // P_groups)
        Pm = P_C / (len(xc_sel) * P_groups)
        if alpha_mode == "closed":
            a_w = closed_form_alpha(u, v, points, xc_sel, Pm, fading)
        elif alpha_mode == "best":
            _, k, _ = best_alpha(u, v, points, xc_sel, Pm, fading, weights); a_w = float(ALPHA_LEVELS[k])
        else:
            a_w = pr[2]
        tot += pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w)
    return tot


def pairs_utility_per_user(pairs, points, fading, weights, x_c, P_groups, alpha_mode="given"):
    """AVERAGE semantic utility PER USER = total semantic utility of all pairs / number of
    served users. This is what the demos, eval, and learning curve report (not the raw total)."""
    n_users = 2 * len(pairs)
    if n_users == 0:
        return 0.0
    return pairs_total_utility(pairs, points, fading, weights, x_c, P_groups, alpha_mode) / n_users


def greedy_pairs(points, fading, weights, x_c, P_groups):
    """Exhaustive greedy grouping with CLOSED-FORM alpha (fixed baseline)."""
    # THE BASELINE TO BEAT - no learning: take the best pair, repeat with whoever is left.
    # Strong but short-sighted (an early grab can strand two bad users together).
    remaining = set(range(len(points)))
    pairs = []
    while len(remaining) >= 2:
        best, bp = -1e18, None
        pool = list(range(N_C))
        for (u, v) in combinations(remaining, 2):
            gc = 0.5 * (points[u, 0] + points[v, 0])
            picks = sorted(pool, key=lambda n: abs(x_c[n] - gc))[:N_C // P_groups]
            mm = max(1, len(picks))
            xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * DMIN, 0.0, L_PITCH)
            Pm = P_C / (mm * P_groups)
            a_w = closed_form_alpha(u, v, points, xc_sel, Pm, fading)
            sc = pair_utility_at_alpha(u, v, points, xc_sel, Pm, fading, weights, a_w)
            if sc > best:
                best, bp = sc, (u, v)
        pairs.append(bp); remaining.discard(bp[0]); remaining.discard(bp[1])
    return pairs


# report card during training: DDQN vs greedy vs random on the same fixed scenes (fixed seed = comparable)
def periodic_eval(gnet, anet, n_scenes=50, K=8, seed=2024):
    rng = np.random.default_rng(seed)
    P_groups = K // 2
    d_tot = g_tot = r_tot = 0.0
    for _ in range(n_scenes):
        pts = np.zeros((K, 3))
        pts[:, 0] = rng.uniform(0, L_PITCH, K); pts[:, 1] = rng.uniform(0, W_PITCH, K)
        vel = rng.normal(0, SIGMA_V, (K, 2)); w = rng.choice(WEIGHT_SET, K)
        fad = 10.0 ** (rng.normal(0, 2.0, K) / 20.0)
        x_s, x_c = reposition_pas(pts)
        base = build_base_features(pts, vel, w, fad, x_c)
        dp = ddqn_form_pairs(gnet, anet, base, [True] * K, x_s, x_c)
        gp = greedy_pairs(pts, fad, w, x_c, P_groups)
        perm = rng.permutation(K); rp = [(perm[2 * i], perm[2 * i + 1]) for i in range(K // 2)]
        d_tot += pairs_utility_per_user(dp, pts, fad, w, x_c, P_groups, "given")
        g_tot += pairs_utility_per_user(gp, pts, fad, w, x_c, P_groups, "closed")
        r_tot += pairs_utility_per_user(rp, pts, fad, w, x_c, P_groups, "closed")
    return d_tot / n_scenes, g_tot / n_scenes, r_tot / n_scenes   # avg semantic utility per user


# ---------------------------------------------------------------------
#  TRAINING
# ---------------------------------------------------------------------

# MAIN TRAINING LOOP - one episode = one random scene of N users, paired start to finish.
# The physics above can compute the TRUE best pair/alpha exactly, just too slowly to use online,
# so we compute those answers and the nets regress toward them (MSE) until they can guess instantly.
# Despite the "DDQN" name this is supervised regression - no target net, no bootstrapping.
# (v1 used a real Double-DQN Bellman target  y = r + gamma * Q_target(s', argmax_a' Q_policy(s', a'))  with a
#  separate target network synced every N steps. v2 has NONE of that: the regression targets below - pair_value
#  for grouping and best_alpha's objective vector for alpha - ARE the exact analytic answers, so we just do MSE.)
# Epsilon 1.0 -> 0.05: explore randomly early for variety, use the nets' own choices later.
def train():
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    PENALTY_FACTOR = 1.0
    episodes = int(os.environ.get("EPISODES", "12000"))
    lr = 5e-4
    batch_size = 128
    eps, eps_end, eps_decay = 1.0, 0.05, 0.9997
    eval_every = int(os.environ.get("EVAL_EVERY", "500"))
    sel_scenes = 50          # robust best-checkpoint selection (12 was too noisy)
    best_ratio = -1.0

    gnet, anet = GroupingNet(), AlphaNet()
    g_opt = optim.Adam(gnet.parameters(), lr=lr)
    a_opt = optim.Adam(anet.parameters(), lr=lr)
    g_mem, a_mem = ReplayBuffer(50000), ReplayBuffer(50000)
    mse = nn.MSELoss()

    print("=" * 64)
    print(f" TRAINING pair-scoring DDQN + learned alpha | {episodes} episodes")
    print(f" alpha grid: {ALPHA_LEVELS.tolist()}")
    print("=" * 64)

    reward_log, eval_log = [], []

    for ep in range(1, episodes + 1):
        # --- build a fresh random scene: user count, positions, speeds, priorities, blockage ---
        N = random.choice([2, 4, 6, 8, 10, 12]); P_groups = N // 2
        pts = np.zeros((N, 3))
        pts[:, 0] = np.random.uniform(0, L_PITCH, N); pts[:, 1] = np.random.uniform(0, W_PITCH, N)
        vel = np.random.normal(0, SIGMA_V, (N, 2)); w = np.random.choice(WEIGHT_SET, N)
        fad = shadowing((N,))
        x_s, x_c = reposition_pas(pts)
        base = build_base_features(pts, vel, w, fad, x_c)

        remaining = [True] * N
        cum = 0.0
        sens_db = {i: sensing_snr_db(pts[i], x_s) for i in range(N)}
        pen = {i: PENALTY_FACTOR * max(0.0, GAMMA_MIN_DB - sens_db[i]) for i in range(N)}   # radar-QoS fine per user

        while sum(remaining) >= 2:   # pair off two users at a time until nobody is left
            avail = [i for i, r in enumerate(remaining) if r]
            # ---- pick u1: coin-flip on eps, either explore randomly or use GroupingNet ----
            if np.random.rand() < eps:
                u1 = int(np.random.choice(avail))
            else:
                best_v, u1 = -1e18, avail[0]
                for i in avail:
                    others = [j for j in avail if j != i]
                    v = score_candidates(gnet, base, remaining, [i], others, x_s, x_c).max()
                    if v > best_v:
                        best_v, u1 = v, i
            others = [j for j in avail if j != u1]
            # ---- pick u2: same explore-or-exploit choice for u1's partner ----
            if np.random.rand() < eps:
                u2 = int(np.random.choice(others))
            else:
                qs = score_candidates(gnet, base, remaining, [u1], others, x_s, x_c)
                u2 = others[int(np.argmax(qs))]

            xc_sel, Pm = _pair_pa(pts, u1, u2, x_c, P_groups)
            best_u, best_k, obj_vec = best_alpha(u1, u2, pts, xc_sel, Pm, fad, w)   # obj = utility - rate penalty

            # DENSE + advantage-centered grouping supervision: regress the (centered) value of
            # pairing u1 with EVERY candidate partner -> net learns the relative partner ranking.
            g_in, g_tg = [], []
            for j in others:
                g_in.append(pair_input(base, u1, j))
                g_tg.append(pair_value(u1, j, pts, fad, w, x_c, P_groups) - (pen[u1] + pen[j]))
            g_tg = np.array(g_tg, dtype=np.float32); g_tg -= g_tg.mean()
            for st, tg in zip(g_in, g_tg):
                g_mem.push(st, np.float32(tg))

            # DENSE alpha supervision: regress the QoS-aware objective (utility - R_min penalty)
            # at every alpha level, minus the sensing penalty.
            a_mem.push(pair_input(base, u1, u2), (obj_vec - (pen[u1] + pen[u2])).astype(np.float32))

            cum += best_u
            remaining[u1] = remaining[u2] = False

            # ---- one gradient step per net: sample a batch of stored examples, predict,
            #      measure the error against the true values, backprop ----
            if len(g_mem) >= batch_size:
                gs, gt = g_mem.sample(batch_size)
                loss_g = mse(gnet(torch.tensor(np.stack(gs), dtype=torch.float32)),
                             torch.tensor(np.array(gt), dtype=torch.float32).unsqueeze(1))
                g_opt.zero_grad(); loss_g.backward(); g_opt.step()
            if len(a_mem) >= batch_size:
                as_, av = a_mem.sample(batch_size)
                loss_a = mse(anet(torch.tensor(np.stack(as_), dtype=torch.float32)),
                             torch.tensor(np.stack(av), dtype=torch.float32))
                a_opt.zero_grad(); loss_a.backward(); a_opt.step()

        eps = max(eps_end, eps * eps_decay)   # explore a little less next episode
        reward_log.append(cum)

        # every eval_every episodes, grade the current nets and keep the checkpoint only if it
        # beats greedy by more than any previous one (so a late overfit run can't overwrite a good model)
        if ep % eval_every == 0 or ep == 1:
            gnet.eval(); anet.eval()
            d, g, r = periodic_eval(gnet, anet, n_scenes=sel_scenes)
            gnet.train(); anet.train()
            eval_log.append((ep, d, g, r))
            flag = ""
            if d / g > best_ratio:
                best_ratio = d / g
                torch.save({"grouping": gnet.state_dict(), "alpha": anet.state_dict()}, MODEL_PATH)
                flag = " *saved"
            print(f"ep {ep:5d}/{episodes} | eps {eps:.3f} | EVAL({sel_scenes}) avg util/user  "
                  f"DDQN {d:6.3f}  Greedy {g:6.3f}  Random {r:6.3f}  | DDQN/Greedy {100*d/g:5.1f}%{flag}")

    np.save("training_log.npy", np.array(reward_log, dtype=np.float32))
    np.save("eval_curve.npy", np.array(eval_log, dtype=np.float32))
    # FINAL held-out verification of the saved (best) model on a large, different-seed sample
    ck = torch.load(MODEL_PATH)
    gnet.load_state_dict(ck["grouping"]); anet.load_state_dict(ck["alpha"]); gnet.eval(); anet.eval()
    d, g, r = periodic_eval(gnet, anet, n_scenes=200, seed=777)
    print("=" * 64)
    print(f" DONE -> {MODEL_PATH} | best(sel) {100*best_ratio:.1f}%")
    print(f" FINAL saved-model check (200 held-out scenes) avg semantic utility/user: "
          f"DDQN {d:.3f}  Greedy {g:.3f}  Random {r:.3f}  | DDQN/Greedy {100*d/g:.1f}%  DDQN/Random {100*d/r:.1f}%")
    print("=" * 64)


if __name__ == '__main__':
    train()
