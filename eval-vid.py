"""Evaluation for the v3 Unified Joint DDQN (loads NEW_isac_pass_noma_ddqn.pth). Verifies:
  1. DDQN >= greedy (grouping reaches greedy; learned alpha surpasses it).
  2. Learning improved with episodes (plots eval_curve.npy).
  3. Learned power split beats the closed-form alpha on identical pairings.
Plus a partner-alignment diagnostic and Table-1 SINR / R_min stats.
"""
import os, sys, importlib.util
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("tv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "training-vid.py"))
tv = importlib.util.module_from_spec(spec); sys.modules["tv"] = tv; spec.loader.exec_module(tv)


def load_model(path):
    net = tv.JointQNet(); net.load_state_dict(torch.load(path)); net.eval()
    return net


def evaluate(net, tag, curve_path):
    rng = np.random.default_rng(123)
    K = 8; P_groups = K // 2; SCENES = 60
    d_u, g_u, r_u, learned_a_u, closed_a_u, partner_opt = [], [], [], [], [], []
    comm, weak, strong, sens, d_rate_ok, g_rate_ok = [], [], [], [], [], []

    for _ in range(SCENES):
        scene = tv.make_scene(K, rng)
        base, pts, fad, w, x_s, x_c = scene['base'], scene['pts'], scene['fad'], scene['w'], scene['x_s'], scene['x_c']
        dp = tv.ddqn_form_pairs(net, base, [True] * K, x_s, x_c)
        gp = tv.greedy_pairs(pts, fad, w, x_c, P_groups)
        perm = rng.permutation(K); rp = [(perm[2 * i], perm[2 * i + 1]) for i in range(K // 2)]

        d_u.append(tv.pairs_utility_per_user(dp, pts, fad, w, x_c, P_groups, "given"))
        g_u.append(tv.pairs_utility_per_user(gp, pts, fad, w, x_c, P_groups, "closed"))
        r_u.append(tv.pairs_utility_per_user(rp, pts, fad, w, x_c, P_groups, "closed"))
        # isolate the power-split win: same DDQN pairing, learned alpha vs closed-form alpha
        learned_a_u.append(tv.pairs_utility_per_user(dp, pts, fad, w, x_c, P_groups, "given"))
        closed_a_u.append(tv.pairs_utility_per_user([(u, v) for (u, v, _) in dp], pts, fad, w, x_c, P_groups, "closed"))

        # partner-selection quality: for one random user, does JointQNet's grouping value pick the best partner?
        ctx = tv.build_context(base, [True] * K, list(range(tv.N_C)))
        u1 = int(rng.integers(K)); others = [j for j in range(K) if j != u1]
        net_u2 = others[int(np.argmax([tv.pair_group_value(net, base, ctx, u1, j) for j in others]))]
        utils = []
        for j in others:
            gc = 0.5 * (pts[u1, 0] + pts[j, 0]); picks = sorted(range(tv.N_C), key=lambda n: abs(x_c[n] - gc))[:max(1, tv.N_C // P_groups)]
            mm = max(1, len(picks)); xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * tv.DMIN, 0, tv.L_PITCH)
            Pm = tv.P_C / (mm * P_groups); bu, _, _ = tv.best_alpha(u1, j, pts, xc_sel, Pm, fad, w); utils.append(bu)
        u_net = utils[others.index(net_u2)]; ub, uw = max(utils), min(utils)
        if ub > uw:
            partner_opt.append((u_net - uw) / (ub - uw))   # 1 = best partner, 0.5 = chance, 0 = worst

        # SINR + R_min stats: redo the sequential PA assignment like the utility function does
        pool = list(range(tv.N_C))
        for (u1, u2, a_w) in dp:
            gc = 0.5 * (pts[u1, 0] + pts[u2, 0]); picks = sorted(pool, key=lambda n: abs(x_c[n] - gc))[:max(1, tv.N_C // P_groups)]
            for n in picks: pool.remove(n)
            mm = max(1, len(picks)); xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * tv.DMIN, 0, tv.L_PITCH)
            Pm = tv.P_C / (mm * P_groups)
            _, _, gw, gs = tv.pair_sinrs(u1, u2, pts, xc_sel, Pm, fad, a_w)
            weak.append(10 * np.log10(gw)); strong.append(10 * np.log10(gs)); comm += [10 * np.log10(gw), 10 * np.log10(gs)]
            d_rate_ok += [np.log2(1 + gw) >= tv.R_MIN, np.log2(1 + gs) >= tv.R_MIN]
        for k in range(K):
            sens.append(tv.sensing_snr_db(pts[k], x_s))
        gpool = list(range(tv.N_C))
        for (u1, u2) in gp:
            gc = 0.5 * (pts[u1, 0] + pts[u2, 0]); picks = sorted(gpool, key=lambda n: abs(x_c[n] - gc))[:max(1, tv.N_C // P_groups)]
            for n in picks: gpool.remove(n)
            mm = max(1, len(picks)); xc_sel = np.clip(gc + (np.arange(mm) - mm / 2.0) * tv.DMIN, 0, tv.L_PITCH)
            Pm = tv.P_C / (mm * P_groups); a_w = tv.closed_form_alpha(u1, u2, pts, xc_sel, Pm, fad)
            _, _, gw, gs = tv.pair_sinrs(u1, u2, pts, xc_sel, Pm, fad, a_w)
            g_rate_ok += [np.log2(1 + gw) >= tv.R_MIN, np.log2(1 + gs) >= tv.R_MIN]

    d, g, r = np.mean(d_u), np.mean(g_u), np.mean(r_u)
    print("=" * 62)
    print(f" EVALUATION [{tag}]  (K={K}, {SCENES} scenes)")
    print("=" * 62)
    print(f" Avg semantic utility PER USER:  DDQN {d:.3f} | Greedy {g:.3f} | Random {r:.3f}")
    print(f"   DDQN vs Greedy: {100*d/max(g,1e-9):.1f}%   DDQN vs Random: {100*d/max(r,1e-9):.1f}%")
    print("-" * 62)
    print(f" Power split on identical DDQN pairings (avg semantic utility/user):")
    print(f"   learned alpha {np.mean(learned_a_u):.3f}  vs  closed-form alpha {np.mean(closed_a_u):.3f}"
          f"   (+{100*(np.mean(learned_a_u)/max(np.mean(closed_a_u),1e-9)-1):.1f}%)")
    print("-" * 62)
    print(f" Grouping partner-selection quality: {np.mean(partner_opt):.2f} of optimal (random=0.5)")
    print("-" * 62)
    print(f" comm median {np.median(comm):.2f} | weak {np.median(weak):.2f} | strong {np.median(strong):.2f}"
          f" | sensing {np.median(sens):.2f} dB | clear Gamma_min {100*np.mean(np.array(sens)>=tv.GAMMA_MIN_DB):.1f}%")
    print("-" * 62)
    print(f" R_min ({tv.R_MIN:.0f} bit/s/Hz) satisfaction:  DDQN {100*np.mean(d_rate_ok):.1f}%  vs  "
          f"Greedy {100*np.mean(g_rate_ok):.1f}%  of users")
    print("=" * 62)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if os.path.exists(curve_path):
        c = np.load(curve_path)
        if c.ndim == 2 and len(c):
            ax[0].plot(c[:, 0], c[:, 1], "-o", ms=3, label="DDQN", color="#2a9d8f")
            ax[0].plot(c[:, 0], c[:, 2], "--", label="Greedy", color="#e63946")
            ax[0].plot(c[:, 0], c[:, 3], ":", label="Random", color="#888")
            ax[0].set_xlabel("episode"); ax[0].set_ylabel("avg semantic utility per user")
            ax[0].set_title(f"Learning curve [{tag}]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].bar(["DDQN", "Greedy", "Random"], [d, g, r], color=["#2a9d8f", "#e63946", "#adb5bd"])
    ax[1].set_ylabel("avg semantic utility per user"); ax[1].set_title(f"Avg utility per user [{tag}] (K=8)")
    ax[1].grid(alpha=0.3, axis="y")
    out = "eval_results.png"
    plt.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved -> {out}")


def main():
    if not os.path.exists(tv.MODEL_PATH):
        print(f"No trained model at {tv.MODEL_PATH}. Run pure-ddqn-training.py first."); return
    try:
        net = load_model(tv.MODEL_PATH)
    except Exception as e:
        print(f"Could not load {tv.MODEL_PATH}: {e}"); return
    evaluate(net, "DDQN", "eval_curve.npy")


if __name__ == '__main__':
    main()
