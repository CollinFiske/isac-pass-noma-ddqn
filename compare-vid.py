# Side-by-side animation: the SAME moving users paired two ways - trained JointQNet vs exhaustive greedy.
# Both see identical scenes each slot, so any gap is purely the pairing + power-split decisions.
# Loads NEW_isac_pass_noma_ddqn.pth (override with MODEL_PATH=...).
import os, sys, importlib.util
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
import numpy as np, torch
import matplotlib
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

spec = importlib.util.spec_from_file_location("tv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "training-vid.py"))
tv = importlib.util.module_from_spec(spec); sys.modules["tv"] = tv; spec.loader.exec_module(tv)


def run_comparison():
    np.random.seed(7)
    try:
        net = tv.JointQNet()
        net.load_state_dict(torch.load(tv.MODEL_PATH)); net.eval()
    except Exception as e:
        print("Run training first.", e); return
    print(f"--- DDQN ({tv.MODEL_PATH}) vs Greedy (exhaustive + closed-form alpha) ---")

    N, STEPS, SUBSTEPS = 8, 20, 60; P_groups = N // 2
    pts = np.zeros((N, 3))
    pts[:, 0] = np.random.uniform(0, tv.L_PITCH, N); pts[:, 1] = np.random.uniform(0, tv.W_PITCH, N)
    mean_vel = np.random.uniform(-1.2, 1.2, (N, 2)); vel = mean_vel.copy()
    w = np.random.choice(tv.WEIGHT_SET, N); fad = tv.shadowing((N,))
    d_cum = g_cum = 0.0
    hist = []
    for step in range(STEPS):
        x_s, x_c = tv.reposition_pas(pts)
        base = tv.build_base_features(pts, vel, w, fad, x_c)
        dp = tv.ddqn_form_pairs(net, base, [True] * N, x_s, x_c)     # learned grouping + learned alpha
        gp = tv.greedy_pairs(pts, fad, w, x_c, P_groups)             # exhaustive grouping + closed-form alpha
        d_s = tv.pairs_utility_per_user(dp, pts, fad, w, x_c, P_groups, "given")
        g_s = tv.pairs_utility_per_user(gp, pts, fad, w, x_c, P_groups, "closed")
        d_cum += d_s; g_cum += g_s
        hist.append((pts.copy(), dp, gp, vel.copy(), x_s.copy(), x_c.copy(), d_cum, g_cum, d_s, g_s))
        print(f"slot {step+1:2d} | util/user: DDQN {d_s:6.3f}  Greedy {g_s:6.3f} | "
              f"cumulative/user: DDQN {d_cum:7.3f}  Greedy {g_cum:7.3f} | DDQN/Greedy {100*d_cum/max(g_cum,1e-9):5.1f}%")
        vel = tv.step_mobility(pts, vel, mean_vel, n_sub=SUBSTEPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    if not HEADLESS:
        fig.canvas.manager.set_window_title('v3 DDQN vs Greedy')
    cols = ['#e63946', '#2a9d8f', '#9b5de5', '#f4a261', '#2ec4b6', '#e0aaff']

    def update(frame):
        ax1.clear(); ax2.clear()
        p, dp, gp, v, xs, xc, dc, gcm, ds, gs = hist[frame]
        for ax, title, pairs, cum, stp in [(ax1, "DDQN (learned grouping+alpha)", dp, dc, ds),
                                           (ax2, "Greedy (closed-form alpha)", gp, gcm, gs)]:
            ax.set_xlim(-5, tv.L_PITCH + 5); ax.set_ylim(-5, tv.W_PITCH + 5)
            ax.set_title(title, fontsize=12, fontweight='bold'); ax.grid(True, ls=':', alpha=0.6)
            ax.plot([xs.min(), xs.max()], [tv.Y_S, tv.Y_S], 'k-', lw=2)
            ax.plot([xc.min(), xc.max()], [tv.Y_C, tv.Y_C], 'k-', lw=2)
            ax.scatter(xs, np.full_like(xs, tv.Y_S), color='darkorange', marker='^', s=35, label="Sensing PAs")
            ax.scatter(xc, np.full_like(xc, tv.Y_C), color='deepskyblue', marker='s', s=30, label="Comm PAs")
            for i in range(N):
                ax.scatter(p[i, 0], p[i, 1], c='royalblue', s=80, edgecolors='black', lw=0.5, zorder=5,
                           label="Users" if i == 0 else None)
                ax.text(p[i, 0] + 1, p[i, 1] + 1, f"U{i}", fontsize=9, fontweight='bold')
            ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color='gray', alpha=0.5, scale=25, width=0.003)
            for k, pr in enumerate(pairs):
                u, vv = pr[0], pr[1]
                ax.plot([p[u, 0], p[vv, 0]], [p[u, 1], p[vv, 1]], color=cols[k % len(cols)], lw=2, alpha=0.8)
            ax.legend(loc='lower right', fontsize=8)
            ax.text(0.03, 0.96, f"Slot {frame+1}/{STEPS}\nAvg util/user: {stp:.3f}\nCumulative/user: {cum:.3f}",
                    transform=ax.transAxes, fontsize=10, family='monospace', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=800, repeat=False)
    plt.tight_layout()
    if HEADLESS:
        update(STEPS - 1); fig.savefig("compare_demo.png", dpi=110); print("Saved -> compare_demo.png")
    else:
        plt.show()


if __name__ == '__main__':
    run_comparison()
