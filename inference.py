# Animated demo of a TRAINED JointQNet: users walk around and the net re-pairs them (with a learned
# power split) every slot. Inference only. Train first; loads NEW_isac_pass_noma_ddqn.pth (override with MODEL_PATH=...).
import os, sys, importlib.util
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
import numpy as np, torch
import matplotlib
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

spec = importlib.util.spec_from_file_location("tv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
tv = importlib.util.module_from_spec(spec); sys.modules["tv"] = tv; spec.loader.exec_module(tv)


def run_inference():
    np.random.seed(7)
    try:
        net = tv.JointQNet()
        net.load_state_dict(torch.load(tv.MODEL_PATH)); net.eval()   # ONE unified Q-net now
    except Exception as e:
        print("Train model first!", e); return
    print(f"--- INFERENCE using {tv.MODEL_PATH} ---")

    N, STEPS, SUBSTEPS = 8, 20, 60   # each frame advances 60x40ms = 2.4 s of mobility
    pts = np.zeros((N, 3))
    pts[:, 0] = np.random.uniform(0, tv.L_PITCH, N); pts[:, 1] = np.random.uniform(0, tv.W_PITCH, N)
    mean_vel = np.random.uniform(-1.2, 1.2, (N, 2)); vel = mean_vel.copy()
    w = np.random.choice(tv.WEIGHT_SET, N); fad = tv.shadowing((N,))

    hist = []
    for step in range(1, STEPS + 1):
        x_s, x_c = tv.reposition_pas(pts)
        base = tv.build_base_features(pts, vel, w, fad, x_c)
        # THE MODEL'S DECISION: ddqn_form_pairs picks argmax (pair, alpha) from JointQNet, repeatedly,
        # until all users are grouped -> returns [(u1, u2, alpha), ...]
        pairs = tv.ddqn_form_pairs(net, base, [True] * N, x_s, x_c)
        util_pu = tv.pairs_utility_per_user(pairs, pts, fad, w, x_c, N // 2, "given")
        print(f"[slot {step:2d}] avg util/user: {util_pu:.3f} | "
              + ", ".join(f"U{u}-U{v}(a={a:.2f})" for u, v, a in pairs))
        hist.append((pts.copy(), pairs, vel.copy(), x_s.copy(), x_c.copy(), util_pu))
        vel = tv.step_mobility(pts, vel, mean_vel, n_sub=SUBSTEPS)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not HEADLESS:
        fig.canvas.manager.set_window_title('v3 DDQN Inference')

    def update(frame):
        ax.clear()
        p, pairs, v, xs, xc, upu = hist[frame]
        ax.set_xlim(-5, tv.L_PITCH + 5); ax.set_ylim(-5, tv.W_PITCH + 5)
        ax.set_title("Unified Joint DDQN + ISAC + PASS", fontsize=12, fontweight='bold')
        ax.grid(True, ls=':', alpha=0.6)
        ax.plot([xs.min(), xs.max()], [tv.Y_S, tv.Y_S], 'k-', lw=2)
        ax.plot([xc.min(), xc.max()], [tv.Y_C, tv.Y_C], 'k-', lw=2)
        ax.scatter(xs, np.full_like(xs, tv.Y_S), color='darkorange', marker='^', s=35, label="Sensing PAs")
        ax.scatter(xc, np.full_like(xc, tv.Y_C), color='deepskyblue', marker='s', s=30, label="Comm PAs")
        for i in range(N):
            ax.scatter(p[i, 0], p[i, 1], c='royalblue', s=80, edgecolors='black', lw=0.5, zorder=5,
                       label="Users" if i == 0 else None)
            ax.text(p[i, 0] + 1, p[i, 1] + 1, f"U{i}", fontsize=9, fontweight='bold')
        ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color='gray', alpha=0.5, scale=25, width=0.003)
        cols = ['#e63946', '#2a9d8f', '#9b5de5', '#f4a261', '#2ec4b6', '#e0aaff']
        for k, (u, vv, a) in enumerate(pairs):
            ax.plot([p[u, 0], p[vv, 0]], [p[u, 1], p[vv, 1]], color=cols[k % len(cols)], lw=2, alpha=0.8)
            ax.text((p[u, 0] + p[vv, 0]) / 2, (p[u, 1] + p[vv, 1]) / 2, f"a={a:.2f}", fontsize=7, color=cols[k % len(cols)])
        ax.legend(loc='lower right', fontsize=8)
        ax.text(0.03, 0.96, f"Slot {frame+1}/{STEPS}\nAvg semantic util/user: {upu:.3f}", transform=ax.transAxes,
                fontsize=10, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=800, repeat=False)
    plt.tight_layout()
    if HEADLESS:
        update(STEPS - 1); fig.savefig("inference_demo.png", dpi=110); print("Saved -> inference_demo.png")
    else:
        plt.show()


if __name__ == '__main__':
    run_inference()
