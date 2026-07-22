# Trains the JointQNet agent with a real Double-DQN, purely from sampled rewards + Bellman
# bootstrapping (no analytic labels, no supervised shortcut). Saves -> NEW_isac_pass_noma_ddqn.pth
#
# Per episode: build a random scene, pair it off end-to-end while collecting transitions and taking a
# DDQN gradient step per pair (tv.train_one_episode -> tv.ddqn_update), decay epsilon, and every
# EVAL_EVERY episodes grade the frozen policy vs greedy/random - keeping the BEST-ratio checkpoint so
# a late unlucky episode can't overwrite a good model. Ends with a 200-scene held-out verification.
# Env vars: EPISODES (6000), EVAL_EVERY (500), LR (3e-4), TARGET_SYNC (250, target-net refresh), GAMMA (0.99)
import os, sys, importlib.util, random
import numpy as np, torch, torch.nn as nn, torch.optim as optim

spec = importlib.util.spec_from_file_location("tv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "training-vid.py"))
tv = importlib.util.module_from_spec(spec); sys.modules["tv"] = tv; spec.loader.exec_module(tv)


def main():
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    EPISODES = int(os.environ.get("EPISODES", "6000"))
    EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
    lr = float(os.environ.get("LR", "3e-4"))
    target_sync = int(os.environ.get("TARGET_SYNC", "250"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    batch_size, sel_scenes = 128, 50
    eps, eps_end, eps_decay = 1.0, 0.05, 0.9997

    net = tv.JointQNet()                      # the ONLINE Q-net (the one we train and keep)
    target_net = tv.JointQNet()               # slow copy, used only to form the bootstrap targets
    target_net.load_state_dict(net.state_dict()); target_net.eval()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    memory = tv.ReplayBuffer(50000)           # replay of past transitions, sampled in random minibatches
    mse = nn.MSELoss()

    print("=" * 64)
    print(f" PURE DDQN (from scratch) | {EPISODES} episodes | gamma {gamma} | lr {lr}")
    print(f" alpha grid: {tv.ALPHA_LEVELS.tolist()}")
    print("=" * 64)

    best_ratio, reward_log, eval_log = -1.0, [], []
    for ep in range(1, EPISODES + 1):
        N = random.choice([2, 4, 6, 8, 10, 12])
        scene = tv.make_scene(N)
        cum, _ = tv.train_one_episode(net, target_net, memory, optimizer, eps, scene, batch_size, gamma, mse)
        reward_log.append(cum)
        if ep % target_sync == 0:                       # periodically refresh the target net
            target_net.load_state_dict(net.state_dict())
        eps = max(eps_end, eps * eps_decay)             # explore a little less each episode

        if ep % EVAL_EVERY == 0 or ep == 1:
            net.eval(); d, g, r = tv.periodic_eval(net, sel_scenes); net.train()
            eval_log.append((ep, d, g, r)); flag = ""
            if g > 0 and d / g > best_ratio:            # keep only the best-vs-greedy checkpoint so far
                best_ratio = d / g
                torch.save(net.state_dict(), tv.MODEL_PATH); flag = " *saved"
            print(f"ep {ep:5d}/{EPISODES} | eps {eps:.3f} | EVAL avg util/user  "
                  f"DDQN {d:6.3f}  Greedy {g:6.3f}  Random {r:6.3f}  | DDQN/Greedy {100*d/max(g,1e-9):5.1f}%{flag}")

    np.save("training_log.npy", np.array(reward_log, dtype=np.float32))
    np.save("eval_curve.npy", np.array(eval_log, dtype=np.float32))
    # reload the BEST saved checkpoint and verify it on a large, different-seed held-out sample
    net.load_state_dict(torch.load(tv.MODEL_PATH)); net.eval()
    d, g, r = tv.periodic_eval(net, 200, seed=777)
    print("=" * 64)
    print(f" DONE -> {tv.MODEL_PATH} | best(sel) {100*best_ratio:.1f}%")
    print(f" FINAL 200-scene held-out avg util/user: DDQN {d:.3f}  Greedy {g:.3f}  Random {r:.3f}"
          f"  | DDQN/Greedy {100*d/max(g,1e-9):.1f}%  DDQN/Random {100*d/max(r,1e-9):.1f}%")
    print("=" * 64)


if __name__ == '__main__':
    main()
