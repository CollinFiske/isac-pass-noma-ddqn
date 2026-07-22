# CLAUDE.md — mobility-ddqn-v3 (Unified Joint DDQN)

Guidance for Claude Code when working in this directory.

## What this is

A reinforcement-learning study for a wireless paper (`../mobility-ddqn-v2/paper.tet`): an agent jointly
performs **user grouping** (which two mobile users share a NOMA group) and **NOMA power splitting**
(`alpha`) on a **dual-waveguide pinching-antenna ISAC** system under mobility, maximizing a **semantic**
utility subject to sensing- and rate-QoS constraints, benchmarked against exhaustive greedy.

**v3 is a REAL Double-DQN** — unlike v2 (which was supervised regression named "DDQN"). One network
`JointQNet` scores `(pair, alpha)` actions; the policy is a TD-bootstrapped Q-function with a target
network, trained **purely from experienced rewards + Bellman bootstrapping** (no analytic labels). It
keeps v2's two winning ideas — **learned alpha** and the **compact pair input + `chan_dB`**
representation — and adds a **remaining-pool context** vector and a **sequential comm-PA pool** so the
agent has genuine sequential lookahead.

## Commands

```bash
pip install torch numpy matplotlib
python pure-ddqn-training.py   # train the agent -> NEW_isac_pass_noma_ddqn.pth (+ eval_curve.npy, training_log.npy)
python eval-vid.py             # DDQN vs greedy vs random, learned-vs-closed alpha, SINR + R_min -> eval_results.png
python inference-vid.py        # animate learned pairing + power split
python compare-vid.py          # DDQN vs exhaustive-greedy, side by side
```

- Train first; the other three load `NEW_isac_pass_noma_ddqn.pth` and abort if missing.
- Seeds all RNGs to 42 (reproducible). Device hardcoded `cpu`.
- **Env vars**: `EPISODES` (6000), `EVAL_EVERY` (500), `LR` (3e-4), `GAMMA` (0.99), `TARGET_SYNC` (250);
  `MODEL_PATH` overrides the weights file the scripts load; `HEADLESS=1` saves PNGs.
- Python block-buffers redirected stdout — use `python -u` when logging a background run to a file.

## File layout — one shared module, one trainer

**`training-vid.py` is the single source of truth** (physics, features, `JointQNet`, DDQN helpers, policy,
eval). The trainer `pure-ddqn-training.py` and the three demo/eval scripts `importlib`-load it as `tv` and
call `tv.<fn>` — **do NOT re-duplicate physics/constants**; add shared logic to `training-vid.py`. The
trained model file is a **plain `JointQNet` state_dict** (NOT v2's `{"grouping","alpha"}` dict —
incompatible; retrain).

## The MDP / decision structure

- **Episode** = one pairing round over a frozen scene of `N` users (positions frozen during the round;
  mobility only advances between demo frames). Pairs form one at a time until `< 2` users remain.
- **State** = remaining users' features + current comm-PA pool. Encoded per candidate action as
  `[pair_input(base,i,j) (14) || build_context(base,remaining,pool) (6)]` = `IN_DIM = 20`.
  - `pair_input` = two users' 7 features `[pos_x,pos_y,vel_x,vel_y,fading,weight,chan_dB]` (`chan_dB` is the
    key learnability feature carried from v2).
  - `build_context` = `[#remaining/N_MAX, #free_PAs/N_C, mean/std chan_dB, mean weight, mean pos_x]` — the
    NEW piece that lets a compact pair-scoring net represent the bootstrapped future value.
- **Action** = `(pair (u1,u2), alpha_k in ALPHA_LEVELS)`. `JointQNet(state,pair) -> R^K_ALPHA` (9 values,
  one per alpha level). Grouping value of a pair = `max_k Q`; chosen alpha = `argmax_k Q`.
- **Reward** (`pair_reward`) = `pair_utility_at_alpha(alpha) - rate_penalty(alpha) - sensing_penalty(u1,u2)`.
- **Transition** = remove u1,u2 **and consume their comm PAs from a pool** (`assign_comm_pas`), so later
  pairs see a depleted pool — this makes the environment a true sequential decision process (v2 scored each
  pair in isolation; that myopia is why regression can't do lookahead).
- **Terminal** when `< 2` remain. **`gamma = 0.99`** (finite horizon; every user gets paired regardless).

## Learning rule (real Double-DQN — the whole point of v3)

`ddqn_update` (in `training-vid.py`):
```
a* = argmax_a' Q_online(s', a')          # select with online net
y  = r + gamma * Q_target(s', a*)        # evaluate with target net;  y = r if terminal
loss = MSE(Q_online(s,a), y)
```
Target network synced every `TARGET_SYNC` episodes; transition `ReplayBuffer` (cap 50000); `Adam`;
grad-clip norm 1.0; eps 1.0 -> 0.05 (`eps_decay 0.9997`). `train_one_episode` rolls out one scene
epsilon-greedy over `(pair,alpha)` and takes one gradient step per formed pair.

## The trainer

- **`pure-ddqn-training.py`** — trains from scratch, learning ONLY from sampled rewards + bootstrapping (no
  analytic labels). Every `EVAL_EVERY` episodes it grades the frozen policy on 50 fixed K=8 scenes
  (`periodic_eval`) and saves only the **best-ratio** checkpoint, so a late unlucky episode can't overwrite
  a good model. Ends with a **200-scene held-out** verification — trust that number, not the per-eval
  prints. Writes `NEW_isac_pass_noma_ddqn.pth`, `eval_curve.npy`, `training_log.npy`.

## Physics chain (all in `training-vid.py`, mapped to paper equations)

Unchanged from v2: `comm_pa_gain` (eq 4-5), `sensing_snr_db` (eq 12/14/16), `assign_comm_pas` (eq 6/7),
`fidelity` (eq 17), `pair_sinrs` (eq 10/11, SIC), `pair_utility_at_alpha` (eq 18), `rate_penalty`
(eq c_rate), `closed_form_alpha` (eq 23, greedy's split), `reposition_pas` (PA heuristic), `step_mobility`
(eq 1-2). `best_alpha` is the analytic oracle used ONLY by `eval-vid.py` (to grade the net's partner picks
against the true best) — the agent never trains against it.

## Reporting convention

All reported/plotted numbers are **average semantic utility PER USER** (`pairs_utility_per_user`), not
totals and not raw rate. `eval-vid.py` also reports learned-vs-closed alpha gain, grouping partner-selection
quality (0.5 = chance, via `pair_group_value`), comm/weak/strong/sensing SINR medians, %clearing
`Gamma_min`, and R_min satisfaction (DDQN vs greedy).

## Verified result

Trained pure DDQN reaches **~100-103% of greedy** and **~119-122% of random** on the 200-scene held-out
(learned alpha +~17% over closed-form; grouping partner-selection ~0.7). Beating exhaustive greedy from
scratch is the headline: the learned alpha plus the sequential-PA lookahead is what gets it there.

## How v3 differs from v1 / v2

- **v1** = real Double-DQN but a **monolithic** net over the whole flattened scene (stayed at chance) and
  **closed-form alpha** (couldn't beat greedy).
- **v2** = compact pair nets + **learned alpha**, but **supervised regression** (no target net / no
  bootstrapping) — myopic per pair.
- **v3** = **real Double-DQN** on the compact representation, **learned alpha as part of the action**, plus
  **context + sequential PA pool** for genuine lookahead. Best of both, at the cost of harder training.

## Caveats & known simplifications (inherited from v2)

- **PA positions are a heuristic, not optimized** (`reposition_pas`) — the biggest departure from the paper.
- **Comm SINR runs ~5 dB under the paper's targets** — operating-point matter, not a bug.
- **R_min is a SOFT penalty** -> not 100% satisfaction.
- **Demo mobility differs from training** (adds visible drift; training/eval use `v_bar=0`).
- Training from scratch is harder/noisier than v2's supervised regression: the periodic best-checkpoint
  selection is what protects the saved model from a late unlucky swing.

## Common tasks

- **Retrain**: rerun `pure-ddqn-training.py` (any change to feature layout, `JointQNet` shape, `ALPHA_LEVELS`,
  `CTX_DIM`, or physics constants **requires a retrain** — the checkpoint is a fixed-shape state_dict).
- **Tune lookahead vs stability**: `GAMMA`, `TARGET_SYNC`, `LR`, `EPISODES`.
- **Tune R_min enforcement**: `RATE_PENALTY_FACTOR` in `training-vid.py` (higher = more satisfaction, less
  utility) + retrain.
- **Add a state feature**: extend `build_base_features` and bump `USER_FEAT` (then `PAIR_DIM`/`IN_DIM`
  follow); or extend `build_context` and bump `CTX_DIM`. Retrain.
