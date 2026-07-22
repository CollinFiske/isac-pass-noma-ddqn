# Mobility-Aware ISAC-PASS-NOMA — Unified Joint DDQN 

**Double-DQN**. One network (`JointQNet`)
scores every `(pair, alpha)` action; Temporal difference (TD) bootstrapping with a target network chooses **both** which two
users share a NOMA group **and** their power split. Both v2 wins are kept: **alpha is a learned action**,
and the compact pair-input + `chan_dB` representation (plus a new remaining-pool context) keeps it
learnable. A sequential comm-PA pool gives the agent the **lookahead** greedy and v2-regression lack.

```bash
pip install torch numpy matplotlib

python pure-ddqn-training.py   # train the DDQN agent -> NEW_isac_pass_noma_ddqn.pth (+ eval_curve.npy, training_log.npy)
python eval-vid.py             # DDQN vs greedy vs random, learned-vs-closed alpha, partner quality, SINR + R_min -> eval_results.png
python inference-vid.py        # animate the learned pairing + power split over time
python compare-vid.py          # DDQN vs exhaustive-greedy, side by side
```

- Train first; the demos/eval load `NEW_isac_pass_noma_ddqn.pth` and abort if it is missing.
- `inference/compare/eval` import the shared physics + net + policy from `training-vid.py`.
- Env vars: `EPISODES`, `EVAL_EVERY`, `LR`, `GAMMA`, `TARGET_SYNC`; `HEADLESS=1` saves PNGs instead of
  opening windows; `MODEL_PATH=...` overrides the weights file the scripts load.

