# Mobility-Aware ISAC-PASS-NOMA DDQN Implementation

**Double-DQN**. One network (`JointQNet`) scores every `(pair, alpha)` action. 
Temporal difference (TD) bootstrapping with a target network chooses **both** which two
users share a NOMA group and their power split (alpha values). 

### to run: 
```bash
pip install torch numpy matplotlib

python train.py                # train the DDQN agent -> NEW_isac_pass_noma_ddqn.pth (+ eval_curve.npy, training_log.npy)
python evaluate-ddqn.py            # DDQN vs greedy vs random, learned-vs-closed alpha, partner quality, SINR + R_min -> eval_results.png
python inference.py               # animate the learned pairing + power split over time
python compare-ddqn-vs-greedy.py  # DDQN vs exhaustive-greedy, side by side
```

- Train first; then the demos/eval will load `NEW_isac_pass_noma_ddqn.pth` and abort if it is missing.
- `inference`/`compare-ddqn-vs-greedy`/`evaluate-ddqn` import the shared physics + nn + policy from `train.py`.
- Env vars: `EPISODES`, `EVAL_EVERY`, `LR`, `GAMMA`, `TARGET_SYNC`; `HEADLESS=1` saves PNGs instead of
  opening windows; `MODEL_PATH=...` overrides the weights file the scripts load.

