import numpy as np
from common import exp_dirs, load_json, save_json, save_csv
from evaluate import run_episodes
from policy import load_actor, torch_policy

EXP, RES, MODELS = exp_dirs()
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2]
runs = load_json(RES / "training.json")["runs"]
actors = [load_actor(r["checkpoint"]) for r in runs]

cells = []
for s in SIGMAS:
    m = []
    for r, a in zip(runs, actors):
        rows = list(run_episodes(torch_policy(a), sigma=s))
        save_csv(RES / f"noise/{r['seed']}_sigma{s}.csv", rows)
        m.append(float(np.mean([row[1] for row in rows])))
    cells.append({"sigma": s, "mean": float(np.mean(m)), "sd": float(np.std(m))})
    print(f"sigma {s}: {np.mean(m):.1f} ± {np.std(m):.1f}", flush=True)

save_json(RES / "noise.json", {"cells": cells})
