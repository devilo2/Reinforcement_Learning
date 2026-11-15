import shutil
import numpy as np
from common import exp_dirs, load_json, save_json, save_csv, check_same_config
from evaluate import run_episodes
from policy import load_actor, torch_policy

EXP, RES, MODELS = exp_dirs()
runs = load_json(RES / "training.json")["runs"]
check_same_config(runs)

models = []
for run in sorted(runs, key=lambda r: r["seed"]):
    actor = load_actor(run["checkpoint"])
    m = {k: run[k] for k in ("seed", "checkpoint")}
    for mode, s in (("argmax", False), ("sample", True)):
        pi = torch_policy(actor, s, run["seed"])
        rows = list(run_episodes(pi))
        r = np.array([row[1] for row in rows])
        save_csv(RES / f"eval_{run['seed']}_{mode}.csv", rows)
        m[mode] = {"mean": float(r.mean()), "sd": float(r.std())}
    models.append(m)

key = lambda m: (m["argmax"]["mean"], -m["seed"])
best = max(models, key=key)

summary = {}
for mode in ("argmax", "sample"):
    mr = [m[mode]["mean"] for m in models]
    ms = [m[mode]["sd"] for m in models]
    summary[mode] = {"mean": float(np.mean(mr)), "sd": float(np.std(mr)),
                     "episode_sd": float(np.mean(ms))}
    print(mode, f"{np.mean(mr):.1f} ± {np.std(mr):.1f}")

save_json(RES / "evaluation.json", {"models": models, "summary": summary,
          "best_seed": best["seed"], "best_checkpoint": best["checkpoint"]})
MODELS.mkdir(parents=True, exist_ok=True)
shutil.copy(best["checkpoint"], MODELS / "best.pt")
print("best seed:", best["seed"])
