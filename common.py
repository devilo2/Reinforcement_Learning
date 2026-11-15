
import argparse, csv, json, sys
from pathlib import Path

COLS = ["episode_seed", "return", "terminated", "truncated"]

def exp_dirs():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    exp = Path(ap.parse_known_args()[0].output)
    return exp, exp / "results", exp / "models"

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def load_json(path):
    return json.loads(Path(path).read_text())

def record_run(exp, info):
    save_json(exp / "models" / f"{info['run_name']}.json", info)
    merge_runs(exp)

def merge_runs(exp):
    runs = [load_json(p) for p in (exp / "models").glob("*.json")]
    runs.sort(key=lambda r: r["seed"])
    save_json(exp / "results" / "training.json", {"runs": runs})

def check_same_config(runs, skip=("seed", "output_dir")):
    cfg = [{k: v for k, v in r["config"].items() if k not in skip}
           for r in runs]
    seeds = {r["seed"] for r in runs}
    if len(runs) != 5 or len(seeds) != 5 or cfg.count(cfg[0]) != 5:
        sys.exit("stop: need 5 runs, 5 seeds, same settings")
