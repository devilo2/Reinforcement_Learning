import argparse
import csv
import hashlib
import sys
from pathlib import Path

from common import load_json

ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True)
EXP = Path(ap.parse_args().output)
RES, MODELS = EXP / "results", EXP / "models"
BASE = EXP.parent / "baseline" / "results"
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2]
problems = []


def need(path):
    if not Path(path).exists():
        problems.append(f"파일 없음: {path}")
        return False
    return True


def check_csv(path):
    if not need(path):
        return
    rows = list(csv.DictReader(open(path)))
    if len(rows) != 100:
        problems.append(f"{path}: {len(rows)}행 (100행이어야 함)")
        return
    seeds = [int(r["episode_seed"]) for r in rows]
    if seeds != list(range(10000, 10100)):
        problems.append(f"{path}: episode_seed가 10000~10099가 아님")


# 베이스라인
for n in ("random", "noop"):
    check_csv(BASE / f"baseline_{n}.csv")

# 학습 기록
need(RES / "training.json")
if (RES / "training.json").exists():
    runs = load_json(RES / "training.json")["runs"]
    seeds = {r["seed"] for r in runs}
    skip = ("seed", "output_dir")
    cfg = [{k: v for k, v in r["config"].items() if k not in skip} for r in runs]
    if len(runs) != 5 or len(seeds) != 5:
        problems.append(f"training.json: run {len(runs)}개, seed {len(seeds)}개 (5개여야 함)")
    if cfg and cfg.count(cfg[0]) != len(cfg):
        problems.append("training.json: seed 외 설정이 서로 다름")
    for r in runs:
        need(r["checkpoint"])
        for mode in ("argmax", "sample"):
            check_csv(RES / f"eval_{r['seed']}_{mode}.csv")
        for s in SIGMAS:
            need(RES / "noise" / f"{r['seed']}_sigma{s}.csv")

# 결과 파일
for p in (RES / "evaluation.json", RES / "noise.json", RES / "export.json",
          RES / "conversion.json", RES / "observations.npy", RES / "demo_cases.csv",
          MODELS / "best.pt", MODELS / "policy.onnx", EXP.parent / "selected_experiment.json"):
    need(p)

# 노이즈 csv 개수
noise_files = list((RES / "noise").glob("*.csv")) if (RES / "noise").exists() else []
if len(noise_files) != 25:
    problems.append(f"noise/ csv {len(noise_files)}개 (25개여야 함)")

# 해시
if (RES / "export.json").exists() and (MODELS / "policy.onnx").exists():
    sha = hashlib.sha256((MODELS / "policy.onnx").read_bytes()).hexdigest()
    if load_json(RES / "export.json")["policy_sha256"] != sha:
        problems.append("export.json의 policy_sha256이 policy.onnx와 다름")

if problems:
    print("[검증 실패]")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("검증 통과: 모든 항목 OK")	
