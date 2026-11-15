import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("artifacts/team")
SEL = ROOT / "try5"
OUT = SEL / "report"
OUT.mkdir(parents=True, exist_ok=True)
REP = 20237141
X = np.arange(1, 11) / 10


def load(p):
    return json.load(open(p))


def curve(run_dir):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    ev = ea.Scalars("charts/episodic_return")
    s = np.array([e.step for e in ev])
    r = np.array([e.value for e in ev])
    out = []
    for lo in range(0, 1_000_000, 100_000):
        m = (s > lo) & (s <= lo + 100_000)
        out.append(r[m].mean() if m.any() else np.nan)
    return np.array(out)


def rep_run(n):
    runs = load(ROOT / f"try{n}" / "results" / "training.json")["runs"]
    return [r for r in runs if r["seed"] == REP][0]


md = []

# 1) 베이스라인 표
md.append("## 베이스라인 (평가 seed 10000~10099, 100회)\n")
md.append("| 정책 | 평균 return | 표준편차 | 성공률 |\n|---|---|---|---|")
for n, label in (("random", "랜덤"), ("noop", "무동작")):
    r = np.array([float(x["return"]) for x in
                  csv.DictReader(open(ROOT / "baseline" / "results" / f"baseline_{n}.csv"))])
    md.append(f"| {label} | {r.mean():.1f} | {r.std():.1f} | {np.mean(r >= 200):.0%} |")

# 2) try별 기록 표와 그래프
md.append("\n## 대표 seed try별 last_return\n")
md.append("| try | last_return | 판정 |\n|---|---|---|")
plt.figure(figsize=(8, 4))
for n in range(1, 6):
    run = rep_run(n)
    lr = run["last_return"]
    md.append(f"| try{n} | {lr:.1f} | {'통과' if lr > 200 else '미달'} |")
    plt.plot(X, curve(run["run_dir"]), marker="o", ms=3, label=f"try{n}: last={lr:.1f}")
plt.axhline(200, ls="--", c="gray")
plt.xlabel("Steps (millions), 100k bins")
plt.ylabel("Mean episodic return")
plt.title("Representative seed: improvement attempts")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "tries_curve.png", dpi=150)
plt.close()

# 3) 5 seed 학습 곡선
runs = load(SEL / "results" / "training.json")["runs"]
C = np.array([curve(r["run_dir"]) for r in runs])
m, sd = np.nanmean(C, 0), np.nanstd(C, 0)
plt.figure(figsize=(8, 4))
plt.plot(X, m, marker="o", ms=3, label="mean of 5 seeds")
plt.fill_between(X, m - sd, m + sd, alpha=0.25, label="SD across 5 seeds")
plt.axhline(200, ls="--", c="gray")
plt.xlabel("Steps (millions), 100k bins")
plt.ylabel("Mean episodic return")
plt.title("try5: five-seed learning curve")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "learning_curve_5seeds.png", dpi=150)
plt.close()

# 4) 행동 방식 비교
ev = load(SEL / "results" / "evaluation.json")
md.append("\n## 두 방식 비교 (모델별)\n")
md.append("| 학습 seed | argmax 평균 | 표준편차 | 확률적 평균 | 표준편차 |\n|---|---|---|---|---|")
for mo in ev["models"]:
    a, s_ = mo["argmax"], mo["sample"]
    md.append(f"| {mo['seed']} | {a['mean']:.1f} | {a['sd']:.1f} | {s_['mean']:.1f} | {s_['sd']:.1f} |")
sm = ev["summary"]
md.append(f"| 5개 평균 ± 표준편차 | {sm['argmax']['mean']:.1f} ± {sm['argmax']['sd']:.1f} | {sm['argmax']['episode_sd']:.1f} "
          f"| {sm['sample']['mean']:.1f} ± {sm['sample']['sd']:.1f} | {sm['sample']['episode_sd']:.1f} |")
seeds = [str(mo["seed"]) for mo in ev["models"]]
w = 0.35
idx = np.arange(len(seeds))
plt.figure(figsize=(8, 4))
plt.bar(idx - w / 2, [mo["argmax"]["mean"] for mo in ev["models"]], w, label="argmax")
plt.bar(idx + w / 2, [mo["sample"]["mean"] for mo in ev["models"]], w, label="sample")
plt.xticks(idx, seeds, fontsize=8)
plt.ylabel("Mean return over 100 episodes")
plt.xlabel("Training seed")
plt.title("Same evaluation seeds; two action modes")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "action_modes.png", dpi=150)
plt.close()

# 5) 노이즈
cells = load(SEL / "results" / "noise.json")["cells"]
sg = [c["sigma"] for c in cells]
mn = np.array([c["mean"] for c in cells])
nsd = np.array([c["sd"] for c in cells])
md.append("\n## 관측 노이즈 (5개 모델 × 100회, ± = 5개 모델 평균 사이 표준편차)\n")
md.append("| σ | 평균 return | 표준편차 | σ=0 대비 |\n|---|---|---|---|")
for c in cells:
    md.append(f"| {c['sigma']} | {c['mean']:.1f} | {c['sd']:.1f} | {c['mean'] - cells[0]['mean']:+.1f} |")
pos = np.arange(len(sg))
plt.figure(figsize=(8, 4))
plt.plot(pos, mn, marker="o")
plt.fill_between(pos, mn - nsd, mn + nsd, alpha=0.25, label="SD across 5 model means")
plt.xticks(pos, [str(s) for s in sg])
plt.xlabel("Noise sigma, all 8 observations")
plt.ylabel("Mean episodic return")
plt.title("5 models x 100 episodes at each noise level")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "noise.png", dpi=150)
plt.close()

# 6) 변환 비교
cv = load(SEL / "results" / "conversion.json")
md.append("\n## 변환 비교 (best 모델, 같은 PC, 스레드 1개)\n")
md.append("| 항목 | PyTorch | ONNX |\n|---|---|---|")
md.append(f"| actor 크기 (KiB) | {cv['actor_kib']['PyTorch']:.1f} | {cv['actor_kib']['ONNX']:.1f} |")
lp, lo = cv["latency_ms"]["PyTorch"], cv["latency_ms"]["ONNX"]
md.append(f"| 지연 평균·p95 (ms) | {lp[0]:.4f} · {lp[1]:.4f} | {lo[0]:.4f} · {lo[1]:.4f} |")
md.append(f"| 행동 일치율 | 기준 | {cv['action_agreement']} |")
md.append(f"| 최대 오차 (logits) | 기준 | {cv['max_abs_err']:.2e} |")
md.append(f"| 평균 오차 (logits) | 기준 | {cv['mean_abs_err']:.2e} |")

(OUT / "report_tables.md").write_text("\n".join(md), encoding="utf-8")
print("만들어진 파일:", sorted(p.name for p in OUT.iterdir()))
