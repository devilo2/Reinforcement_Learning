import argparse
import csv
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import imageio

import client

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--output", required=True)
args = ap.parse_args()

client.URL = args.url
EXP = Path(args.output)
RES = EXP / "results"
VID = EXP / "videos"
RES.mkdir(parents=True, exist_ok=True)
VID.mkdir(parents=True, exist_ok=True)


def episode(seed, video=None):
    env = gym.make("LunarLander-v3", render_mode="rgb_array" if video else None)
    w = imageio.get_writer(video, fps=50) if video else None
    ret = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            if w:
                w.append_data(env.render())
            obs, r, term, trunc, _ = env.step(client.act(obs))
            ret += r
            done = term or trunc
    finally:
        if w:
            w.close()
        env.close()
    return ret


seeds = list(range(10000, 10100)) + list(range(20000, 20200))
rows, found = [], {}
for seed in seeds:
    ret = episode(seed)
    ok = ret >= 200
    kind = "success" if ok else "miss"
    rows.append({"episode_seed": seed, "return": round(ret, 1),
                 "success": ok, "video": ""})
    print(seed, round(ret, 1), kind, flush=True)
    if kind not in found:
        found[kind] = len(rows) - 1
    if len(found) == 2:
        break

# 찾은 사례만 영상 저장
for kind, idx in found.items():
    seed = rows[idx]["episode_seed"]
    path = VID / f"seed{seed}_{uuid4().hex[:8]}.mp4"
    ret = episode(seed, str(path))
    rows[idx]["video"] = str(path)
    print(f"[{kind}] seed {seed} return {ret:.1f} -> {path}")

with open(RES / "demo_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["episode_seed", "return", "success", "video"])
    w.writeheader()
    w.writerows(rows)

print("탐색한 seed 수:", len(rows), "/ 찾은 종류:", list(found))
