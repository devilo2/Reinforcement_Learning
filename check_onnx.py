import time
import numpy as np
import torch
import gymnasium as gym
import onnxruntime as ort
from common import exp_dirs, load_json, save_json
from policy import load_actor, torch_policy

EXP, RES, MODELS = exp_dirs()


def onnx_policy(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def pi(obs):
        x = {"obs": obs[None].astype(np.float32)}
        return int(sess.run(None, x)[0].argmax())
    return pi


def latency_ms(pi, x, warmup=100, n=1000):
    torch.set_num_threads(1)
    ts = []
    for _ in range(warmup):
        pi(x)
    for _ in range(n):
        t0 = time.perf_counter()
        pi(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


# 1) 모델 로딩과 실제 관측 10,000개 수집
actor = load_actor(MODELS / "best.pt")
sess = ort.InferenceSession(str(MODELS / "policy.onnx"),
                            providers=["CPUExecutionProvider"])
pi = torch_policy(actor)

env, buf = gym.make("LunarLander-v3"), []
try:
    obs, _ = env.reset(seed=0)
    while len(buf) < 10_000:
        buf.append(obs)
        obs, _, term, trunc, _ = env.step(pi(obs))
        if term or trunc:
            obs, _ = env.reset()
finally:
    env.close()
X = np.array(buf, dtype=np.float32)
np.save(RES / "observations.npy", X)

# 2) 출력 차이
tl = lambda x: actor(torch.tensor(x[None])).numpy()
ox = lambda x: sess.run(None, {"obs": x[None]})[0]
with torch.inference_mode():
    t = np.concatenate([tl(x) for x in X])
o = np.concatenate([ox(x) for x in X])
d = np.abs(t - o)
res = {"finite": bool(np.isfinite(d).all()),
       "action_agreement": float((t.argmax(1) == o.argmax(1)).mean()),
       "max_abs_err": float(d.max()),
       "mean_abs_err": float(d.mean())}
print(res)
save_json(RES / "conversion.json", res)

# 3) 지연과 크기
ONNX = str(MODELS / "policy.onnx")
pis = {"PyTorch": pi, "ONNX": onnx_policy(ONNX)}
lat = {k: latency_ms(p, X[0]) for k, p in pis.items()}
torch.save(actor.state_dict(), MODELS / "actor.pt")
kib = lambda n: (MODELS / n).stat().st_size / 1024
c = load_json(RES / "conversion.json")
c.update(latency_ms=lat,
         actor_kib={"PyTorch": kib("actor.pt"), "ONNX": kib("policy.onnx")})
save_json(RES / "conversion.json", c)
print("latency (mean, p95) ms:", lat)
print("size KiB:", c["actor_kib"])
