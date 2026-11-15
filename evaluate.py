import numpy as np
import gymnasium as gym
from common import exp_dirs, save_csv

def run_episodes(policy, n=100, seed0=10_000, sigma=0.0):
    env = gym.make("LunarLander-v3")
    try:
        for ep in range(seed0, seed0 + n):
            obs, _ = env.reset(seed=ep)
            rng = np.random.default_rng([ep, 98765])
            ret, done = 0.0, False
            while not done:
                o = obs.copy()
                o += sigma * rng.standard_normal(8)
                obs, r, term, trunc, _ = env.step(policy(o))
                ret, done = ret + r, term or trunc
            yield ep, ret, term, trunc
    finally:
        env.close()

if __name__ == "__main__":
    EXP, RES, MODELS = exp_dirs()
    rng_a = np.random.default_rng(12345)
    rand = lambda o: int(rng_a.integers(4))
    for name, pi in [("random", rand), ("noop", lambda o: 0)]:
        save_csv(RES / f"baseline_{name}.csv", run_episodes(pi))
