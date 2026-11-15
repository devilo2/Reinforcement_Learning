import argparse
from pathlib import Path
from uuid import uuid4

import gradio as gr
import gymnasium as gym
import imageio

import client

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--output", required=True)
args = ap.parse_args()

client.URL = args.url
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)


def run(seed):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    path = OUT / f"seed{int(seed)}_{uuid4().hex[:8]}.mp4"
    w, ret = imageio.get_writer(path, fps=50), 0.0
    try:
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            w.append_data(env.render())
            obs, r, term, trunc, _ = env.step(client.act(obs))
            ret, done = ret + r, term or trunc
    finally:
        w.close()
        env.close()
    return str(path), round(ret, 1)


demo = gr.Interface(
    fn=run,
    inputs=gr.Number(value=10000, label="seed"),
    outputs=[gr.Video(label="에피소드"), gr.Number(label="return")],
    title="RL Policy Inference & Evaluation",
)
demo.launch(server_name="127.0.0.1", server_port=args.port)
