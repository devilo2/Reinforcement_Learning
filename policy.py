import torch, torch.nn as nn

def load_actor(path):
    actor = nn.Sequential(
        nn.Linear(8, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 4))
    sd = torch.load(path, map_location="cpu")
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("actor.")}
    actor.load_state_dict(sd, strict=True)
    return actor.float().eval()

def torch_policy(actor, sample=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    def pi(obs):
        x = torch.tensor(obs, dtype=torch.float32)
        with torch.inference_mode():
            z = actor(x[None])[0]
            if not sample:
                return int(z.argmax())
            p = z.softmax(-1)
            return int(p.multinomial(1, generator=g))
    return pi
