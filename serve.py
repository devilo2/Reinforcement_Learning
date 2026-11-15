import hashlib, os
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field, FiniteFloat

FMAX = float(np.finfo(np.float32).max)
F32 = Annotated[FiniteFloat, Field(ge=-FMAX, le=FMAX)]


class Obs(BaseModel):
    obs: list[F32] = Field(min_length=8, max_length=8)


MODEL = Path(os.environ["LUNAR_MODEL"])
SHA = hashlib.sha256(MODEL.read_bytes()).hexdigest()
sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
app = FastAPI()


@app.get("/model")
def model():
    return {"path": str(MODEL), "sha256": SHA}


@app.post("/act")
def act(req: Obs):
    x = np.asarray(req.obs, dtype=np.float32)[None]
    logits = sess.run(None, {"obs": x})[0][0]
    return {"action": int(logits.argmax())}
