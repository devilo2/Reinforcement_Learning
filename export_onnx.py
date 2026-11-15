import hashlib, torch, onnx
from common import exp_dirs, load_json, save_json
from policy import load_actor

EXP, RES, MODELS = exp_dirs()

def export(pt_path, onnx_path):
    actor = load_actor(pt_path)
    dummy = torch.zeros(1, 8, dtype=torch.float32)
    torch.onnx.export(actor, (dummy,), str(onnx_path),
                      input_names=["obs"], output_names=["logits"],
                      dynamo=True, opset_version=18,
                      external_data=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))

P = MODELS / "policy.onnx"
export(MODELS / "best.pt", P)
best = load_json(RES / "evaluation.json")["best_seed"]
sha = hashlib.sha256(P.read_bytes()).hexdigest()
save_json(RES / "export.json", {"best_seed": best, "policy_sha256": sha})
print("exported:", P, "best_seed:", best, "sha256:", sha[:8])
