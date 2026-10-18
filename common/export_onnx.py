import sys, os, torch, torch.nn as nn
sys.path.insert(0, "/home/hesheep/kinect")
from train_detector import build_detector

class TRTWrapper(nn.Module):
    """绕过 GeneralizeRCNNTransform，输出 raw 预测（无 NMS/If）"""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        feats = self.backbone(x)              # OrderedDict[str, Tensor]
        out = self.head(list(feats.values())) # ★ head 要 list
        # ★ head 返回 dict: {"bbox_regression": [B,N,4], "cls_logits": [B,N,5]}
        return out["cls_logits"], out["bbox_regression"]

dev = torch.device("cpu")
ck = torch.load("runs/det/best.pt", map_location=dev, weights_only=False)
cfg, classes = ck.get("cfg", {}), ck.get("classes")
print(f"类别: {classes}")

model = build_detector(cfg, len(classes)).to(dev)
model.load_state_dict(ck["model"])

w = TRTWrapper(model)
w.eval()                                  # ★ 关键

SIZE = 320
dummy = torch.randn(1, 3, SIZE, SIZE)
with torch.no_grad():
    out = w(dummy)
print("✅ 前向成功:")
for n, o in zip(["cls_logits", "bbox_reg"], out):
    print(f"    {n}: {tuple(o.shape)}")

out_path = "runs/det/best_trt.onnx"
try:
    torch.onnx.export(w, (dummy,), out_path, input_names=["images"],
                      output_names=["cls_logits", "bbox_reg"],
                      opset_version=12, dynamo=False, do_constant_folding=True)
except TypeError:
    torch.onnx.export(w, (dummy,), out_path, input_names=["images"],
                      output_names=["cls_logits", "bbox_reg"],
                      opset_version=12, do_constant_folding=True)
print(f"\n✅ 导出: {out_path}  {os.path.getsize(out_path)/1048576:.1f} MB")

import onnx
from collections import Counter
m = onnx.load(out_path)
c = Counter(n.op_type for n in m.graph.node)
print(f"总节点数: {len(m.graph.node)}  算子种类: {len(c)}")
print("\nTensorRT 风险算子:")
bad = 0
for op in ["If", "NonMaxSuppression", "NonZero", "GatherND", "Loop", "ScatterND"]:
    n = c.get(op, 0)
    if n: bad += 1
    print(f"  {op:<20} {n:>3}  {'✅' if n==0 else '❌'}")
print("\n" + ("✅ 干净，可以转 TensorRT" if bad == 0 else "⚠️ 仍有风险算子"))
