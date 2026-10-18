"""SSD 检测器解码 —— 纯 numpy，不依赖 torchvision"""
import math
import numpy as np
from itertools import product

# SSDlite320 的 anchor 配置（必须和训练时一致）
FEATURE_SIZES = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
SCALES = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
ASPECT_RATIOS = [2, 3]
IMG_SIZE = 320
BBOX_XFORM_CLIP = math.log(1000.0 / 16.0)      # = 4.135
# ★ SSD 专用的 box coder 权重（torchvision SSD 用的是这个，不是默认的 1,1,1,1）
BBOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def generate_anchors(image_size=IMG_SIZE):
    """返回 [N,4] 的 anchor，格式 xyxy（像素坐标）"""
    out = []
    for k, (fh, fw) in enumerate(FEATURE_SIZES):
        s = SCALES[k]
        s2 = math.sqrt(s * SCALES[k + 1])
        wh = [(s, s), (s2, s2)]
        for ar in ASPECT_RATIOS:
            wh.append((s * math.sqrt(ar), s / math.sqrt(ar)))
            wh.append((s / math.sqrt(ar), s * math.sqrt(ar)))
        for i, j in product(range(fh), range(fw)):
            cx, cy = (j + 0.5) / fw, (i + 0.5) / fh
            for w, h in wh:
                out.append([cx, cy, w, h])
    a = np.clip(np.array(out, np.float32), 0, 1)
    return np.concatenate([
        (a[:, :2] - 0.5 * a[:, 2:]) * [image_size, image_size],
        (a[:, :2] + 0.5 * a[:, 2:]) * [image_size, image_size],
    ], axis=-1).astype(np.float32)


def decode_boxes(rel, anchors):
    """SSD 回归偏移 -> 绝对 xyxy（像素）。复刻 torchvision BoxCoder.decode_single"""
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * w
    cy = anchors[:, 1] + 0.5 * h
    wx, wy, ww, wh = BBOX_WEIGHTS
    dx = rel[:, 0] / wx
    dy = rel[:, 1] / wy
    dw = np.clip(rel[:, 2] / ww, None, BBOX_XFORM_CLIP)
    dh = np.clip(rel[:, 3] / wh, None, BBOX_XFORM_CLIP)
    pcx = dx * w + cx
    pcy = dy * h + cy
    pw = np.exp(dw) * w
    ph = np.exp(dh) * h
    return np.stack([pcx - 0.5 * pw, pcy - 0.5 * ph,
                     pcx + 0.5 * pw, pcy + 0.5 * ph], axis=1)


def nms_per_class(boxes, scores, labels, iou_thr=0.55, max_det=20):
    """逐类 NMS。全程用原始索引。"""
    keep = []
    for c in np.unique(labels):
        order = np.where(labels == c)[0]
        order = order[np.argsort(-scores[order])]        # 按分数降序（原索引）
        while order.size:
            cur = order[0]
            keep.append(int(cur))
            if order.size == 1:
                break
            b0 = boxes[cur]
            rest = boxes[order[1:]]
            x0 = np.maximum(b0[0], rest[:, 0]); y0 = np.maximum(b0[1], rest[:, 1])
            x1 = np.minimum(b0[2], rest[:, 2]); y1 = np.minimum(b0[3], rest[:, 3])
            inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
            a0 = (b0[2] - b0[0]) * (b0[3] - b0[1])
            a1 = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            iou = inter / np.maximum(a0 + a1 - inter, 1e-9)
            order = order[1:][iou < iou_thr]
    if not keep:
        return np.zeros(0, int)
    keep = np.array(keep)
    return keep[np.argsort(-scores[keep])][:max_det]


def postprocess(cls_logits, bbox_reg, anchors,
                score_thresh=0.4, iou_thr=0.55, max_det=20):
    """返回 [(x0,y0,x1,y1,cls_id,score), ...]，坐标是 320x320 尺度"""
    boxes = decode_boxes(bbox_reg, anchors)
    # ★ torchvision 的 SSD 用 softmax（不是 sigmoid！）
    x = cls_logits - cls_logits.max(axis=-1, keepdims=True)   # 数值稳定
    e = np.exp(x)
    probs = e / e.sum(axis=-1, keepdims=True)
    fg = probs[:, 1:]
    labels = fg.argmax(1)
    scores = fg.max(1)
    m = scores >= score_thresh
    if not m.any():
        return []
    b, l, s = boxes[m], labels[m], scores[m]
    # ★★ 参数顺序必须是 (boxes, scores, labels)！
    #    写成 (b, l, s) 会静默出错：NMS 把"分数"当类别 → 每个分数自成一组 →
    #    完全不抑制；最后又按"类别号"排序 → 结果全变成类别号最大的那一类。
    k = nms_per_class(b, s, l, iou_thr, max_det)
    return [(float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]),
             int(l[i]), float(s[i])) for i in k]
