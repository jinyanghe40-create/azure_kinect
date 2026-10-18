#!/usr/bin/env python3
"""
几何法点云管线 —— 不训练，纯几何。

完整流程（对应你的方案）:
    深度图 -> 点云 -> 距离滤波 -> RANSAC 去桌面 -> 体素聚类 -> 分簇平面分割
    -> 法向统一 -> 选识别面 -> 算夹角 -> 输出

两种运行模式:
    --synthetic   用虚拟场景（已知真值）验证算法和精度，不需要相机
    --replay      从 .npy/.pcd 读点云
    （默认）       从 Azure Kinect 实时读（需要 libk4a + 深度可用）

合成模式的意义: 相机还没通也能把算法调通，并且能直接测出法向量精度。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------- PLY 读写


def write_ply(path, pts, colors=None):
    """写出二进制 PLY。CloudCompare / MeshLab 都能打开。"""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    with open(path, "wb") as f:
        header = (f"ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {n}\n"
                  f"property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            header += ("property uchar red\nproperty uchar green\n"
                       "property uchar blue\n")
        header += "end_header\n"
        f.write(header.encode("ascii"))
        if colors is not None:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            c = np.asarray(colors, dtype=np.uint8)
            rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
            f.write(rec.tobytes())
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            f.write(rec.tobytes())
    return path


def read_ply(path):
    """读二进制 / ASCII PLY，返回 (N,3) float64。"""
    with open(path, "rb") as f:
        raw = f.read()
    # 找 end_header
    idx = raw.find(b"end_header")
    if idx < 0:
        raise ValueError("不是合法的 PLY 文件")
    header = raw[:idx].decode("ascii", errors="replace")
    body = raw[idx + len("end_header"):]
    if body.startswith(b"\r\n"):
        body = body[2:]
    elif body.startswith(b"\n"):
        body = body[1:]

    n = 0
    props = []          # [(name, type), ...] 顺序即内存布局
    in_vertex = False
    for line in header.splitlines():
        t = line.strip().split()
        if len(t) >= 3 and t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex:
                n = int(t[2])
        elif len(t) >= 3 and t[0] == "property" and in_vertex:
            props.append((t[2], t[1]))

    # 按真实布局构造 dtype（颜色是 uchar，不能只算 float 的步长）
    np_types = {"float": "<f4", "float32": "<f4", "double": "<f8",
                "float64": "<f8", "uchar": "u1", "uint8": "u1",
                "char": "i1", "int8": "i1", "short": "<i2", "ushort": "<u2",
                "int": "<i4", "uint": "<u4"}
    coords = [i for i, (nm, _) in enumerate(props) if nm in ("x", "y", "z")]

    if "binary_little_endian" in header:
        dt = np.dtype([(f"p{i}", np_types.get(ty, "<f4"))
                       for i, (_, ty) in enumerate(props)])
        rec = np.frombuffer(body[:n * dt.itemsize], dtype=dt, count=n)
        if len(coords) < 3:
            raise ValueError("PLY 缺少 x/y/z 属性")
        pts = np.stack([rec[f"p{i}"] for i in coords[:3]], axis=1)
    else:
        arr = np.array([l.split()[:3] for l in body.decode().splitlines()[:n]],
                       dtype=np.float64)
        pts = arr
    return pts.astype(np.float64)


# ---------------------------------------------------------------- 几何工具


def plane_from_3pts(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    d = -float(np.dot(n, p0))
    return n, d


def ransac_plane(pts, thresh=0.008, iters=150, min_inliers=200, rng=None,
                 normal_hint=None, cos_tol=0.5, max_fit_pts=20000):
    """
    RANSAC 平面拟合。
    normal_hint: 若给定，只接受法向与该方向夹角小于 acos(cos_tol) 的平面。
    """
    rng = rng or np.random.default_rng(0)
    n_pts = len(pts)
    # ★ 必须同时判 n_pts < 3：只要 3 个点才能定一个平面。
    #   否则当 min_inliers 被算成 0（点云为空时 int(0.20*0)=0）就会去
    #   rng.choice(0, 3) 然后崩。
    if n_pts < 3 or n_pts < min_inliers:
        return None, None, None

    # ★ 性能：RANSAC 是统计方法，拟合用 2 万点就足够
    if n_pts > max_fit_pts:
        fit = pts[rng.choice(n_pts, max_fit_pts, replace=False)]
    else:
        fit = pts
    n_fit = len(fit)

    # ★★ 向量化：一次性算完【所有】候选平面。
    #   原来每轮都调 plane_from_3pts —— 那是个纯 Python 函数 + 5 次
    #   三元素数组的 numpy 调用，单次就要 0.5ms，80 轮 = 40ms，
    #   而每帧要被调用十几次 → 光这一项就 500ms+。
    #   现在全部改成大数组批量运算，同样的 80 个候选，只要几毫秒。
    idx = rng.integers(0, n_fit, size=(iters, 3))
    p0 = fit[idx[:, 0]]
    p1 = fit[idx[:, 1]]
    p2 = fit[idx[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)               # (iters,3)
    ln = np.linalg.norm(nrm, axis=1)               # (iters,)
    good = ln > 1e-12
    if not good.any():
        return None, None, None
    nunit = np.zeros_like(nrm)
    nunit[good] = nrm[good] / ln[good, None]
    d_all = -(nunit * p0).sum(axis=1)              # (iters,)
    if normal_hint is not None:                    # 软约束：法向大致朝某方向
        good &= np.abs(nunit @ normal_hint) >= cos_tol
    if not good.any():
        return None, None, None
    n_good = nunit[good]
    d_good = d_all[good]

    # 一次性算出每个候选平面在【所有抽样点】上的距离：(n_fit, n_cand)
    dist = np.abs(fit @ n_good.T + d_good[None, :])
    cnt = (dist < thresh).sum(axis=0)
    k = int(cnt.argmax())
    best_model = (n_good[k], float(d_good[k]))

    # 用【全量点】确认这个模型够不够格（只算一次）
    if int(np.count_nonzero(np.abs(pts @ best_model[0] + best_model[1])
                            < thresh)) < min_inliers:
        return None, None, None

    if best_model is None:
        return None, None, None
    # 用【全量点】确认这个模型够不够格（这里只算一次）
    if int(np.count_nonzero(np.abs(pts @ best_model[0] + best_model[1])
                            < thresh)) < min_inliers:
        return None, None, None

    # 自洽收紧：反复「用当前平面选内点 -> 对内点做 PCA 重定平面」。
    # 关键是用内点质心重新定位平面（不能再用全体的质心），否则会漂走。
    n, d = best_model
    for tight in (thresh, thresh * 0.6, thresh * 0.4, thresh * 0.3):
        dist = np.abs(pts @ n + d)
        inl = pts[dist < tight]
        if len(inl) < 50:
            break
        c = inl.mean(axis=0)
        cov = np.cov((inl - c).T)
        w, v = np.linalg.eigh(cov)
        n_new = v[:, 0]
        n_new = n_new / np.linalg.norm(n_new)
        if float(np.dot(n_new, n)) < 0:
            n_new = -n_new
        n = n_new
        d = -float(np.dot(n, c))       # 用内点质心重新定位

    # ★★ 关键：PCA 精修会【转动】法向，所以必须在这里【重新检查】法向提示！
    #   之前只在 RANSAC 候选上检查过一次，精修之后再没查 —— 后果很严重：
    #   某次候选侥幸通过了提示 → PCA 把法向转成物体表面的真实法向 →
    #   提示形同虚设 → "去桌面"把方块整个吃掉。
    #   实测踩到过：相机离方块 30cm 时，画面里最大的平面就是方块正面，
    #   于是 2882/2884 个点被当成桌面删掉，obj=0。
    if normal_hint is not None:
        if abs(float(np.dot(n, normal_hint))) < cos_tol:
            return None, None, None

    mask = np.abs(pts @ n + d) < thresh
    if int(mask.sum()) < min_inliers // 2:
        return None, None, None
    return n, d, mask


def orient_toward_camera(n, pts, cam_origin=None):
    """让法向指向相机（相机在原点）。

    ★★ cam_origin 是【必须】的参数化，不是可选的洁癖：
      原来这里写死"相机在原点"（`dot(n, -c)`）。点云一旦换到【车体系】，
      光心在 (0.25, 0, 0.60) 而不在原点，这个判据就会把法向【翻反】。
      后果不是报错，而是 cube_center_from_faces() 沿着错方向退 edge/2
      ≈16.7cm（往返 ~33cm），**而 cube_ok 仍然是 true** —— 完全静默，
      机械臂照着抓就撞。

    cam_origin 给 None 时行为与旧代码逐位相同（RealSense 那路不受影响）。
    """
    c = pts.mean(axis=0)
    o = np.zeros(3) if cam_origin is None else np.asarray(cam_origin, float)
    if float(np.dot(n, o - c)) < 0:
        n = -n
    return n


# ---------------------------------------------------------------- 体素聚类


def voxel_cluster(pts, voxel=0.02, min_points=60):
    """
    体素栅格 + 3D 连通域（26 邻域）。

    ★ 性能：原来用 Python 的 dict + 栈做 flood fill，10 万点要 4~8 秒。
      现在优先用 scipy.ndimage.label —— 那是 C 实现，毫秒级。
      没有 scipy 时自动退回原来的 Python 版本（结果一样，只是慢）。

    返回 label 数组，-1 表示未归类。
    """
    if len(pts) == 0:
        return np.array([], dtype=int)
    ijk = np.floor(pts / voxel).astype(np.int64)

    # ★★ 性能：这里原来有两个坑，都解决掉：
    #   ① np.unique(axis=0) 对 10 万行做 lexsort —— 300~500ms，改用 1-D key
    #   ② scipy.ndimage.label 要遍历【整个包围盒栅格】（房间场景上百万体素），
    #      而实际只有几千个体素被占据。改成【稀疏图连通域】，
    #      复杂度只跟【被占据的体素数】成正比，跟场景多大无关。
    mn = ijk.min(axis=0)
    g = ijk - mn
    shape = np.asarray(g.max(axis=0) + 1, dtype=np.int64)
    if int(shape.prod()) <= 0:
        return np.array([], dtype=int)
    key = (g[:, 0] * shape[1] + g[:, 1]) * shape[2] + g[:, 2]
    # 1-D 的 unique 比 axis=0 快一个数量级
    uk, inv = np.unique(key, return_inverse=True)
    n_uk = len(uk)

    # 26 邻域里只取"正向"的 13 个偏移（无向图，反向由对称性覆盖）
    off13 = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
             (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
             (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
             (1, -1, 1), (1, -1, -1)]
    ux = uk // (shape[1] * shape[2])
    uy = (uk // shape[2]) % shape[1]
    uz = uk % shape[2]
    rows, cols = [], []
    for dx, dy, dz in off13:
        off = (dx * shape[1] + dy) * shape[2] + dz
        tgt = uk + off
        pos = np.searchsorted(uk, tgt)
        valid = pos < n_uk
        posc = np.where(valid, pos, 0)
        hit = valid & (uk[posc] == tgt)
        if not hit.any():
            continue
        s_i = np.nonzero(hit)[0]
        d_i = posc[s_i]
        # ★ 坐标校验：防止线性 key 在 z/y 方向"回绕"到下一行
        ok = ((ux[d_i] - ux[s_i]) == dx) & ((uy[d_i] - uy[s_i]) == dy) \
            & ((uz[d_i] - uz[s_i]) == dz)
        s_i, d_i = s_i[ok], d_i[ok]
        if len(s_i):
            rows.append(s_i)
            cols.append(d_i)

    if rows:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            A = coo_matrix((np.ones(len(r), np.int8), (r, c)),
                           shape=(n_uk, n_uk))
            _, lab_vox = connected_components(A, directed=False)
            pt_labels = lab_vox[inv].astype(np.int64)
            return _drop_small_clusters(pt_labels, min_points, len(pts))
        except ImportError:
            pass

    # ---- 兜底：没有 scipy 才走原来的 Python flood fill ----
    occupied = {int(k): i for i, k in enumerate(uk)}
    labels = -np.ones(n_uk, dtype=np.int64)
    neigh = [(dx, dy, dz)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if not (dx == 0 and dy == 0 and dz == 0)]
    lab = 0
    for start in range(n_uk):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = lab
        while stack:
            cur = stack.pop()
            kx, ky, kz = int(ux[cur]), int(uy[cur]), int(uz[cur])
            for dx, dy, dz in neigh:
                nb = occupied.get(((kx + dx) * shape[1] + (ky + dy)) * shape[2]
                                  + (kz + dz))
                if nb is not None and labels[nb] == -1:
                    labels[nb] = lab
                    stack.append(nb)
        lab += 1
    return _drop_small_clusters(labels[inv], min_points, len(pts))


def _drop_small_clusters(pt_labels, min_points, n_pts):
    """丢掉点数太少的簇，重新编号成 0..k-1（-1 = 未归类）"""
    out = -np.ones(n_pts, dtype=int)
    uniq, counts = np.unique(pt_labels, return_counts=True)
    new_id = 0
    for l, c in zip(uniq, counts):
        if l < 0 or c < min_points:
            continue
        out[pt_labels == l] = new_id
        new_id += 1
    return out


# ---------------------------------------------------------------- 单簇分析


def analyze_cluster(pts, dist_thresh=0.008, max_faces=3, cam_origin=None):
    """
    对一个簇做迭代平面分割，最多找 3 个互相垂直的面。
    返回面列表，每个面: {'normal','center','n_points','rms','extent'}

    cam_origin: 相机光心在【当前点云坐标系】里的位置。车体系下必须传，
                否则 orient_toward_camera 会把法向翻反（静默 16.7cm 级错误）。
    """
    faces = []
    remaining = pts
    for _ in range(max_faces):
        if len(remaining) < 80:
            break
        hint = None
        if faces:
            # 新面必须与已有面近似垂直
            pass
        # 簇内点数不多，80 轮足够（150 轮是给整幅点云去桌面用的）
        n, d, mask = ransac_plane(remaining, thresh=dist_thresh,
                                  min_inliers=100, iters=80)
        if n is None:
            break
        if faces:
            # 检查与已有面是否近似垂直
            ok = True
            for f in faces:
                # 正方体相邻面近似垂直 -> |cos| 应接近 0
                if abs(float(np.dot(n, f["normal"]))) > 0.35:   # 夹角 < 70.5°
                    ok = False
                    break
            if not ok:
                break
        # 用内点重新定平面（ransac_plane 内部已做过自洽收紧）
        inl = remaining[mask]
        if len(inl) < 50:
            break
        c = inl.mean(axis=0)
        n = orient_toward_camera(n, inl, cam_origin)
        d = -float(np.dot(n, c))
        mask = np.abs(remaining @ n + d) < dist_thresh
        inl = remaining[mask]
        if len(inl) < 50:
            break
        c = inl.mean(axis=0)
        d = -float(np.dot(n, c))
        resid = np.abs(inl @ n + d)
        faces.append({
            "normal": n,
            "center": c,
            "n_points": int(mask.sum()),
            "rms": float(np.sqrt(np.mean(resid ** 2))),
            "extent": float(np.linalg.norm(inl.max(axis=0) - inl.min(axis=0))),
        })
        remaining = remaining[~mask]
    # 默认按点数排序；识别面选择策略在 run_pipeline 里做
    faces.sort(key=lambda f: -f["n_points"])
    return faces


def face_edge_len(dp, e1, e2):
    """面内点 -> 正方形边长。用【各个方向投影宽度的最小值】。

    ★★ 为什么不能直接取"某个方向的投影宽度"（之前就是这么错的）：
      正方形沿任意方向的投影宽度 = e·(|cosθ|+|sinθ|)，
      θ=0（和边平行）时是 e，θ=45°（沿对角线）时是 e·√2 —— **最多高估 41%**。
      之前 e1 是用 np.cross(n, 某个坐标轴) 随便取的，方向不可控，
      于是边长被系统性高估 0~41%，再按 e/2 把体心推歪。
      合成真值实测：真边长 33.5cm，量出来 37.4~42.4cm（高估 12%~27%）❌

    ★ 正解：凸形状在所有方向上的投影宽度，**最小值就是它的宽度**。
      正方形的宽度恒等于边长，和怎么转无关 → 取方向扫描的最小值即可。
      不依赖任何库，且对面内噪声比最小外接矩形更稳。

    方向步进 0.5°，误差上界约 0.4%。
    """
    ang = np.linspace(0.0, np.pi, 361, endpoint=False)
    # dp 是面内的【三维】向量，e1/e2 是面内的一组正交基 →
    # 面内方向就在三维里构造：dir = cosθ·e1 + sinθ·e2
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    dirs = (np.cos(ang)[:, None] * e1[None, :]
            + np.sin(ang)[:, None] * e2[None, :])            # (361, 3)
    proj = dp @ dirs.T                                       # (N, 361)
    return float((proj.max(axis=0) - proj.min(axis=0)).min())


def measure_edge(faces, pts, tol=0.005, min_pts=50, spread_tol=0.25):
    """从拟合面的"面内长宽"反推立方体边长。

    ★ 为什么必须自己量：
      --edge 是手填的。填小了，画出来的面片就盖不满真实那张面
      （25cm 的片贴在 33.5cm 的面上，覆盖率只有 (25/33.5)² ≈ 56%，
       看起来就是"只盖住一半"）；更要命的是几何中心是按 e/2 推的，
       边长错 8.5cm，中心每个轴就偏 4.25cm。

    ★ 顺便做了个自洽性校验（这才是"方块"的硬约束）：
      立方体每个面都是**同样大的正方形**，所以各面量出来应当一致。
      差太多说明这压根不是方块（比如一面墙）→ 拒绝，退回 --edge。
      实测：药品方块三个面 33.8 / 33.5 / 32.4 cm，离散 4% ✅
            背景大平面   51.3 / 74.7 cm，       离散 31% ❌ 被拒

    ★ 兜底策略（因为"手填常数"是最不可信的一环）：
      立方体的面被遮挡只会量得【偏小】，不会偏大 —— 所以各面不一致时，
      取【最大的那个面】比用 --edge 靠谱得多。
      三种来源分清楚，并一起发出去，让下游知道这个值可不可信：
        "auto"     各面一致      → 中位数，可信
        "auto-max" 各面不一致    → 取最大面（可能有一个面被遮住），勉强可信
        "cli"      一个面都没量到 → 用 --edge，不可信

    返回 (边长, 各面测量值列表, 来源)，量不出来返回 (None, [], "cli")。
    """
    vals = []
    pts = np.asarray(pts, float)
    for f in faces:
        n = np.asarray(f["normal"], float)
        n = n / max(1e-9, float(np.linalg.norm(n)))
        c = np.asarray(f["center"], float)
        d = pts - c
        on = np.abs(d @ n) < tol              # 只取真落在这张面上的点
        if int(on.sum()) < min_pts:
            continue
        dp = d[on] - np.outer(d[on] @ n, n)   # 去掉法向分量 = 投到面内
        ref = (np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9
               else np.array([1.0, 0.0, 0.0]))
        e1 = np.cross(n, ref)
        e1 = e1 / max(1e-9, float(np.linalg.norm(e1)))
        e2 = np.cross(n, e1)
        vals.append(face_edge_len(dp, e1, e2))
    if not vals:
        return None, [], "cli"
    if len(vals) >= 2:
        lo, hi = min(vals), max(vals)
        if hi > 0 and (hi - lo) / hi > spread_tol:
            # 各面差太多：可能是被遮挡的方块，也可能压根不是方块。
            # 遮挡只会让面变小 → 取最大面；同时把来源标出来。
            return float(hi), vals, "auto-max"
    return float(np.median(vals)), vals, "auto"


def cube_center_from_faces(faces, edge_len, cluster_pts=None):
    """由已知的若干正交面 + 边长 L 推算立方体几何中心。

    ★ 为什么不能"面少于 3 个就退回面心"（之前的做法）：
      面心在物体【表面上】，立方体中心在往里 e/2 的地方。
      拿面心当中心去画方块，方块就有一半戳在物体外面 ——
      RViz 里看起来就是"黄方块没套住真实方块 / 只盖住一半"。
      法向是朝外的（orient_toward_camera 保证），所以往里推就是 -n 方向。

    做法：
      3 个面 → 三平面求交得顶点，再沿三个法向各退 e/2（精确解）。
      1~2 个面 → 先用点云质心定"面内"位置，再沿每个已知法向推到 e/2 深处。
                 缺的那个方向没法确定，用质心当最佳猜测，比面心好得多。
    """
    if not faces or not edge_len:
        return None
    ns = [np.asarray(f["normal"], float) for f in faces]
    cs = [np.asarray(f["center"], float) for f in faces]
    ns = [n / max(1e-9, float(np.linalg.norm(n))) for n in ns]

    if len(faces) >= 3:
        A = np.array(ns[:3])
        b = np.array([float(np.dot(n, c)) for n, c in zip(ns[:3], cs[:3])])
        try:
            vertex = np.linalg.solve(A, b)
            return vertex - (edge_len / 2.0) * sum(ns[:3])
        except np.linalg.LinAlgError:
            pass                     # 退化（三面共面）→ 走下面的通用分支

    if cluster_pts is not None and len(cluster_pts):
        p = np.asarray(cluster_pts, float).mean(axis=0)
    else:
        p = np.mean(cs, axis=0)
    for n, c in zip(ns, cs):
        target = float(np.dot(n, c)) - edge_len / 2.0
        p = p + (target - float(np.dot(p, n))) * n
    return p


def cube_center_from_3faces(faces, edge_len):
    """旧接口，保留兼容（只用 3 个面的精确解）。"""
    return cube_center_from_faces(faces, edge_len, None) if len(faces) >= 3 \
        else None


# ---------------------------------------------------------------- 合成场景


def make_synthetic_scene(n_cubes=4, edge=0.05, seed=0, noise_sigma=0.002,
                         table_y=None):
    """
    生成虚拟点云：桌面 + 若干随机旋转的正方体。
    返回 pts(米), 真值列表。
    """
    rng = np.random.default_rng(seed)
    if table_y is None:
        table_y = max(0.25, edge * 1.2)
    pts_all = []
    truth = []

    # 桌面：相机系 +y 向下，桌面在方块下方（y 更大）
    span = max(0.45, edge * 3.2)
    xs = rng.uniform(-span, span, 30000)
    zs = rng.uniform(0.5, 0.5 + span * 1.8, 30000)
    ys = np.full(30000, table_y) + rng.normal(0, noise_sigma, 30000)
    pts_all.append(np.stack([xs, ys, zs], axis=1))

    # 桌面上的正方体（z 为深度方向，方块中心在 table_y 上方）
    # 2x2 网格摆放，间距 > 2 倍边长，避免互相重叠
    sp = edge * 2.2
    c0 = 0.5 + span * 0.9
    positions = [(-sp / 2, c0 - sp / 2), (sp / 2, c0 - sp / 2),
                 (-sp / 2, c0 + sp / 2), (sp / 2, c0 + sp / 2)]
    for i in range(n_cubes):
        cx, cz = positions[i % len(positions)]
        center = np.array([cx, table_y - edge / 2.0, cz])

        # 随机旋转（但保持一个面基本朝上，模拟"放在桌上"）
        ax = rng.uniform(-0.5, 0.5)
        ay = rng.uniform(-0.5, 0.5)
        az = rng.uniform(0, 2 * math.pi)
        from math import cos, sin
        Rx = np.array([[1, 0, 0], [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]])
        Ry = np.array([[cos(ay), 0, sin(ay)], [0, 1, 0], [-sin(ay), 0, cos(ay)]])
        Rz = np.array([[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        # 每个面撒点（必须按面朝向换轴，否则造出来的不是正方体）
        for axis in range(3):
            others = [k for k in range(3) if k != axis]
            for sign in (-1, 1):
                # 背面/底面相机看不到 -> 不生成（真实深度图就是这个样）
                axis_vec_chk = np.zeros(3)
                axis_vec_chk[axis] = sign
                n_world_chk = R @ axis_vec_chk
                fc_chk = center + R @ (axis_vec_chk * edge / 2)
                if float(np.dot(n_world_chk, -fc_chk)) <= 0:
                    continue
                n = 900
                # 面内两个坐标随机，第三个固定为 ±edge/2
                local = np.zeros((n, 3))
                local[:, others[0]] = rng.uniform(-edge / 2, edge / 2, n)
                local[:, others[1]] = rng.uniform(-edge / 2, edge / 2, n)
                local[:, axis] = sign * edge / 2
                p = (R @ local.T).T + center
                pts_all.append(p + rng.normal(0, noise_sigma, p.shape))
                axis_vec = np.zeros(3)
                axis_vec[axis] = sign
                truth.append({"cube": i, "normal": R @ axis_vec,
                              "face_center": center + R @ (axis_vec * edge / 2)})
    pts = np.vstack(pts_all).astype(np.float64)
    # 只保留真值里"朝向相机"的面（真实相机看不到背面/底面）
    truth = [t for t in truth
             if float(np.dot(t["normal"], -t["face_center"])) > 0]

    # 只保留可见的（深度相机在原点，只看到 z>0 且在视锥内的）
    # 简化：保留 z 在 0.3~1.5 之间，且 x,y 在合理范围
    keep = (pts[:, 2] > 0.3) & (pts[:, 2] < 2.0) & (np.abs(pts[:, 1]) < span)
    pts = pts[keep]
    return pts, truth


# ---------------------------------------------------------------- 主管线


# ---------------------------------------------------------------- 按框切簇


def split_clusters_by_boxes(pts, labels, n_clusters, uv, boxes,
                            min_pts=60, log=print):
    """★ 按 2D 检测框把一个簇再切成几个 —— 治"两个方块贴在一起"。

    为什么需要：voxel_cluster 是体素栅格上的连通域，两个方块贴死就是一团；
    而下游 match_classes 是【一簇一个物体】（把簇心投影进检测框找类别）。
    于是 4 块摆成"两列×两行、列内贴死"时只出 2 个物体，标签还是每列
    上面那块。实测 --voxel 从 0.02 一路调到 0.004 都是 2 个 —— 贴死就是
    贴死，调体素没用。

    切法：簇内的点按"投影落在哪个框里"分组。

    框外的点分两种，不能一视同仁：
      · 只是"贴着框边"（方块的顶面/侧面常常露出框外一点点）→ 收给最近的框。
        ★ 不能直接丢：丢了面数会变少，cube_center_from_faces 会退化。
      · 离所有框都远 → 说明那一块【根本没被检测到】，丢掉。
        硬塞给最近的框会把它错标成邻居的类别（实测 药品 在默认阈值
        0.4 下压根不出框）—— 掉一个没检出的，比污一个检出的干净。

    整个簇一个框都没碰到的，原样保留（= 老行为，兜底）。

    uv:    (N,2) 每个点在【检测框所在那张图】里的像素坐标
    boxes: [(x0,y0,x1,y1,cid,score), ...]
    """
    if n_clusters == 0 or not boxes:
        return labels, n_clusters
    uv = np.asarray(uv, np.float64)
    if len(uv) != len(pts) or not np.isfinite(uv).any():
        log("[4b] 按检测框切簇：uv 对不上或全无效，跳过")
        return labels, n_clusters

    rect = np.array([[float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                     for b in boxes], np.float64)          # (B,4)
    bcx = 0.5 * (rect[:, 0] + rect[:, 2])
    bcy = 0.5 * (rect[:, 1] + rect[:, 3])

    inside = ((uv[:, None, 0] >= rect[None, :, 0]) &
              (uv[:, None, 0] <= rect[None, :, 2]) &
              (uv[:, None, 1] >= rect[None, :, 1]) &
              (uv[:, None, 1] <= rect[None, :, 3]))        # (N,B)
    # ★ 框是会重叠的（实测这场景里 仪器/食品 两个框在图上叠了约 40px），
    #   所以"落在框里"不唯一 —— 再挑【中心离得最近】的那个。
    d2 = ((uv[:, None, 0] - bcx[None, :]) ** 2 +
          (uv[:, None, 1] - bcy[None, :]) ** 2)            # (N,B)
    with np.errstate(invalid="ignore"):
        box_of = np.argmin(np.where(inside, d2, np.inf), axis=1)

    # 框外点：只有"贴着框边"的才收（收得下就收，收不下就丢 —— 见 docstring）
    dx = np.maximum(np.maximum(rect[None, :, 0] - uv[:, None, 0],
                               uv[:, None, 0] - rect[None, :, 2]), 0.0)
    dy = np.maximum(np.maximum(rect[None, :, 1] - uv[:, None, 1],
                               uv[:, None, 1] - rect[None, :, 3]), 0.0)
    dout = np.sqrt(dx * dx + dy * dy)                        # (N,B)
    n_b, n_uv = len(rect), len(uv)
    marg = 0.25 * np.hypot(rect[:, 2] - rect[:, 0],
                           rect[:, 3] - rect[:, 1])          # (B,)
    dout = np.where(dout <= marg[None, :], dout, np.inf)
    box_near = np.argmin(dout, axis=1)
    near_ok = np.isfinite(dout[np.arange(n_uv), box_near])
    has_box = inside.any(axis=1)
    owner = np.where(has_box, box_of, box_near)
    owner[(~has_box) & (~near_ok)] = -1                      # 离得太远 → 丢

    new_labels = -np.ones(len(pts), dtype=np.int64)
    n_new = 0
    n_kept = 0
    parts = []
    for cid in range(n_clusters):
        m = labels == cid
        if not m.any():
            continue
        if not has_box[m].any():          # 这个簇没碰到任何框 → 原样保留
            new_labels[m] = n_new
            n_new += 1
            n_kept += 1
            continue
        idx = np.flatnonzero(m)
        for b in np.unique(owner[idx]):
            if b < 0:                     # 框外又离得远 → 丢（不是任何方块）
                continue
            sel = idx[owner[idx] == b]
            if len(sel) < min_pts:        # 太少的当噪声丢掉
                continue
            new_labels[sel] = n_new
            parts.append((cid, int(b), len(sel)))
            n_new += 1

    if n_new == 0:
        return labels, n_clusters
    n_drop = int((new_labels < 0).sum())
    log(f"[4b] 按检测框切簇（{len(boxes)} 个框，最少 {min_pts} 点）："
        f"{n_clusters} 簇 → {n_new} 块，丢 {n_drop} 点"
        + (f"（{n_kept} 簇没碰到框，原样保留）" if n_kept else ""))
    for cid, b, cnt in parts:
        log(f"     簇{cid} × 框{b}({boxes[b][4]}) → {cnt} 点")
    return new_labels, n_new


def _apply_box_split(pts, labels, n_clusters, args, min_pts, log):
    """run_pipeline 里的一层薄壳：从 args 里取投影函数和检测框。

    投影函数由调用方（run_azure.py）提供 —— pc_pipeline 只做几何，
    不该知道自己是被哪台相机、哪张彩色图的框切的。
    """
    fn = getattr(args, "split_uv_fn", None)
    boxes = getattr(args, "split_boxes", None)
    if fn is None or not boxes or n_clusters == 0:
        # ★ 必须说话：以前这里是静默 return，"没切开"和"根本没开"在日志上
        #   一模一样，害得我对着 obj=2 查了半天。只在 verbose 下打。
        log(f"[4b] 按框切簇：跳过（投影函数={'有' if fn is not None else '缺'}，"
            f"检测框={len(boxes) if boxes else 0} 个，簇={n_clusters}）")
        return labels, n_clusters
    try:
        uv = fn(pts)
    except Exception as e:                          # noqa: BLE001
        log(f"[4b] 按检测框切簇：投影失败（{e}），跳过")
        return labels, n_clusters
    return split_clusters_by_boxes(pts, labels, n_clusters, uv, boxes,
                                   min_pts=min_pts, log=log)


def _unit3(v):
    if v is None:
        return None
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"要三个数，实际 {a.shape}")
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return None
    return a / n


def _vec3(v, what):
    """把 "0,0,1" / [0,0,1] / (0,0,1) 统一成 (3,) 数组。"""
    if v is None:
        return None
    if isinstance(v, str):
        parts = [s for s in v.replace(";", ",").split(",") if s.strip()]
        if len(parts) != 3:
            raise ValueError(f"{what} 要三个数，形如 0,0,1；实际给了 {v!r}")
        v = [float(s) for s in parts]
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"{what} 要三个数，实际 {a.shape}")
    return a


def resolve_frame(args):
    """把"点云在哪个坐标系里"这件事解析成三个显式向量。

    ★ 为什么不硬编码：这一份 pipeline 同时被三条路用 ——
        · RealSense（相机系，相机在原点）
        · Azure + 重力对齐（物理竖直系，相机在原点）
        · Azure + 车体标定（车体系 X前/Y左/Z上，光心在 t_out）
      硬编码任何一个都会把另外两个弄错。

    返回 (up_axis, cam_axis, cam_origin)：
      up_axis    : 输出帧里的"上"（算 lean_deg 用）。None = 不知道 → 不输出
      cam_axis   : 相机光轴在该帧的方向（算 angle_deg / 挑最正对的面用）
      cam_origin : 相机光心在该帧的位置（距离滤波 / 法向朝外用）

    兼容老接口：`--gravity` 等价于 `--up-axis 0,1,0`。
    """
    up = _vec3(getattr(args, "up_axis", None), "--up-axis")
    if up is None and getattr(args, "gravity", False):
        up = np.array([0.0, 1.0, 0.0])   # 老 --gravity：点云已转正，+Y 是竖直
    up_axis = _unit3(up) if up is not None else None

    ca = _unit3(_vec3(getattr(args, "cam_axis", None), "--cam-axis"))
    if ca is None:
        ca = np.array([0.0, 0.0, 1.0])
    cam_axis = ca
    co = _vec3(getattr(args, "cam_origin", None), "--cam-origin")
    cam_origin = np.zeros(3) if co is None else co
    return up_axis, cam_axis, cam_origin


def run_pipeline(pts, args, verbose=True):
    t0 = time.time()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    dump = getattr(args, "dump", None)
    if dump:
        os.makedirs(dump, exist_ok=True)

    # ★★ 先定下"点云在哪个坐标系里"。后面每一处用到"上/光轴/相机位置"
    #    的地方都必须走这三个变量，不许再写字面量 (0,0,1) / (0,1,0) / 原点。
    up_axis, cam_axis, cam_origin = resolve_frame(args)
    log(f"[0] 坐标系: 上={None if up_axis is None else np.round(up_axis,3).tolist()} "
        f"光轴={np.round(cam_axis,3).tolist()} "
        f"光心={np.round(cam_origin,3).tolist()}")

    log(f"[1] 输入点云: {len(pts)} 点")
    if dump:
        write_ply(os.path.join(dump, "step1_raw.ply"), pts)

    # 距离滤波
    # ★★ 必须是【到相机的距离】，不是到坐标原点的距离。
    #    车体系下原点是车心，直接 norm(pts) 量的是"到车心多远"，
    #    而 --min-dist 0.3 / --max-dist 1.5 是按【相机距离】实测调出来的。
    #    减去 cam_origin 之后，阈值语义和数值【完全不变】，不用重标。
    d = np.linalg.norm(pts - cam_origin, axis=1)
    keep = (d > args.min_dist) & (d < args.max_dist)
    pts = pts[keep]
    log(f"[2] 距离滤波 {args.min_dist}-{args.max_dist}m -> {len(pts)} 点")
    if dump:
        write_ply(os.path.join(dump, "step2_range.ply"), pts)

    # ★ 点数太少就别往下跑了（后面 RANSAC/聚类都需要足够多的点），
    #   直接返回"这一帧没东西"，而不是崩掉。
    if len(pts) < 100:
        log(f"[!] 只有 {len(pts)} 个点在 {args.min_dist}-{args.max_dist}m 范围内，"
            f"跳过几何分析")
        log("    常见原因：① 相机对着太远的地方 → 调大 --max-dist")
        log("              ② 画面里没有物体/太近 → 调小 --min-dist")
        log("              ③ 深度图有问题 → 跑 realsense/test_streams.py 确认")
        return {"objects": [], "timing_ms": round((time.time() - t0) * 1000, 1)}

    # 去掉桌面：法向约束为"大致朝上"
    # ★ 用 up_axis 而不是写死 (0,1,0)。判据是 abs(n@hint)，不分正负 ✔
    #   所以 (0,1,0)（相机系点云里 Y 朝下）和 (0,0,1)（车体系里 Z 朝上）
    #   在这里等价 —— 只是"哪个方向算桌面"这件事得说对。
    table_normal_hint = (up_axis if up_axis is not None
                         else np.array([0.0, 1.0, 0.0]))
    n, dd, mask = ransac_plane(pts, thresh=args.table_thresh,
                               min_inliers=int(0.20 * len(pts)),
                               normal_hint=table_normal_hint, cos_tol=0.3)
    if n is not None:
        removed = int(mask.sum())
        pts_obj = pts[~mask]
        log(f"[3] 去桌面 (阈值{args.table_thresh*1000:.0f}mm): 移除 {removed} 点 "
            f"({removed/len(pts):.0%})，剩余 {len(pts_obj)} 点")
        log(f"    桌面法向={np.round(n, 4)}  平面偏移 d={dd:.4f} m")
        if dump:
            write_ply(os.path.join(dump, "step3_table.ply"), pts[mask])   # 被删掉的
            write_ply(os.path.join(dump, "step3_objects.ply"), pts_obj)   # 剩下的
    else:
        pts_obj = pts
        log("[3] 去桌面: 未找到可靠桌面平面（可能场景里没有）")
        if dump:
            write_ply(os.path.join(dump, "step3_objects.ply"), pts_obj)

    if len(pts_obj) < args.min_cluster_points:
        return {"objects": [], "timing_ms": (time.time() - t0) * 1000}

    # 体素聚类
    min_pts = args.min_cluster_points or max(60, int((args.edge / args.voxel) ** 2 * 3))
    labels = voxel_cluster(pts_obj, voxel=args.voxel, min_points=min_pts)
    n_clusters = 0 if len(labels) == 0 else int(labels.max()) + 1
    log(f"[4] 体素聚类 (voxel={args.voxel}m, min_pts={min_pts}): "
        f"{n_clusters} 个簇")
    # ★★ 按 2D 检测框切簇：两个方块贴一起时聚类只能给出一团（见函数注释）。
    #    放在体素聚类【之后】，所以它切的是"已经去过桌面、已经按距离筛过"
    #    的点 —— 不会把背景墙上的点误当成方块。没给框时完全是老行为。
    labels, n_clusters = _apply_box_split(
        pts_obj, labels, n_clusters, args, max(60, min_pts // 4), log)
    if dump and n_clusters > 0:
        # 每个簇一种颜色，导出成一张彩色 PLY，一眼看出哪几个簇
        palette = np.array([
            [230, 60, 60], [60, 200, 60], [70, 110, 240], [240, 200, 50],
            [190, 70, 220], [60, 210, 210], [250, 140, 40], [150, 150, 150],
        ], dtype=np.uint8)
        colors = palette[labels % len(palette)]
        colors[labels < 0] = [0, 0, 0]
        write_ply(os.path.join(dump, "step4_clusters.ply"), pts_obj, colors)
        # 簇统计
        for cid in range(n_clusters):
            cp = pts_obj[labels == cid]
            ext = cp.max(axis=0) - cp.min(axis=0)
            log(f"    簇{cid}: {len(cp):>6} 点  质心={np.round(cp.mean(axis=0), 4)}  "
                f"尺寸(xyz)={np.round(ext * 1000, 1)} mm")

    objects = []
    n_skipped = 0
    for cid in range(n_clusters):
        cp = pts_obj[labels == cid]

        # ★ 性能：快速预筛。方块的包围盒不可能比它自己大太多，
        #   而桌面/墙/设备这些"大杂簇"会白白吃掉几秒钟的 RANSAC。
        #   超过 4 倍边长就直接跳过（放宽一点，避免误杀只露出一部分的面）。
        ext = cp.max(axis=0) - cp.min(axis=0)
        if float(ext.max()) > args.edge * 4.0:
            n_skipped += 1
            continue

        # ★★ 平面阈值必须随【深度噪声】放大，否则远一点就崩。
        #   RealSense 的深度噪声模型：σ_z ≈ coef · z²    （视差亚像素噪声放大）
        #     D435i: f=385px, 基线 50mm, 亚像素 σ_d≈0.15px
        #     → σ_z ≈ z² · 0.0078  (z 单位 m)
        #   实测校验：z≈0.5m 时预测 1.95mm，实测平面 RMS 0.9~1.7mm ✅
        #   合成真值实测：阈值固定 3mm 时，σz=8mm 角度误差 1.18°；
        #                 自适应 max(3mm, 2σz) 后降到 0.38°（好 3 倍）。
        # ★★ z_med 必须是【沿光轴的深度】，不是 cp[:,2]。
        #   相机系里 cp[:,2] 恰好就是深度，所以老代码对；车体系里
        #   cp[:,2] 是【高度】—— 拿高度去套 σz ∝ z² 的 ToF 噪声模型，
        #   平面阈值会忽松忽紧（近处物体算成"深" → 阈值过大 → 面糊）。
        #   叠加仓库已知的模型偏差（部署指南）后，角度精度会悄悄变差。
        z_cam = (cp - cam_origin) @ cam_axis
        z_med = float(np.median(np.clip(z_cam, 0.1, None)))
        sig_z = getattr(args, "depth_noise_coef", 0.008) * z_med ** 2
        thresh_use = max(args.plane_thresh, 2.0 * sig_z)
        faces = analyze_cluster(cp, dist_thresh=thresh_use,
                                max_faces=args.max_faces,
                                cam_origin=cam_origin)
        if not faces:
            continue
        if args.face_select == "most_facing":
            # 法向与【光轴】夹角最小 = 最正对相机
            main = min(faces, key=lambda f: abs(float(np.dot(f["normal"],
                                                           cam_axis))) * -1)
        else:
            main = faces[0]                  # 点数最多
        nvec = main["normal"]
        center = main["center"]
        angle = math.degrees(math.acos(min(1.0, abs(float(np.dot(nvec,
                                                                 cam_axis))))))

        if dump:
            flat = faces[0]
            sel = np.abs(cp @ flat["normal"] + (-float(np.dot(flat["normal"], flat["center"])))) < args.plane_thresh
            write_ply(os.path.join(dump, f"step5_cluster{cid}_face0.ply"), cp[sel])
        obj = {
            "id": len(objects),
            "face_center_m": [round(float(v), 4) for v in center],
            "normal": [round(float(v), 4) for v in nvec],
            "angle_deg": round(angle, 2),
            "n_points": int(len(cp)),
            "n_faces": len(faces),
            "face_rms_mm": round(main["rms"] * 1000, 2),
            # ★★ 只有点云【对齐到了某个物理竖直系】时，才谈得上"物体到底
            #    歪没歪"。up_axis 就是那个"上"：
            #       · 车体系标定 → (0,0,1)
            #       · 老 --gravity → (0,1,0)
            #       · 相机系（RealSense / 没标定）→ None → 不输出，保持和
            #         以前一样（老代码用 args.gravity 门控，语义没变）
            #        lean_deg = 面偏离【铅垂面】的角度 = asin(|n · up|)
            #           0°  = 面是铅垂的（像一堵墙）← 方块立在桌上应该≈0°
            #           90° = 面是水平的（像桌面/地板）
            #    ★ 凭什么要单列一个：angle_deg 是"和【光轴】的夹角"，
            #      它把【相机自己的俯仰】也算了进去 —— 实测这台相机仰
            #      5.90°，于是一个完全铅垂的方块面被报成 4.4°。
            #      两者之差 |0.103−0.031| 恰好就是 sin(5.90°)，反过来
            #      证明只有"上"真竖直时这两件事才分得开。
            "lean_deg": (round(math.degrees(math.asin(min(1.0, abs(float(
                np.dot(nvec, up_axis)))))), 2)
                if up_axis is not None else None),
            # ★ 本簇的点也带上：RViz 里要画、离线核查时要把它投回照片
            #   看"拟合的面"到底盖住的是哪一块（调角度问题全靠这个）。
            #   注意它只在内存里传递，--json / ROS 载荷都是按字段挑的，
            #   不会把几千个点打到串口或网络上。
            "pts": cp,
            "faces": [
                {"normal": [round(float(v), 4) for v in f["normal"]],
                 "center": [round(float(v), 4) for v in f["center"]],
                 "angle_deg": round(math.degrees(math.acos(
                     min(1.0, abs(float(np.dot(f["normal"], cam_axis)))))), 2),
                 "lean_deg": (round(math.degrees(math.asin(min(1.0, abs(float(
                     np.dot(f["normal"], up_axis)))))), 2)
                     if up_axis is not None else None),
                 "n_points": f["n_points"],
                 "rms_mm": round(f["rms"] * 1000, 2)}
                for f in faces
            ],
        }
        # ★ 边长：优先用"现场量出来的"，量不出来才退回 --edge。
        #   量出来的值直接决定面片多大、几何中心往里推多少。
        edge_meas, edge_vals, edge_src = measure_edge(faces, cp)
        use_auto = getattr(args, "edge_auto", True)
        if use_auto and edge_meas:
            edge_use = edge_meas
        else:
            edge_use, edge_src = args.edge, "cli"
        if verbose:
            log(f"      边长: 各面实测 {[round(v*100,1) for v in edge_vals]} cm "
                f"→ 采用 {edge_use*100:.1f} cm（来源 {edge_src}"
                + ("，拿命令行兜底，不可信" if edge_src == "cli" else ")")
                + (f"，--edge 填的是 {args.edge*100:.0f} cm" if edge_src != "cli" else ""))

        geo = cube_center_from_faces(faces, edge_use, cp)
        if geo is not None:
            obj["cube_center_m"] = [round(float(v), 4) for v in geo]
        # ★ 离相机最近的【物体点】—— 卷尺量的就是它。
        #   为什么不能拿面心代替：面心比最近点远 (e/2)·sin(倾角)，
        #   斜放 33° 时能差 6.5cm，看起来就像"测距差了 15%"。
        # ★ 主面上的内点数 —— 精度的决定性因素：
        #   横向抖动 ≈ (半边长) / √面上点数，所以点越多越稳。
        #   实测（stride 5，0.78m）：面上 397 点 → 抖动 6~13mm
        #        （stride 2，0.77m）：面上点数 ×4  → 抖动 2.0mm
        obj["n_face_pts"] = int(main["n_points"])
        # ★ 到【相机】的距离（不是到原点）。车体系下原点是车心，
        #   直接 norm(cp) 量的是"到车心多远" —— 而 quality_of 的阈值
        #   (nearest ≤ 1.05 / ≤ 1.30) 是按相机距离实测标定的，
        #   换个基准会让 quality 整体掉成 COARSE，而且【不报错】。
        #   减掉 cam_origin 后阈值语义和数值完全不变。
        d_all = np.linalg.norm(cp - cam_origin, axis=1)
        i_near = int(np.argmin(d_all))
        obj["nearest_m"] = round(float(d_all[i_near]), 4)
        obj["nearest_pt_m"] = [round(float(v), 4) for v in cp[i_near]]
        obj["plane_thresh_used_mm"] = round(float(thresh_use) * 1000, 2)
        obj["sigma_z_mm"] = round(float(sig_z) * 1000, 2)
        obj["edge_used_m"] = round(float(edge_use), 4)
        obj["edge_src"] = edge_src          # auto / auto-max / cli
        if edge_vals:
            obj["edge_faces_m"] = [round(float(v), 4) for v in edge_vals]
        objects.append(obj)
        log(f"    簇{cid}: {len(cp)} 点, {len(faces)} 个面, "
            f"面心={np.round(center, 4)}, 法向={np.round(nvec, 3)}, "
            f"夹角={angle:.1f}°, RMS={main['rms']*1000:.1f}mm")

    if n_skipped:
        log(f"    （跳过了 {n_skipped} 个过大的簇：包围盒 > "
            f"{args.edge*4*100:.0f}cm，多半是桌面/墙）")
    return {"objects": objects, "timing_ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------- 精度评估


def evaluate(truth, objects, verbose=True):
    """把检测到的面与真值面按面心最近邻配对，统计法向量误差。"""
    errors = []
    for t in truth:
        tf = t["face_center"]
        best = None
        for o in objects:
            fc = np.array(o["face_center_m"])
            d = np.linalg.norm(fc - tf)
            if d < 0.03 and (best is None or d < best[0]):
                best = (d, o)
        if best is None:
            continue
        on = np.array(best[1]["normal"])
        tn = t["normal"] / np.linalg.norm(t["normal"])
        # 法向允许反向（面朝向约定可能不同）
        cosang = abs(float(np.dot(on, tn)))
        cosang = min(1.0, max(-1.0, cosang))
        errors.append(math.degrees(math.acos(cosang)))

    if not errors:
        print("\n[精度] 没有配对成功的面，无法评估")
        return None

    e = np.array(errors)
    print(f"\n{'='*56}")
    print(f"[精度评估] 配对 {len(e)} 个面 / 真值 {len(truth)} 个面")
    print(f"  法向量误差  平均 {e.mean():.2f}°   "
          f"中位 {np.median(e):.2f}°   最大 {e.max():.2f}°")
    print(f"  95% 分位    {np.percentile(e, 95):.2f}°")
    ok = np.percentile(e, 95) <= 5.0
    print(f"  是否达标(95% ≤ 5°): {'✅ 是' if ok else '❌ 否'}")
    print(f"{'='*56}")
    return e


# ---------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(description="几何法点云管线")
    p.add_argument("--synthetic", action="store_true", help="用虚拟场景测试")
    p.add_argument("--n-cubes", type=int, default=4)
    p.add_argument("--edge", type=float, default=0.05, help="正方体边长(米)")
    p.add_argument("--noise-mm", type=float, default=2.0, help="深度噪声标准差(mm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-dist", type=float, default=0.3, help="最近距离(m)")
    p.add_argument("--max-dist", type=float, default=1.5, help="最远距离(m)")
    p.add_argument("--plane-thresh", type=float, default=0.003,
                   help="【量法向量用】平面内点阈值(m)。太松会把相邻面吸进来")
    p.add_argument("--table-thresh", type=float, default=0.010,
                   help="【去桌面用】平面内点阈值(m)。桌面噪声大，要比上面松")
    p.add_argument("--voxel", type=float, default=0.02, help="体素边长(m)")
    p.add_argument("--min-cluster-points", type=int, default=0,
                   help="最小簇点数（0=按边长自动）")
    p.add_argument("--depth-noise-coef", type=float, default=0.008,
                   help="深度噪声模型系数：σz ≈ coef·z²（米）。D435i 实测约 0.008。"
                        "平面阈值会自动取 max(--plane-thresh, 2σz)，"
                        "这样远离相机时不会因为阈值太紧而拟合出假平面")
    p.add_argument("--max-faces", type=int, default=3)
    p.add_argument("--gravity", action="store_true",
                   help="输入点云的 +Y 是【物理竖直方向】（Azure 那路开了重力"
                        "对齐时会自动置上）。开了才会输出 lean_deg ——"
                        "物体偏离铅垂面的角度，这才是「物体歪没歪」。\n"
                        "★ 等价于 --up-axis 0,1,0（保留是为兼容老脚本）。")
    # ★★ 三个显式坐标系参数。为什么必须显式而不是写死：同一份 pipeline
    #    同时被三条路用（RealSense 相机系 / 重力对齐竖直系 / 车体系），
    #    硬编码任何一个都会把另外两个弄错，而且大多是【静默】错的。
    #    缺省值 = 今天的行为，所以 RealSense 和 --synthetic 完全不受影响。
    p.add_argument("--up-axis", default=None, metavar="X,Y,Z",
                   help="输入点云里『物理竖直向上』的方向。给了才会算 lean_deg。"
                        "车体系 = 0,0,1；重力对齐过的相机系 = 0,1,0。")
    p.add_argument("--cam-axis", default=None, metavar="X,Y,Z",
                   help="相机光轴在输入点云坐标系里的方向（算 angle_deg / "
                        "挑最正对的面用）。缺省 0,0,1 = 相机系。")
    p.add_argument("--cam-origin", default=None, metavar="X,Y,Z",
                   help="相机光心在输入点云坐标系里的位置（米）。距离滤波、"
                        "法向朝外都靠它。缺省 0,0,0 = 相机系。"
                        "★ 车体系下【必须】给，否则法向会翻反、体心退错 16.7cm，"
                        "而且 cube_ok 仍然是 true。")
    p.add_argument("--face-select", choices=["most_facing", "largest"],
                   default="most_facing",
                   help="识别面选择: most_facing=最正对相机(默认), largest=点数最多")
    p.add_argument("--json", action="store_true", help="只输出 JSON")
    p.add_argument("--dump", default=None, metavar="DIR",
                   help="把每一步的点云导出成 PLY，存到 DIR（用于可视化/教学）")
    p.add_argument("--replay", default=None, metavar="FILE",
                   help="从 PLY 文件读点云替代相机")
    return p.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        print(f"[合成场景] {args.n_cubes} 个正方体, 边长 {args.edge*1000:.0f}mm, "
              f"噪声 σ={args.noise_mm}mm, seed={args.seed}")
        pts, truth = make_synthetic_scene(
            n_cubes=args.n_cubes, edge=args.edge, seed=args.seed,
            noise_sigma=args.noise_mm / 1000.0)
        result = run_pipeline(pts, args, verbose=not args.json)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            evaluate(truth, result["objects"])
        return 0

    if args.replay:
        print(f"[回放] 读取点云: {args.replay}")
        pts = read_ply(args.replay)
        result = run_pipeline(pts, args, verbose=not args.json)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        print(f"[回放] 完成，耗时 {result['timing_ms']} ms")
        return 0

    print("[ERROR] 请指定 --synthetic（虚拟场景）或 --replay FILE（点云文件）")
    print("        实时模式需要深度可用（等电源线），届时接入 libk4a 取深度帧")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
