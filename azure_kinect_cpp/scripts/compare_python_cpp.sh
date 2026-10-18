#!/usr/bin/env bash
# compare_python_cpp.sh — run the same scene through both implementations and
# report per-field differences.
#
# Usage (ON THE BOARD):
#   bash ~/work/azure_kinect_cpp/scripts/compare_python_cpp.sh [N] [extra args…]
#
# N defaults to 20.  Extra args are forwarded to BOTH programs, so only pass
# flags that exist in both (--rotate --stride --min-dist --max-dist
# --min-cluster-points --only-classified --no-gravity …).
#
# Keep the scene still while this runs: the two programs run one after the
# other (the camera can only be opened by one process at a time), so motion
# between the runs shows up as an "error" that is not a code difference.
#
# Exit code 0 = the run completed and the report was written; the report's
# own pass/fail verdict is on the last lines.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/run_azure_cpp}"
# The Python reference implementation is not part of this repository.
PY="${PY:-}"
PYTHON="${PYTHON:-python3}"
if [[ -z "$PY" ]]; then
    echo "❌ 需要 PY=/path/to/run_azure.py（Python 参考实现不在本仓库）" >&2
    exit 2
fi
N="${1:-20}"
case "$N" in ''|*[!0-9]*) echo "第一个参数必须是帧数" >&2; exit 2;; esac
shift || true

CALIB="${CALIB:-$ROOT/body_calib.json}"
OUT="${OUT_DIR:-/tmp/kvc-compare}"
mkdir -p "$OUT"

for f in "$BIN" "$PY" "$CALIB"; do
    [[ -e "$f" ]] || { echo "❌ 找不到 $f" >&2; exit 2; }
done

# 相机倒装：rotate 必须与 body_calib.json 的 image_rotate_deg 一致（现在是 180）。
COMMON=(--json --frames "$N" --rotate 180 --calib "$CALIB" --no-gravity
        --stride 2 --min-dist 0.3 --max-dist 1.5 --min-cluster-points 200
        --only-classified "$@")

echo "== Python: $PY ${COMMON[*]}"
# NOTE: never add -I here: on the board it swaps numpy 1.19.5 for 1.13.3 and
# drops pyserial from sys.path.
"$PYTHON" "$PY" "${COMMON[@]}" > "$OUT/python.jsonl"

echo "== C++: $BIN ${COMMON[*]}"
"$BIN" "${COMMON[@]}" > "$OUT/cpp.jsonl"

"$PYTHON" - "$OUT/python.jsonl" "$OUT/cpp.jsonl" "$N" <<'PYEOF'
import json, sys, math

py_path, cpp_path, want = sys.argv[1], sys.argv[2], int(sys.argv[3])

def load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "obj" in r:
                recs.append(r)
    return recs

pys, cpps = load(py_path), load(cpp_path)
print("\n" + "=" * 72)
print(f" 帧数: Python {len(pys)}  C++ {len(cpps)}  (期望 {want})")
print("=" * 72)

if not pys or not cpps:
    print("❌ 至少一边没有 JSONL —— 先看上面的程序输出")
    sys.exit(1)

def objs(r):
    return [o for o in r.get("obj", []) if isinstance(o, dict)]

def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

pairs = []
n_py = n_cpp = 0
for rp, rc in zip(pys, cpps):
    op, oc = objs(rp), objs(rc)
    n_py += len(op); n_cpp += len(oc)
    used = set()
    for a in op:
        best, bj = None, None
        for j, b in enumerate(oc):
            if j in used or b.get("cls") != a.get("cls"):
                continue
            d = dist(a.get("xyz", [0, 0, 0]), b.get("xyz", [0, 0, 0]))
            if best is None or d < best:
                best, bj = d, j
        if bj is None:
            pairs.append(("only-py", a, None, None))
        else:
            used.add(bj)
            pairs.append(("pair", a, oc[bj], best))
    for j, b in enumerate(oc):
        if j not in used:
            pairs.append(("only-cpp", None, b, None))

matched = [p for p in pairs if p[0] == "pair"]
only_py = [p for p in pairs if p[0] == "only-py"]
only_cpp = [p for p in pairs if p[0] == "only-cpp"]

print(f" 物体总数: Python {n_py}  C++ {n_cpp}")
print(f" 配对成功 {len(matched)}   仅 Python {len(only_py)}   仅 C++ {len(only_cpp)}")

if matched:
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        vals.sort()
        return f"中位 {vals[len(vals)//2]:.1f}  最大 {vals[-1]:.1f}"

    dxyz = [p[3] * 1000.0 for p in matched]
    dang = [abs(p[1].get("angle", 0) - p[2].get("angle", 0)) for p in matched]
    dlean = [abs(p[1].get("lean", 0) - p[2].get("lean", 0)) for p in matched]
    dscore = [abs(p[1].get("score", 0) - p[2].get("score", 0)) for p in matched]
    dpos = sum(1 for p in matched if p[1].get("pos_type") != p[2].get("pos_type"))

    print("\n 配对物体的差异（Python 为参考）")
    print(f"   中心距 |Δxyz|   {stats(dxyz)}  mm")
    print(f"   夹角 |Δangle|   {stats(dang)}  °")
    print(f"   倾角 |Δlean|    {stats(dlean)}  °")
    print(f"   分数 |Δscore|   {stats([x*1000 for x in dscore])}  ‰")
    print(f"   pos_type 不同   {dpos} / {len(matched)}")

    worst = max(dxyz) if dxyz else 0.0
    wa = max(dang) if dang else 0.0
    ok_count = (len(only_py) + len(only_cpp) <= max(1, n_py // 10))
    ok_xyz = worst <= 20.0
    ok_ang = wa <= 1.0
    print("\n 判定（容差来自计划 M4）")
    print(f"   类别配对: {'✅' if ok_count else '❌'} "
          f"（漏/多 {(len(only_py)+len(only_cpp))/max(n_py,1)*100:.0f}% ≤ 10%）")
    print(f"   中心 ≤20mm: {'✅' if ok_xyz else '❌'} (最大 {worst:.1f} mm)")
    print(f"   角度 ≤1°:   {'✅' if ok_ang else '❌'} (最大 {wa:.2f}°)")

if only_py:
    print("\n 仅 Python 检出的物体（前 5 个）：")
    for _, a, _, _ in only_py[:5]:
        print(f"   cls={a.get('cls')} score={a.get('score')} xyz={a.get('xyz')}")
if only_cpp:
    print("\n 仅 C++ 检出的物体（前 5 个）：")
    for _, _, b, _ in only_cpp[:5]:
        print(f"   cls={b.get('cls')} score={b.get('score')} xyz={b.get('xyz')}")

print("\n 原始数据:")
print(f"   {py_path}\n   {cpp_path}")
print("=" * 72)
PYEOF
