#include "kvc/geom_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace kvc {
namespace geom {

std::vector<int> split_clusters_by_boxes(
    const std::vector<Vec3>& pts, const std::vector<int>& labels,
    int n_clusters, const std::vector<Eigen::Vector2d>& uv,
    const std::vector<char>& uv_valid, const std::vector<Detection>& boxes,
    int min_pts, int* n_out, bool verbose) {
    *n_out = n_clusters;
    if (n_clusters == 0 || boxes.empty()) return labels;
    if (uv.size() != pts.size() || uv_valid.size() != pts.size()) {
        if (verbose) std::printf("[4b] 按检测框切簇：uv 对不上或全无效，跳过\n");
        return labels;
    }
    bool any_valid = false;
    for (char v : uv_valid) any_valid = any_valid || v;
    if (!any_valid) {
        if (verbose) std::printf("[4b] 按检测框切簇：uv 对不上或全无效，跳过\n");
        return labels;
    }

    const size_t n_pts = pts.size();
    const size_t n_b = boxes.size();
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> bcx(n_b), bcy(n_b), marg(n_b);
    for (size_t b = 0; b < n_b; ++b) {
        const Detection& r = boxes[b];
        bcx[b] = 0.5 * (r.x0 + r.x1);
        bcy[b] = 0.5 * (r.y0 + r.y1);
        marg[b] = 0.25 * std::hypot(r.x1 - r.x0, r.y1 - r.y0);
    }

    std::vector<int> owner(n_pts, -1);
    std::vector<char> has_box(n_pts, 0);
    for (size_t i = 0; i < n_pts; ++i) {
        if (!uv_valid[i]) continue;  // NaN in Python: never inside, never near
        const double u = uv[i][0];
        const double v = uv[i][1];
        int box_of = -1;
        double best_d2 = inf;
        int box_near = -1;
        double best_dout = inf;
        for (size_t b = 0; b < n_b; ++b) {
            const Detection& r = boxes[b];
            const bool inside = (u >= r.x0 && u <= r.x1 && v >= r.y0 && v <= r.y1);
            if (inside) {
                const double d2 = (u - bcx[b]) * (u - bcx[b]) +
                                  (v - bcy[b]) * (v - bcy[b]);
                if (box_of < 0 || d2 < best_d2) {
                    best_d2 = d2;
                    box_of = static_cast<int>(b);
                }
                continue;
            }
            const double dx = std::max(std::max(r.x0 - u, u - r.x1), 0.0);
            const double dy = std::max(std::max(r.y0 - v, v - r.y1), 0.0);
            const double dout = std::sqrt(dx * dx + dy * dy);
            if (dout <= marg[b] && dout < best_dout) {
                best_dout = dout;
                box_near = static_cast<int>(b);
            }
        }
        if (box_of >= 0) {
            has_box[i] = 1;
            owner[i] = box_of;
        } else {
            owner[i] = box_near;  // -1 when every box is too far away
        }
    }

    std::vector<int> new_labels(n_pts, -1);
    int n_new = 0;
    int n_kept = 0;
    std::vector<std::array<int, 3>> parts;  // (cluster, box, count)
    for (int cid = 0; cid < n_clusters; ++cid) {
        std::vector<size_t> idx;
        bool cluster_has_box = false;
        for (size_t i = 0; i < n_pts; ++i) {
            if (labels[i] != cid) continue;
            idx.push_back(i);
            if (has_box[i]) cluster_has_box = true;
        }
        if (idx.empty()) continue;
        if (!cluster_has_box) {  // no box touched this cluster -> keep as-is
            for (size_t i : idx) new_labels[i] = n_new;
            ++n_new;
            ++n_kept;
            continue;
        }
        std::vector<int> owners;
        owners.reserve(idx.size());
        for (size_t i : idx) owners.push_back(owner[i]);
        std::sort(owners.begin(), owners.end());
        owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
        for (int b : owners) {
            if (b < 0) continue;  // outside every box and too far -> drop
            std::vector<size_t> sel;
            for (size_t i : idx) {
                if (owner[i] == b) sel.push_back(i);
            }
            if (static_cast<int>(sel.size()) < min_pts) continue;
            for (size_t i : sel) new_labels[i] = n_new;
            parts.push_back({cid, b, static_cast<int>(sel.size())});
            ++n_new;
        }
    }

    if (n_new == 0) return labels;

    int n_drop = 0;
    for (size_t i = 0; i < n_pts; ++i) {
        if (new_labels[i] < 0) ++n_drop;
    }
    if (verbose) {
        std::printf(
            "[4b] 按检测框切簇（%zu 个框，最少 %d 点）：%d 簇 → %d 块，丢 %d 点",
            n_b, min_pts, n_clusters, n_new, n_drop);
        if (n_kept) std::printf("（%d 簇没碰到框，原样保留）", n_kept);
        std::printf("\n");
        for (const auto& p : parts) {
            std::printf("     簇%d × 框%d(%d) → %d 点\n", p[0], p[1],
                        boxes[static_cast<size_t>(p[1])].cls_id, p[2]);
        }
    }
    *n_out = n_new;
    return new_labels;
}

}  // namespace geom
}  // namespace kvc
