#include "kvc/geom_internal.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace kvc {
namespace geom {
namespace {

struct Dsu {
    std::vector<int> parent;
    explicit Dsu(size_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] =
                parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[static_cast<size_t>(b)] = a;
    }
};

}  // namespace

std::vector<int> voxel_cluster(const std::vector<Vec3>& pts, double voxel,
                               int min_points) {
    std::vector<int> out(pts.size(), -1);
    if (pts.empty()) return out;

    const size_t n_pts = pts.size();
    std::vector<long long> gx(n_pts), gy(n_pts), gz(n_pts);
    long long mnx = LLONG_MAX, mny = LLONG_MAX, mnz = LLONG_MAX;
    long long mxx = LLONG_MIN, mxy = LLONG_MIN, mxz = LLONG_MIN;
    for (size_t i = 0; i < n_pts; ++i) {
        const long long x = static_cast<long long>(std::floor(pts[i][0] / voxel));
        const long long y = static_cast<long long>(std::floor(pts[i][1] / voxel));
        const long long z = static_cast<long long>(std::floor(pts[i][2] / voxel));
        gx[i] = x;
        gy[i] = y;
        gz[i] = z;
        mnx = std::min(mnx, x);
        mny = std::min(mny, y);
        mnz = std::min(mnz, z);
        mxx = std::max(mxx, x);
        mxy = std::max(mxy, y);
        mxz = std::max(mxz, z);
    }
    const long long shape_x = mxx - mnx + 1;
    const long long shape_y = mxy - mny + 1;
    const long long shape_z = mxz - mnz + 1;
    if (shape_x <= 0 || shape_y <= 0 || shape_z <= 0) return {};

    std::vector<long long> key(n_pts);
    for (size_t i = 0; i < n_pts; ++i) {
        const long long x = gx[i] - mnx;
        const long long y = gy[i] - mny;
        const long long z = gz[i] - mnz;
        key[i] = (x * shape_y + y) * shape_z + z;
    }

    // Unique keys, sorted ascending.  `inv[i]` is the unique-voxel index of
    // point i, matching np.unique(key, return_inverse=True).
    std::vector<size_t> order(n_pts);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return key[a] < key[b];
    });
    std::vector<long long> uk;
    std::vector<int> inv(n_pts);
    for (size_t k = 0; k < n_pts; ++k) {
        const size_t i = order[k];
        if (uk.empty() || key[i] != uk.back()) uk.push_back(key[i]);
        inv[i] = static_cast<int>(uk.size()) - 1;
    }
    const int n_uk = static_cast<int>(uk.size());

    std::vector<long long> ux(static_cast<size_t>(n_uk));
    std::vector<long long> uy(static_cast<size_t>(n_uk));
    std::vector<long long> uz(static_cast<size_t>(n_uk));
    for (int k = 0; k < n_uk; ++k) {
        ux[static_cast<size_t>(k)] = uk[static_cast<size_t>(k)] / (shape_y * shape_z);
        uy[static_cast<size_t>(k)] =
            (uk[static_cast<size_t>(k)] / shape_z) % shape_y;
        uz[static_cast<size_t>(k)] = uk[static_cast<size_t>(k)] % shape_z;
    }

    // Only the 13 "forward" offsets of the 26-neighbourhood; the reverse edges
    // are covered by symmetry.
    static const int off13[13][3] = {
        {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},  {1, -1, 0},
        {1, 0, 1},  {1, 0, -1}, {0, 1, 1},  {0, 1, -1}, {1, 1, 1},
        {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
    };

    Dsu dsu(static_cast<size_t>(n_uk));
    for (const auto& off : off13) {
        const long long dx = off[0], dy = off[1], dz = off[2];
        const long long delta = (dx * shape_y + dy) * shape_z + dz;
        for (int s = 0; s < n_uk; ++s) {
            const long long tgt = uk[static_cast<size_t>(s)] + delta;
            const auto it = std::lower_bound(uk.begin(), uk.end(), tgt);
            if (it == uk.end() || *it != tgt) continue;
            const int d_i = static_cast<int>(it - uk.begin());
            // Guard against linear-key wrap-around in the z/y direction.
            if (ux[static_cast<size_t>(d_i)] - ux[static_cast<size_t>(s)] != dx)
                continue;
            if (uy[static_cast<size_t>(d_i)] - uy[static_cast<size_t>(s)] != dy)
                continue;
            if (uz[static_cast<size_t>(d_i)] - uz[static_cast<size_t>(s)] != dz)
                continue;
            dsu.unite(s, d_i);
        }
    }

    // Renumber components in first-node order, like scipy.ndimage's labels.
    std::vector<int> comp_of_root(static_cast<size_t>(n_uk), -1);
    std::vector<int> comp_vox(static_cast<size_t>(n_uk));
    int n_comp = 0;
    for (int k = 0; k < n_uk; ++k) {
        const int r = dsu.find(k);
        if (comp_of_root[static_cast<size_t>(r)] < 0) {
            comp_of_root[static_cast<size_t>(r)] = n_comp++;
        }
        comp_vox[static_cast<size_t>(k)] = comp_of_root[static_cast<size_t>(r)];
    }

    // Point labels, then drop components with fewer than min_points points.
    std::vector<int> pt_labels(n_pts);
    std::vector<int> counts(static_cast<size_t>(n_comp), 0);
    for (size_t i = 0; i < n_pts; ++i) {
        const int c = comp_vox[static_cast<size_t>(inv[i])];
        pt_labels[i] = c;
        ++counts[static_cast<size_t>(c)];
    }
    std::vector<int> remap(static_cast<size_t>(n_comp), -1);
    int new_id = 0;
    for (int c = 0; c < n_comp; ++c) {
        if (counts[static_cast<size_t>(c)] < min_points) continue;
        remap[static_cast<size_t>(c)] = new_id++;
    }
    for (size_t i = 0; i < n_pts; ++i) {
        out[i] = remap[static_cast<size_t>(pt_labels[i])];
    }
    return out;
}

}  // namespace geom
}  // namespace kvc
