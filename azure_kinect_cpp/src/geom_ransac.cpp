#include "kvc/geom_internal.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace kvc {
namespace geom {
namespace {

Vec3 mean_of(const std::vector<Vec3>& v) {
    Vec3 s = Vec3::Zero();
    for (const Vec3& p : v) s += p;
    return v.empty() ? s : s / static_cast<double>(v.size());
}

// Covariance of the given points around c.  numpy's np.cov uses ddof=1, but
// eigenvectors only depend on the shape, so the scalar factor is irrelevant.
Eigen::Matrix3d covariance(const std::vector<Vec3>& pts, const Vec3& c) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const Vec3& p : pts) {
        const Vec3 d = p - c;
        cov += d * d.transpose();
    }
    if (pts.size() > 1) cov /= static_cast<double>(pts.size() - 1);
    return cov;
}

}  // namespace

std::optional<PlaneFit> ransac_plane(const std::vector<Vec3>& pts,
                                     double thresh, int iters,
                                     int min_inliers,
                                     const std::optional<Vec3>& normal_hint,
                                     double cos_tol, int max_fit_pts) {
    const int n_pts = static_cast<int>(pts.size());
    // Python: if n_pts < 3 or n_pts < min_inliers.  The first guard matters
    // when min_inliers has degenerated to 0 on an empty cloud.
    if (n_pts < 3 || n_pts < min_inliers) return std::nullopt;

    // RANSAC is statistical; fitting on 20k points is plenty.  Python uses
    // rng.choice(..., replace=False); we use a partial Fisher-Yates shuffle.
    std::mt19937 rng(0);
    std::vector<Vec3> fit;
    if (n_pts > max_fit_pts) {
        std::vector<int> idx(static_cast<size_t>(n_pts));
        for (int i = 0; i < n_pts; ++i) idx[static_cast<size_t>(i)] = i;
        for (int i = 0; i < max_fit_pts; ++i) {
            std::uniform_int_distribution<int> pick(i, n_pts - 1);
            std::swap(idx[static_cast<size_t>(i)], idx[static_cast<size_t>(pick(rng))]);
        }
        fit.reserve(static_cast<size_t>(max_fit_pts));
        for (int i = 0; i < max_fit_pts; ++i) fit.push_back(pts[static_cast<size_t>(idx[static_cast<size_t>(i)])]);
    } else {
        fit = pts;
    }
    const int n_fit = static_cast<int>(fit.size());

    std::vector<Vec3> cand_n;
    std::vector<double> cand_d;
    cand_n.reserve(static_cast<size_t>(iters));
    cand_d.reserve(static_cast<size_t>(iters));
    std::uniform_int_distribution<int> pick_idx(0, n_fit - 1);
    for (int it = 0; it < iters; ++it) {
        const Vec3 p0 = fit[static_cast<size_t>(pick_idx(rng))];
        const Vec3 p1 = fit[static_cast<size_t>(pick_idx(rng))];
        const Vec3 p2 = fit[static_cast<size_t>(pick_idx(rng))];
        Vec3 nrm = (p1 - p0).cross(p2 - p0);
        const double ln = nrm.norm();
        if (!(ln > 1e-12)) continue;
        nrm /= ln;
        if (normal_hint && std::abs(nrm.dot(*normal_hint)) < cos_tol) continue;
        cand_n.push_back(nrm);
        cand_d.push_back(-nrm.dot(p0));
    }
    if (cand_n.empty()) return std::nullopt;

    // Score every candidate on the fit subset (argmax with first-wins ties).
    int best_k = 0;
    int best_cnt = -1;
    for (size_t k = 0; k < cand_n.size(); ++k) {
        int cnt = 0;
        for (const Vec3& p : fit) {
            if (std::abs(p.dot(cand_n[k]) + cand_d[k]) < thresh) ++cnt;
        }
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best_k = static_cast<int>(k);
        }
    }
    Vec3 n = cand_n[static_cast<size_t>(best_k)];
    double d = cand_d[static_cast<size_t>(best_k)];

    // Confirm on the full cloud exactly once.
    {
        int full = 0;
        for (const Vec3& p : pts) {
            if (std::abs(p.dot(n) + d) < thresh) ++full;
        }
        if (full < min_inliers) return std::nullopt;
    }

    // Self-consistent tightening: choose inliers -> re-fit their PCA plane.
    // Repositioning with the inlier centroid (not the global one) is essential.
    const double tights[4] = {thresh, thresh * 0.6, thresh * 0.4, thresh * 0.3};
    for (double tight : tights) {
        std::vector<Vec3> inl;
        inl.reserve(pts.size());
        for (const Vec3& p : pts) {
            if (std::abs(p.dot(n) + d) < tight) inl.push_back(p);
        }
        if (static_cast<int>(inl.size()) < 50) break;
        const Vec3 c = mean_of(inl);
        const Eigen::Matrix3d cov = covariance(inl, c);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        if (es.info() != Eigen::Success) break;
        Vec3 n_new = es.eigenvectors().col(0);
        const double nn = n_new.norm();
        if (!(nn > 0.0)) break;
        n_new /= nn;
        if (n_new.dot(n) < 0.0) n_new = -n_new;
        n = n_new;
        d = -n.dot(c);  // reposition on the inlier centroid
    }

    // Refinement can rotate the normal, so the hint must be re-checked.  Skipping
    // this once ate whole cubes ("biggest plane = cube face removed as table").
    if (normal_hint && std::abs(n.dot(*normal_hint)) < cos_tol) {
        return std::nullopt;
    }

    PlaneFit out;
    out.normal = n;
    out.d = d;
    out.mask.assign(pts.size(), 0);
    int cnt = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (std::abs(pts[i].dot(n) + d) < thresh) {
            out.mask[i] = 1;
            ++cnt;
        }
    }
    if (cnt < min_inliers / 2) return std::nullopt;
    return out;
}

Vec3 orient_toward_camera(const Vec3& n, const std::vector<Vec3>& inl,
                          const Vec3& cam_origin) {
    const Vec3 c = mean_of(inl);
    if (n.dot(cam_origin - c) < 0.0) return -n;
    return n;
}

}  // namespace geom
}  // namespace kvc
