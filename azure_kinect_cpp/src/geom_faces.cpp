#include "kvc/geom_internal.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace kvc {
namespace geom {
namespace {

constexpr double kPi = 3.14159265358979323846;

Vec3 mean_of(const std::vector<Vec3>& v) {
    Vec3 s = Vec3::Zero();
    for (const Vec3& p : v) s += p;
    return v.empty() ? s : s / static_cast<double>(v.size());
}

}  // namespace

std::vector<ClusterFace> analyze_cluster(const std::vector<Vec3>& pts,
                                         double dist_thresh, int max_faces,
                                         const Vec3& cam_origin) {
    std::vector<ClusterFace> faces;
    std::vector<Vec3> remaining = pts;
    for (int it = 0; it < max_faces; ++it) {
        if (remaining.size() < 80) break;
        // Python uses 80 iterations here.  A cube face at ~1 m with 2 mm noise
        // occasionally draws a tilted triplet whose PCA tightening then locks
        // onto a slightly rotated plane; 400 iterations cut that tail enough
        // that every oracle fixture matches within tolerance (see README).
        auto fit = ransac_plane(remaining, dist_thresh, 400, 100, std::nullopt);
        if (!fit) break;
        Vec3 n = fit->normal;
        if (!faces.empty()) {
            // Cube faces are near-perpendicular; |cos| must stay close to 0.
            bool ok = true;
            for (const ClusterFace& f : faces) {
                if (std::abs(n.dot(f.normal)) > 0.35) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        std::vector<Vec3> inl;
        inl.reserve(remaining.size());
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (fit->mask[i]) inl.push_back(remaining[i]);
        }
        if (inl.size() < 50) break;
        Vec3 c = mean_of(inl);
        n = orient_toward_camera(n, inl, cam_origin);
        double d = -n.dot(c);
        Mask mask(remaining.size(), 0);
        inl.clear();
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (std::abs(remaining[i].dot(n) + d) < dist_thresh) {
                mask[i] = 1;
                inl.push_back(remaining[i]);
            }
        }
        if (inl.size() < 50) break;
        c = mean_of(inl);
        d = -n.dot(c);
        double sum2 = 0.0;
        for (const Vec3& p : inl) {
            const double r = std::abs(p.dot(n) + d);
            sum2 += r * r;
        }
        const double rms = std::sqrt(sum2 / static_cast<double>(inl.size()));
        Vec3 lo = inl.front();
        Vec3 hi = inl.front();
        for (const Vec3& p : inl) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        ClusterFace f;
        f.normal = n;
        f.center = c;
        f.n_points = static_cast<int>(inl.size());
        f.rms = rms;
        f.extent = (hi - lo).norm();
        faces.push_back(f);

        std::vector<Vec3> next;
        next.reserve(remaining.size() - inl.size());
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (!mask[i]) next.push_back(remaining[i]);
        }
        remaining.swap(next);
    }
    // Stable descending sort, like Python's list.sort(key=-n_points).
    std::stable_sort(faces.begin(), faces.end(),
                     [](const ClusterFace& a, const ClusterFace& b) {
                         return a.n_points > b.n_points;
                     });
    return faces;
}

double face_edge_len(const std::vector<Vec3>& dp, const Vec3& e1,
                     const Vec3& e2) {
    // Projection width of a convex shape is minimised exactly at its width
    // (a square's width equals its side for every rotation).  0.5 deg steps.
    double best = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 361; ++k) {
        const double ang = (kPi / 361.0) * static_cast<double>(k);
        const Vec3 dir = std::cos(ang) * e1 + std::sin(ang) * e2;
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (const Vec3& p : dp) {
            const double v = p.dot(dir);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        best = std::min(best, hi - lo);
    }
    return best;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void measure_edge(const std::vector<ClusterFace>& faces,
                  const std::vector<Vec3>& pts, double tol, int min_pts,
                  double spread_tol, std::optional<double>* edge,
                  std::vector<double>* vals, std::string* src) {
    vals->clear();
    for (const ClusterFace& f : faces) {
        Vec3 n = f.normal;
        const double nn = n.norm();
        n /= std::max(1e-9, nn);
        const Vec3 c = f.center;
        std::vector<Vec3> on;
        for (const Vec3& p : pts) {
            const Vec3 d = p - c;
            if (std::abs(d.dot(n)) < tol) on.push_back(d);
        }
        if (static_cast<int>(on.size()) < min_pts) continue;
        // Remove the normal component: project onto the face plane.
        std::vector<Vec3> dp;
        dp.reserve(on.size());
        for (const Vec3& d : on) dp.push_back(d - d.dot(n) * n);
        const Vec3 ref = (std::abs(n[1]) < 0.9) ? Vec3(0.0, 1.0, 0.0)
                                                : Vec3(1.0, 0.0, 0.0);
        Vec3 e1 = n.cross(ref);
        e1 /= std::max(1e-9, e1.norm());
        const Vec3 e2 = n.cross(e1);
        vals->push_back(face_edge_len(dp, e1, e2));
    }
    if (vals->empty()) {
        *edge = std::nullopt;
        *src = "cli";
        return;
    }
    if (vals->size() >= 2) {
        const double lo = *std::min_element(vals->begin(), vals->end());
        const double hi = *std::max_element(vals->begin(), vals->end());
        if (hi > 0.0 && (hi - lo) / hi > spread_tol) {
            // Occlusion only makes a measured face smaller, so the largest
            // face is the best guess; flag it as less trustworthy.
            *edge = hi;
            *src = "auto-max";
            return;
        }
    }
    *edge = median(*vals);
    *src = "auto";
}

std::optional<Vec3> cube_center_from_faces(const std::vector<ClusterFace>& faces,
                                           double edge_len,
                                           const std::vector<Vec3>& cluster_pts) {
    if (faces.empty() || !(edge_len > 0.0)) return std::nullopt;
    std::vector<Vec3> ns;
    std::vector<Vec3> cs;
    ns.reserve(faces.size());
    cs.reserve(faces.size());
    for (const ClusterFace& f : faces) {
        ns.push_back(f.normal / std::max(1e-9, f.normal.norm()));
        cs.push_back(f.center);
    }

    if (faces.size() >= 3) {
        Eigen::Matrix3d A;
        Eigen::Vector3d b;
        for (int i = 0; i < 3; ++i) {
            A.row(i) = ns[static_cast<size_t>(i)].transpose();
            b[i] = ns[static_cast<size_t>(i)].dot(cs[static_cast<size_t>(i)]);
        }
        Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
        if (lu.isInvertible()) {
            const Vec3 vertex = lu.solve(b);
            return vertex - (edge_len / 2.0) *
                                (ns[0] + ns[1] + ns[2]);
        }
        // Degenerate (three coplanar faces) -> generic branch below.
    }

    Vec3 p;
    if (!cluster_pts.empty()) {
        p = mean_of(cluster_pts);
    } else {
        Vec3 s = Vec3::Zero();
        for (const Vec3& c : cs) s += c;
        p = s / static_cast<double>(cs.size());
    }
    for (size_t i = 0; i < ns.size(); ++i) {
        const double target = ns[i].dot(cs[i]) - edge_len / 2.0;
        p += (target - p.dot(ns[i])) * ns[i];
    }
    return p;
}

}  // namespace geom
}  // namespace kvc
