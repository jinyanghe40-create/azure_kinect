#include "kvc/json_util.hpp"

namespace kvc {

ojson vec_json(const Vec3& v, int ndigits) {
    ojson a = ojson::array();
    a.push_back(python_round(v[0], ndigits));
    a.push_back(python_round(v[1], ndigits));
    a.push_back(python_round(v[2], ndigits));
    return a;
}

ojson opt_json(const std::optional<double>& v, int ndigits) {
    if (!v.has_value()) return nullptr;
    return python_round(*v, ndigits);
}

ojson opt_vec_json(const std::optional<Vec3>& v, int ndigits) {
    if (!v.has_value()) return nullptr;
    return vec_json(*v, ndigits);
}

std::string compact_dump(const ojson& j) {
    return j.dump(-1, ' ', false);
}

}  // namespace kvc
