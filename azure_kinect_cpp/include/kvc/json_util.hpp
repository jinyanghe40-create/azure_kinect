#pragma once

#include "kvc/body_calib.hpp"  // ojson
#include "kvc/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kvc {

// Rounded 3-vector, e.g. [0.25,0.0,0.6].
ojson vec_json(const Vec3& v, int ndigits);

// Optional scalar with Python-style rounding; nullopt -> null.
ojson opt_json(const std::optional<double>& v, int ndigits);
ojson opt_vec_json(const std::optional<Vec3>& v, int ndigits);

// Python's json.dumps(..., ensure_ascii=False, separators=(",", ":")).
std::string compact_dump(const ojson& j);

}  // namespace kvc
