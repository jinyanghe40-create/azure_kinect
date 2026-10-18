#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace kvc_test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& extra = "") {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL " << file << ":" << line << "  " << expr;
    if (!extra.empty()) std::cerr << "  [" << extra << "]";
    std::cerr << std::endl;
}

inline int finish(const char* name) {
    if (g_failures == 0) {
        std::cout << name << ": " << g_checks << " checks passed" << std::endl;
        return 0;
    }
    std::cerr << name << ": " << g_failures << " / " << g_checks
              << " checks FAILED" << std::endl;
    return 1;
}

}  // namespace kvc_test

#define CHECK(cond) \
    ::kvc_test::report((cond), #cond, __FILE__, __LINE__)

#define CHECK_MSG(cond, msg) \
    ::kvc_test::report((cond), #cond, __FILE__, __LINE__, (msg))

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        const double _a = static_cast<double>(a);                              \
        const double _b = static_cast<double>(b);                              \
        const double _t = static_cast<double>(tol);                            \
        ::kvc_test::report(std::fabs(_a - _b) <= _t, #a " ~= " #b, __FILE__,   \
                           __LINE__, "got " + std::to_string(_a) + " vs " +    \
                                          std::to_string(_b));                 \
    } while (0)

#define CHECK_MAT_NEAR(A, B, tol)                                              \
    do {                                                                       \
        const double _d = ((A) - (B)).cwiseAbs().maxCoeff();                   \
        ::kvc_test::report(_d <= (tol), #A " ~= " #B, __FILE__, __LINE__,      \
                           "max diff " + std::to_string(_d));                  \
    } while (0)
