// verify_libs_cpp — port of azure_kinect/verify_libs.py.
// No SDK headers needed: this tool only inspects files and dlopen()s sonames.

#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char* kLibDir = "/usr/lib/aarch64-linux-gnu";
constexpr const char* kPass = "✅";
constexpr const char* kFail = "❌";
constexpr const char* kWarn = "⚠️ ";

std::vector<std::string> g_problems;

void hdr(int n, const char* title) {
    std::printf("\n%s\n[%d] %s\n%s\n", std::string(62, '-').c_str(), n, title,
                std::string(62, '-').c_str());
}

bool file_size_mb(const std::string& p, double* mb) {
    struct stat st {};
    if (::stat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    *mb = static_cast<double>(st.st_size) / 1048576.0;
    return true;
}

// Equivalent of subprocess.run(["ldd", p]) and scanning for "not found".
std::string run_ldd(const std::string& path, bool* ok) {
    const std::string cmd = "ldd '" + path + "' 2>&1";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) {
        *ok = false;
        return {};
    }
    std::string out;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), f)) out += buf;
    *ok = ::pclose(f) == 0;
    return out;
}

void check_load(const char* soname) {
    void* h = ::dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (h) {
        std::printf("  %s %s 加载成功\n", kPass, soname);
        ::dlclose(h);
    } else {
        const char* e = ::dlerror();
        std::printf("  %s %s 加载失败\n", kFail, soname);
        std::printf("     %s\n", e ? e : "?");
        g_problems.push_back(std::string(soname) + " 加载失败");
    }
}

}  // namespace

int main() {
    std::printf("==============================================================\n");
    std::printf(" Azure Kinect ARM64 库 —— 安装验证\n");
    std::printf("==============================================================\n");

    // ---------- 1. 架构 ----------
    hdr(1, "架构检查");
    struct utsname u {};
    const char* arch = (::uname(&u) == 0) ? u.machine : "?";
    if (std::strcmp(arch, "aarch64") == 0) {
        std::printf("  %s 架构 %s（Jetson 正确）\n", kPass, arch);
    } else {
        std::printf("  %s 架构是 %s，不是 aarch64\n", kFail, arch);
        std::printf("     这个包只能在 Jetson / ARM64 上用\n");
        g_problems.push_back("架构不对");
        std::printf("\n==============================================================\n");
        std::printf(" ⚠️  发现 %zu 个问题：\n", g_problems.size());
        for (const auto& p : g_problems) std::printf("     · %s\n", p.c_str());
        return 1;
    }

    // ---------- 2. 文件是否存在 ----------
    hdr(2, "库文件是否存在");
    const std::vector<std::string> libs = {"libk4a.so.1.4.2",
                                           "libk4arecord.so.1.4.2",
                                           "libdepthengine.so.2.0"};
    for (const auto& f : libs) {
        const std::string p = std::string(kLibDir) + "/" + f;
        double mb = 0.0;
        if (file_size_mb(p, &mb)) {
            std::printf("  %s %-28s %5.1f MB\n", kPass, f.c_str(), mb);
        } else {
            std::printf("  %s %-28s 缺失\n", kFail, f.c_str());
            std::printf("     修复: sudo bash deploy_setup.sh\n");
            g_problems.push_back(f + " 缺失");
        }
    }

    // ---------- 3. 软链接 ----------
    hdr(3, "软链接是否正确");
    struct Link {
        const char* real;
        const char* link;
    };
    const std::vector<Link> links = {
        {"libk4a.so.1.4.2", "libk4a.so.1.4"},
        {"libk4arecord.so.1.4.2", "libk4arecord.so.1.4"},
        {"libdepthengine.so.2.0", "libdepthengine.so.2.0"}};
    for (const auto& l : links) {
        const std::string linkp = std::string(kLibDir) + "/" + l.link;
        struct stat st {};
        if (::lstat(linkp.c_str(), &st) == 0 && S_ISLNK(st.st_mode)) {
            char buf[1024];
            const ssize_t n = ::readlink(linkp.c_str(), buf, sizeof(buf) - 1);
            const std::string tgt = (n > 0) ? std::string(buf, n) : "?";
            const std::string base =
                tgt.substr(tgt.find_last_of('/') == std::string::npos
                               ? 0
                               : tgt.find_last_of('/') + 1);
            const bool ok = base == l.real || tgt.size() >= std::strlen(l.real) &&
                                              tgt.compare(tgt.size() - std::strlen(l.real),
                                                          std::strlen(l.real), l.real) == 0;
            std::printf("  %s %-28s -> %s\n", ok ? kPass : kWarn, l.link,
                        tgt.c_str());
        } else if (::stat(linkp.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            std::printf("  %s %-28s 是实体文件\n", kPass, l.link);
        } else {
            std::printf("  %s %-28s 不存在\n", kFail, l.link);
            std::printf("     修复: cd %s && sudo ln -sf %s %s\n", kLibDir,
                        l.real, l.link);
            g_problems.push_back(std::string("软链接 ") + l.link + " 缺失");
        }
    }

    // ---------- 4. 依赖是否齐全（ldd）----------
    hdr(4, "依赖库是否齐全（这一步最容易出问题）");
    for (const char* f : {"libk4a.so.1.4.2", "libdepthengine.so.2.0"}) {
        const std::string p = std::string(kLibDir) + "/" + f;
        double mb = 0.0;
        if (!file_size_mb(p, &mb)) continue;
        std::printf("\n  --- %s ---\n", f);
        bool ok = false;
        const std::string out = run_ldd(p, &ok);
        if (!ok) {
            std::printf("  %s ldd 跑不了\n", kWarn);
            continue;
        }
        std::vector<std::string> missing;
        int dep_count = 0;
        size_t pos = 0;
        while (pos <= out.size()) {
            const size_t nl = out.find('\n', pos);
            const std::string line =
                out.substr(pos, nl == std::string::npos ? std::string::npos
                                                        : nl - pos);
            if (line.find("not found") != std::string::npos) {
                missing.push_back(line);
            }
            if (line.find("=>") != std::string::npos) ++dep_count;
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        if (!missing.empty()) {
            std::printf("  %s 缺少 %zu 个依赖:\n", kFail, missing.size());
            for (const auto& m : missing) std::printf("       %s\n", m.c_str());
            g_problems.push_back(std::string(f) + " 缺依赖");
        } else {
            std::printf("  %s 依赖齐全（%d 个）\n", kPass, dep_count);
        }
    }

    // ---------- 5. 实际加载（最关键）----------
    hdr(5, "实际加载测试（最关键）");
    check_load("libk4a.so.1.4");
    check_load("libk4arecord.so.1.4");

    void* depth = ::dlopen("libdepthengine.so.2.0", RTLD_NOW | RTLD_LOCAL);
    if (depth) {
        std::printf("  %s libdepthengine.so.2.0 加载成功  ← 深度功能靠它\n",
                    kPass);
        ::dlclose(depth);
    } else {
        const char* e = ::dlerror();
        std::printf("  %s libdepthengine.so.2.0 加载失败  ← 深度会不可用\n",
                    kFail);
        std::printf("     %s\n", e ? e : "?");
        std::printf("     常见原因: 缺 libGL.so.1 或 libX11.so.6\n");
        std::printf("     修复: sudo apt install libgl1-mesa-glx libx11-6\n");
        g_problems.push_back("深度引擎加载失败");
    }

    // ---------- 6. k4a API 是否可用 ----------
    hdr(6, "k4a API 符号检查");
    void* k4a = ::dlopen("libk4a.so.1.4", RTLD_NOW | RTLD_LOCAL);
    if (!k4a) {
        std::printf("  %s 无法打开 libk4a\n", kFail);
        g_problems.push_back("libk4a 打不开");
    } else {
        const std::vector<const char*> need = {
            "k4a_device_get_installed_count", "k4a_device_open",
            "k4a_device_start_cameras", "k4a_device_get_capture",
            "k4a_calibration_3d_to_3d", "k4a_calibration_2d_to_3d",
            "k4a_calibration_color_2d_to_depth_2d",
            "k4a_transformation_depth_image_to_point_cloud"};
        int ok = 0;
        for (const char* s : need) {
            if (::dlsym(k4a, s)) {
                ++ok;
            } else {
                std::printf("  %s 缺符号 %s\n", kWarn, s);
            }
        }
        std::printf("  %s %d/%zu 个关键 API 可用\n",
                    ok == static_cast<int>(need.size()) ? kPass : kWarn, ok,
                    need.size());
        if (ok < static_cast<int>(need.size())) {
            g_problems.push_back("部分 API 符号缺失");
        }
        ::dlclose(k4a);
    }

    // ---------- 7. 设备检查 ----------
    hdr(7, "设备连接检查");
    void* dev = ::dlopen("libk4a.so.1.4", RTLD_NOW | RTLD_LOCAL);
    if (!dev) {
        std::printf("  %s 检测失败: 无法打开 libk4a\n", kWarn);
    } else {
        using CountFn = unsigned int (*)();
        auto count = reinterpret_cast<CountFn>(
            ::dlsym(dev, "k4a_device_get_installed_count"));
        if (!count) {
            std::printf("  %s 检测失败: 找不到计数函数\n", kWarn);
        } else {
            const unsigned int n = count();
            if (n > 0) {
                std::printf("  %s 检测到 %u 个 Azure Kinect 设备\n", kPass, n);
            } else {
                std::printf("  %s 没检测到设备\n", kWarn);
                std::printf("     如果 Kinect 没插，这是正常的\n");
                std::printf("     如果插了但读不到，跑: lsusb | grep 045e\n");
            }
        }
        ::dlclose(dev);
    }

    // ---------- 总结 ----------
    std::printf("\n==============================================================\n");
    if (g_problems.empty()) {
        std::printf(" 🎉 全部通过！库装好了，可以开始用相机了\n");
        std::printf("==============================================================\n");
        std::printf("\n下一步：确认相机能出彩色和深度\n");
        std::printf("     ./k4a_probe_cpp\n");
        return 0;
    }
    std::printf(" ⚠️  发现 %zu 个问题：\n", g_problems.size());
    for (const auto& p : g_problems) std::printf("     · %s\n", p.c_str());
    std::printf("==============================================================\n");
    std::printf("\n按上面的提示修复，然后重新跑本脚本\n");
    return 1;
}
