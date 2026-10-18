#include "kvc/serial_out.hpp"

#include "kvc/json_util.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace kvc {
namespace {

speed_t baud_constant(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
#ifdef B230400
        case 230400: return B230400;
#endif
#ifdef B460800
        case 460800: return B460800;
#endif
#ifdef B921600
        case 921600: return B921600;
#endif
        default: return B115200;
    }
}

}  // namespace

double epoch_seconds() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

SerialOut::SerialOut(std::string port, int baud, double hz, bool verbose)
    : port_(std::move(port)),
      baud_(baud),
      min_dt_(hz > 0.0 ? 1.0 / hz : 0.0),
      dry_run_(port_.empty()),
      verbose_(verbose) {
    thread_ = std::thread(&SerialOut::worker, this);
}

SerialOut::~SerialOut() { close(); }

void SerialOut::send(uint64_t frame_no, std::vector<ojson> dets,
                     std::optional<double> infer_ms, const ojson& extra) {
    const double now = epoch_seconds();
    if (now - last_t_ < min_dt_) return;  // rate limit; seq does not advance
    last_t_ = now;
    ++seq_;

    ojson rec;
    rec["t"] = python_round(now, 3);
    rec["frame"] = frame_no;
    rec["seq"] = seq_;
    rec["n"] = dets.size();
    ojson arr = ojson::array();
    for (auto& d : dets) arr.push_back(std::move(d));
    rec["obj"] = std::move(arr);
    if (infer_ms.has_value()) rec["infer_ms"] = python_round(*infer_ms, 1);
    if (extra.is_object()) {
        for (auto it = extra.begin(); it != extra.end(); ++it) {
            rec[it.key()] = it.value();
        }
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return;
        if (q_.size() >= 4) {
            q_.pop_front();
            ++dropped_;
        }
        q_.push_back(std::move(rec));
    }
    cv_.notify_one();
}

void SerialOut::consume(const std::string& line) {
    if (dry_run_) {
        std::lock_guard<std::mutex> lk(print_mu_);
        std::cout << line;
        std::cout.flush();
        ++sent_;
        return;
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (fd_ < 0) {
            if (!open_port()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
        }
        if (write_all(line.data(), line.size())) {
            ++sent_;
            return;
        }
        if (verbose_) {
            std::lock_guard<std::mutex> lk(print_mu_);
            std::cout << "[串口] ⚠️ 写失败，重连…" << std::endl;
        }
        close_port_locked();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    // Both attempts failed: drop silently (matching serial_out.py).
}

void SerialOut::worker() {
    for (;;) {
        ojson rec;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(200), [&] {
                return stop_ || !q_.empty();
            });
            if (q_.empty()) {
                if (stop_) return;
                continue;
            }
            rec = std::move(q_.front());
            q_.pop_front();
        }

        // Encoding is inside try/catch: one bad record must never kill the
        // writer thread (the historical numpy-array TypeError).
        std::string line;
        try {
            line = compact_dump(rec);
            line.push_back('\n');
        } catch (const std::exception& e) {
            const uint64_t n = ++encode_errs_;
            if (n <= 3 || n % 100 == 0) {
                std::lock_guard<std::mutex> lk(print_mu_);
                std::cerr << "[串口] ❌ 这一帧编码失败（第 " << n
                          << " 次）：" << e.what() << "\n"
                          << "       这帧丢掉了，但串口还活着。" << std::endl;
            }
            continue;
        }
        consume(line);
    }
}

bool SerialOut::open_port() {
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;

    struct termios tty {};
    if (::tcgetattr(fd_, &tty) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    ::cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(CSTOPB | PARENB | CSIZE);
    tty.c_cflag |= CS8;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    ::cfsetispeed(&tty, baud_constant(baud_));
    ::cfsetospeed(&tty, baud_constant(baud_));
    if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    ::tcflush(fd_, TCOFLUSH);
    if (verbose_) {
        std::lock_guard<std::mutex> lk(print_mu_);
        std::cout << "[串口] ✅ 已打开 " << port_ << " @ " << baud_ << std::endl;
    }
    return true;
}

void SerialOut::close_port_locked() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SerialOut::write_all(const char* data, size_t len) {
    size_t off = 0;
    while (off < len) {
        struct pollfd pfd {};
        pfd.fd = fd_;
        pfd.events = POLLOUT;
        const int pr = ::poll(&pfd, 1, 200);  // pyserial write_timeout=0.2
        if (pr < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (pr == 0) return false;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;

        const ssize_t n = ::write(fd_, data + off, len - off);
        if (n > 0) {
            off += static_cast<size_t>(n);
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

void SerialOut::close() {
    if (closed_) return;
    closed_ = true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    close_port_locked();

    if (verbose_) {
        std::lock_guard<std::mutex> lk(print_mu_);
        std::cout << "[串口] 共发送 " << sent_.load() << " 条，丢弃 "
                  << dropped_.load() << " 条";
        if (encode_errs_.load()) {
            std::cout << "，★ 编码失败 " << encode_errs_.load() << " 条";
        }
        std::cout << std::endl;
    }
}

}  // namespace kvc
