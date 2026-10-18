#pragma once

#include "kvc/body_calib.hpp"  // ojson

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace kvc {

// JSON Lines writer.  The producer never blocks: a small queue (newest wins)
// plus a background thread absorb a stuck UART, exactly like serial_out.py.
//
// port == "" means dry-run: every encoded line goes to stdout instead.
class SerialOut {
public:
    SerialOut(std::string port, int baud = 115200, double hz = 20.0,
              bool verbose = true);
    ~SerialOut();

    SerialOut(const SerialOut&) = delete;
    SerialOut& operator=(const SerialOut&) = delete;

    // dets: already-built serial objects.  extra: coord/rev/cam/... header.
    void send(uint64_t frame_no, std::vector<ojson> dets,
              std::optional<double> infer_ms, const ojson& extra);

    void close();

    uint64_t sent() const { return sent_.load(); }
    uint64_t dropped() const { return dropped_.load(); }
    uint64_t encode_errs() const { return encode_errs_.load(); }

private:
    void worker();
    bool open_port();
    void close_port_locked();
    bool write_all(const char* data, size_t len);
    void consume(const std::string& line);

    std::string port_;
    int baud_ = 115200;
    double min_dt_ = 0.0;
    bool dry_run_ = true;
    bool verbose_ = true;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<ojson> q_;
    bool stop_ = false;

    double last_t_ = 0.0;      // producer only
    uint64_t seq_ = 0;         // producer only

    std::atomic<uint64_t> sent_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> encode_errs_{0};

    int fd_ = -1;
    std::mutex print_mu_;
    std::thread thread_;
    bool closed_ = false;
};

double epoch_seconds();

}  // namespace kvc
