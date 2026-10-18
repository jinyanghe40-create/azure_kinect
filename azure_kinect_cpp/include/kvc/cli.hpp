#pragma once

#include <string>

namespace kvc {

// Command-line options of azure_kinect/run_azure.py, one-for-one.  ROS flags
// are recognised but rejected (not ported); --profile is the C++ addition
// that replaces the diag_framebudget monkey-patch scripts.
struct RunOptions {
    std::string engine;
    double score = 0.4;
    double edge = 0.25;
    bool edge_auto = true;
    double min_dist = 0.3;
    double max_dist = 1.5;
    double plane_thresh = 0.003;
    double table_thresh = 0.010;
    double voxel = 0.02;
    int min_cluster_points = 0;
    std::string face_select = "most_facing";
    double depth_noise_coef = 0.008;
    int max_faces = 3;

    int stride = 2;
    std::string depth_mode = "nfov";
    bool color = true;
    int rotate = 0;
    std::string color_res = "720p";

    std::string calib;  // empty -> default resolution (env / exe dir / sibling)
    bool allow_uncalibrated = false;
    bool gravity = true;

    bool only_classified = false;
    bool show_text = false;
    bool json_out = false;
    std::string serial;
    bool serial_dry = false;
    int baud = 115200;
    double hz = 10.0;
    long long frames = 0;
    std::string save_dir;
    int save_every = 1;
    bool split_boxes = true;
    bool mm = false;
    bool dedup = true;
    bool geo_verbose = false;
    bool profile = false;
};

// 0 = parsed, caller should run.  1 = help/version printed, exit 0.
// 2 = usage error (message already printed), exit 2.
int parse_run_args(int argc, char** argv, RunOptions* out);

// <exe>/models, <exe>/../models, <exe>/../../models, $K4A_ENGINE.
std::string default_engine_path();

}  // namespace kvc
