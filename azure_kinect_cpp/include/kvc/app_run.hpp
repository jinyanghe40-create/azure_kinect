#pragma once

#include "kvc/cli.hpp"

namespace kvc {

// Full production loop, equivalent to azure_kinect/run_azure.py main().
// Returns the process exit code (0 on success / user stop).
int run_app(const RunOptions& opt);

}  // namespace kvc
