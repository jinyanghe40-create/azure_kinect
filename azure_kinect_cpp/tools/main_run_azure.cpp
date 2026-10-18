// run_azure_cpp — production entry point.  Same flags and same output as
// azure_kinect/run_azure.py; see --help.

#include "kvc/app_run.hpp"
#include "kvc/cli.hpp"

int main(int argc, char** argv) {
    kvc::RunOptions opt;
    const int parsed = kvc::parse_run_args(argc, argv, &opt);
    if (parsed == 1) return 0;   // --help
    if (parsed != 0) return 2;   // usage error (already printed)
    return kvc::run_app(opt);
}
