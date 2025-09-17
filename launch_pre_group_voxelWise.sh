#!/bin/bash
# Launch script for pre-group voxel-wise analysis
# This script submits all SLURM scripts in the scripts directory
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
#
# USAGE:
#   # Submit all pre-group analysis jobs
#   bash launch_pre_group_voxelWise.sh
#
#   # Show this help
#   bash launch_pre_group_voxelWise.sh --help

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Submit all pre-group analysis jobs"
    echo "  $0"
    echo ""
    echo "  # Show this help"
    echo "  $0 --help"
    exit 0
fi

# Scripts directory
SCRIPTS_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel_timeEffect/pregroup"


# Change to scripts directory and submit all .sh files
cd "$SCRIPTS_DIR"
for i in pre_group_*.sh; do
    sbatch ${i}
done
