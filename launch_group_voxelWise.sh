#!/bin/bash
# Launch script for group-level voxel-wise analysis
# This script submits all SLURM scripts in the scripts directory
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# Scripts directory
SCRIPTS_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel_timeEffect/whole_brain"

echo "=== Group-level Voxel-wise Analysis Launcher ==="
echo "Scripts directory: $SCRIPTS_DIR"
echo ""

# Change to scripts directory and submit all SLURM job scripts
cd "$SCRIPTS_DIR"
sbatch group_*.sh

echo ""
echo "✅ All jobs submitted!"
echo "Check status with: squeue -u \$USER"


