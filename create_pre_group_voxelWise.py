#!/usr/bin/env python3
"""
Create SLURM scripts for pre-group voxel-wise analysis.

This script generates individual SLURM scripts for each subject and phase,
allowing parallel processing of the pre-group level analysis.

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

USAGE:
    # Create scripts for all phases and copes (uses preset defaults)
    python3 create_pre_group_voxelWise.py
    
    # Create scripts for specific phases
    python3 create_pre_group_voxelWise.py --phases phase2,phase3
    
    # Create scripts for specific data source
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Create scripts with limited factors (2x2 factorial design)
    python3 create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id"
    
    # Create 2x2 drug × gender design for placebo data
    python3 create_pre_group_voxelWise.py --include-columns "subID,drug_id,gender_id" --data-source placebo
    
    # Combine multiple options
    python3 create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id" --phases phase2 --data-source standard
    
    # Dry run to see what would be created
    python3 create_pre_group_voxelWise.py --dry-run
    
    # Show help
    python3 create_pre_group_voxelWise.py --help

EXAMPLES:
    # Quick start with preset defaults
    python3 create_pre_group_voxelWise.py
    
    # Process only Phase 2 data
    python3 create_pre_group_voxelWise.py --phases phase2
    
    # Process placebo data only
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Process guess data only
    python3 create_pre_group_voxelWise.py --data-source guess
    
    # Create 2x2 factorial design (Group × Drug)
    python3 create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id"
    
    # Create 2x2 factorial design for specific phase
    python3 create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id" --phases phase2
    
    # Create 2x2 factorial design with placebo data source
    python3 create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id" --data-source placebo
    
    # Create 2x2 drug × gender design for placebo data
    python3 create_pre_group_voxelWise.py --include-columns "subID,drug_id,gender_id" --data-source placebo
    
    # Create 2x2 drug × gender design for specific phase (placebo)
    python3 create_pre_group_voxelWise.py --include-columns "subID,drug_id,gender_id" --phases phase2 --data-source placebo
    
    # Test with dry run first
    python3 create_pre_group_voxelWise.py --dry-run

SLURM PARAMETERS:
    --partition: SLURM partition (default: ckpt-all)
    --account: SLURM account (default: fang)
    --time: Time limit (default: 04:00:00)
    --mem: Memory limit (default: 32G)
    --cpus-per-task: CPUs per task (default: 4)
    --container: Container image (default: narsad-fmri_1st_level_1.0.sif)

OUTPUT:
    Creates SLURM scripts in script_dir:
    - pre_group_sub-XXX_phaseY.sh (individual job scripts)
    - launch_all_pre_group.sh (launch all jobs)
    - monitor_jobs.sh (monitor job progress)
    - logs/ directory for job outputs
"""

import os
import argparse
import glob
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default SLURM parameters
DEFAULT_SLURM_PARAMS = {
    'partition': 'ckpt-all',
    #'partition': 'cpu-g2',
    'account': 'fang',
    'time': '04:00:00',
    'mem': '32G',
    'cpus_per_task': 4,
    'container': '/gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_time_effect/narsad-fmri_1st_level_1.0.sif'
}

def get_cope_list(derivatives_dir):
    """Get list of copes and phases from derivatives directory."""
    copes = []
    # The derivatives_dir should point to the fMRI_analysis directory
    # so we just need to append 'firstLevel'
    first_level_dir = os.path.join(derivatives_dir, 'firstLevel_timeEffect')
    
    logger.info(f"Looking for first level directory at: {first_level_dir}")
    
    if not os.path.exists(first_level_dir):
        logger.warning(f"First level directory not found: {first_level_dir}")
        return copes
    
    # Look for subject directories (e.g., sub-N101, sub-N102, etc.)
    for item in os.listdir(first_level_dir):
        if item.startswith('sub-') and os.path.isdir(os.path.join(first_level_dir, item)):
            subject_dir = os.path.join(first_level_dir, item)
            
            # Check for session directories (e.g., ses-pilot3mm, ses-001, etc.)
            for session in os.listdir(subject_dir):
                if session.startswith('ses-') and os.path.isdir(os.path.join(subject_dir, session)):
                    session_dir = os.path.join(subject_dir, session)
                    func_dir = os.path.join(session_dir, 'func')
                    
                    if os.path.exists(func_dir):
                        # Check what phases and copes this subject has by looking at the func files
                        phase_cope_files = {}
                        logger.info(f"Scanning func directory: {func_dir}")
                        for file in os.listdir(func_dir):
                            if file.endswith('_bold.nii') and 'task-phase' in file:
                                # Extract phase and cope from filename
                                # Handle complex filenames like: sub-N101_ses-pilot3mm_task-phase3_space-MNI152NLin2009cAsym_desc-varcope9_bold.nii
                                logger.info(f"Found bold file: {file}")
                                
                                # Extract phase
                                if 'task-phase2' in file:
                                    phase = 'phase2'
                                elif 'task-phase3' in file:
                                    phase = 'phase3'
                                else:
                                    continue
                                
                                # Extract cope number from desc-varcopeX or desc-copeX
                                cope_num = None
                                if 'desc-varcope' in file:
                                    cope_num = int(file.split('desc-varcope')[1].split('_')[0])
                                elif 'desc-cope' in file:
                                    cope_num = int(file.split('desc-cope')[1].split('_')[0])
                                
                                if cope_num is not None:
                                    if phase not in phase_cope_files:
                                        phase_cope_files[phase] = set()
                                    phase_cope_files[phase].add(cope_num)
                                    logger.info(f"  -> Phase {phase}, Cope {cope_num} detected")
                        
                        # Add phase-cope combinations for this subject
                        for phase, cope_numbers in phase_cope_files.items():
                            for cope_num in cope_numbers:
                                copes.append((phase, cope_num))
                        # Continue to check other sessions (don't break)
    
    # Remove duplicates and sort
    unique_copes = list(set(copes))
    unique_copes.sort(key=lambda x: (x[0], x[1]))  # Sort by phase, then cope number
    
    logger.info(f"Found copes: {[f'{c[0]}-cope{c[1]}' for c in unique_copes]}")
    return unique_copes

def create_slurm_script(phase, cope_num, output_dir, script_dir, slurm_params, data_source, include_columns):
    """Create a SLURM script for a specific phase and cope."""
    
    script_name = f"pre_group_{phase}_cope{cope_num}.sh"
    script_path = os.path.join(script_dir, script_name)
    
    # Container bind mounts
    container_binds = [
        "-B /gscratch/fang:/data",
        "-B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir",
        "-B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_time_effect:/app"
    ]
    
    # Convert container path to host path for mkdir command
    # Replace /data with /gscratch/fang for host paths
    host_output_dir = output_dir.replace('/data', '/gscratch/fang')

    # Build the command string
    cmd_base = f"""python3 /app/run_pre_group_voxelWise.py \\
    --output-dir {output_dir} \\
    --phase {phase} \\
    --cope {cope_num} \\
    --data-source {data_source}"""
    
    if include_columns:
        cmd_base += f" \\\n    --include-columns {include_columns}"
    
    # Script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pre_group_{phase}_cope{cope_num}
#SBATCH --partition={slurm_params['partition']}
#SBATCH --account={slurm_params['account']}
#SBATCH --time={slurm_params['time']}
#SBATCH --mem={slurm_params['mem']}
#SBATCH --cpus-per-task={slurm_params['cpus_per_task']}
#SBATCH --output=logs/pre_group_{phase}_cope{cope_num}_%j.out
#SBATCH --error=logs/pre_group_{phase}_cope{cope_num}_%j.err

# Pre-group voxel-wise analysis for {phase} - cope{cope_num}
# Generated by create_pre_group_voxelWise.py

set -e

# Load modules if needed
module load apptainer

# Set environment variables
export SCRUBBED_DIR=/scrubbed_dir
export DATA_DIR=/data

# Create output directory on host (before container launch)
mkdir -p {host_output_dir}

# Run the pre-group analysis for this phase and cope
apptainer exec {' '.join(container_binds)} {slurm_params['container']} \\
    {cmd_base}

echo "Completed pre-group analysis for {phase} - cope{cope_num}"
"""
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path



def main():
    """Main function to create SLURM scripts."""
    parser = argparse.ArgumentParser(
        description='Create SLURM scripts for pre-group voxel-wise analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create scripts for all phases and copes
  python create_pre_group_voxelWise.py
  
  # Create scripts for specific phases only
  python create_pre_group_voxelWise.py --phases phase2,phase3
  
  # Create scripts for specific data source
  python create_pre_group_voxelWise.py --data-source placebo
  
  # Create 2x2 factorial design (Group × Drug)
  python create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id"
  
  # Create 2x2 drug × gender design for placebo data
  python create_pre_group_voxelWise.py --include-columns "subID,drug_id,gender_id" --data-source placebo
  
  # Combine multiple options for 2x2 design
  python create_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id" --phases phase2 --data-source standard
        """
    )
    
    parser.add_argument(
        '--data-source',
        choices=['standard', 'placebo', 'guess'],
        default='standard',
        help='Data source to process (default: standard)'
    )
    
    parser.add_argument(
        '--phases',
        help='Comma-separated list of phases to process (e.g., phase2,phase3)'
    )

    parser.add_argument(
        '--include-columns',
        help='Comma-separated list of columns to include (e.g., "subID,group_id,drug_id")'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without writing files'
    )
    
    args = parser.parse_args()
    
    # Use container paths directly since this script runs inside the container
    logger.info("Using container paths directly")
    output_dir = '/data/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel_timeEffect'
    derivatives_dir = '/data/NARSAD/MRI/derivatives/fMRI_analysis'
    
    # Set script directory - use default workdir/pregroup structure
    scrubbed_dir = os.getenv('SCRUBBED_DIR', '/scrubbed_dir')
    workdir = Path(scrubbed_dir) / 'NARSAD' / 'work_flows' / 'groupLevel_timeEffect'
    script_dir = workdir / 'pregroup'
    
    # Ensure script directory is absolute and in a writable location
    if not script_dir.is_absolute():
        script_dir = script_dir.resolve()
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")
    
    if not args.dry_run:
        try:
            script_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created script directory: {script_dir}")
        except OSError as e:
            if "Read-only file system" in str(e) or "Permission denied" in str(e):
                # Try multiple fallback locations
                fallback_locations = [
                    Path("/tmp") / "narsad_slurm_scripts" / "pregroup",
                    Path("/tmp") / "nipype_slurm_scripts" / "pregroup",
                    Path("/scrubbed_dir") / "temp_slurm_scripts" / "pregroup"
                ]
                
                for fallback_dir in fallback_locations:
                    try:
                        fallback_dir.mkdir(parents=True, exist_ok=True)
                        script_dir = fallback_dir
                        logger.warning(f"Target directory read-only, using fallback: {script_dir}")
                        break
                    except OSError:
                        continue
                else:
                    # If all fallbacks fail, use current directory with a unique name
                    import uuid
                    script_dir = Path.cwd() / f"slurm_scripts_{uuid.uuid4().hex[:8]}" / "pregroup"
                    script_dir.mkdir(parents=True, exist_ok=True)
                    logger.warning(f"All fallbacks failed, using current directory: {script_dir}")
            else:
                raise
    
    # Get SLURM parameters - use default values
    slurm_params = {
        'partition': DEFAULT_SLURM_PARAMS['partition'],
        'account': DEFAULT_SLURM_PARAMS['account'],
        'time': DEFAULT_SLURM_PARAMS['time'],
        'mem': DEFAULT_SLURM_PARAMS['mem'],
        'cpus_per_task': DEFAULT_SLURM_PARAMS['cpus_per_task'],
        'container': DEFAULT_SLURM_PARAMS['container']
    }
    
    # Get phases and copes to process
    if args.phases:
        phases_to_process = [p.strip() for p in args.phases.split(',')]
        # Filter by specific phases if specified
        logger.info(f"Filtering by phases: {phases_to_process}")
    else:
        phases_to_process = None  # Process all phases
    
    # Get all copes from derivatives directory
    logger.info(f"Scanning derivatives directory: {derivatives_dir}")
    logger.info(f"Derivatives directory type: {type(derivatives_dir)}")
    logger.info(f"Derivatives directory absolute: {os.path.abspath(derivatives_dir)}")
    phase_cope_pairs = get_cope_list(derivatives_dir)
    
    # Filter by specific phases if specified
    if phases_to_process:
        phase_cope_pairs = [(p, c) for p, c in phase_cope_pairs if p in phases_to_process]
    
    logger.info(f"Found {len(phase_cope_pairs)} phase-cope combinations to process")
    
    if args.dry_run:
        logger.info("DRY RUN - Would create the following scripts:")
        for phase, cope_num in phase_cope_pairs:
            logger.info(f"  pre_group_{phase}_cope{cope_num}.sh")
        return
    
    # Create individual SLURM scripts
    created_scripts = []
    for phase, cope_num in phase_cope_pairs:
        script_path = create_slurm_script(phase, cope_num, output_dir, script_dir, slurm_params, args.data_source, args.include_columns)
        created_scripts.append(script_path)
        logger.info(f"Created: {script_path}")
    
    logger.info(f"\n✅ Successfully created {len(created_scripts)} SLURM scripts!")
    logger.info(f"📁 Scripts saved to: {script_dir}")
    logger.info(f"🔍 Individual scripts can be run separately or use launch_pre_group_voxelWise.sh to submit all")

if __name__ == "__main__":
    main()
