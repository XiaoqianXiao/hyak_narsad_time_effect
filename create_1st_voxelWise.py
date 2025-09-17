#!/usr/bin/env python3
"""
First-level fMRI analysis pipeline for NARSAD project.

This script orchestrates the first-level fMRI analysis for individual subjects,
generating and running Nipype workflows. It supports both standard first-level
analysis and LSS (Least Squares Separate) analysis.

Usage:
    python create_1st_voxelWise.py --subject SUBJECT_ID --task TASK_NAME
    python create_1st_voxelWise.py  # Generate SLURM scripts for all subjects

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import json
import logging
from pathlib import Path
from bids.layout import BIDSLayout
from templateflow.api import get as tpl_get, templates as get_tpl_list
import pandas as pd
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import subprocess

# Import functions from first_level_workflows
from first_level_workflows import extract_cs_conditions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT AND PATH SETUP
# =============================================================================

# Set FSL environment variables for the container
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings for local execution
PLUGIN_SETTINGS = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Use environment variables for data paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
BIDS_DIR = DATA_DIR
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
FMRIPREP_FOLDER = os.path.join(DERIVATIVES_DIR, 'fmriprep')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data/behav')
SCRUBBED_DIR = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_time_effect/narsad-fmri_1st_level_1.0.sif"

# Workflow and output directories
PARTICIPANT_LABEL = []  # Can be set via args or env if needed
RUN = []
SPACE = ['MNI152NLin2009cAsym']

# Output directory
OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# BIDS LAYOUT INITIALIZATION
# =============================================================================

def initialize_bids_layout():
    """Initialize BIDS layout and validate data availability."""
    try:
        layout = BIDSLayout(str(BIDS_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        
        # Get available entities
        subjects = layout.get(target='subject', return_type='id')
        sessions = layout.get(target='session', return_type='id')
        runs = layout.get(target='run', return_type='id')
        
        logger.info(f"BIDS layout initialized: {len(subjects)} subjects, {len(sessions)} sessions")
        return layout, subjects, sessions, runs
        
    except Exception as e:
        logger.error(f"Failed to initialize BIDS layout: {e}")
        raise

def build_query(participant_label=None, run=None, task=None):
    """
    Build query for preprocessed BOLD files.
    
    Args:
        participant_label (list): List of participant labels to filter
        run (list): List of run numbers to filter
        task (str): Task name to filter
    
    Returns:
        dict: Query dictionary for BIDS layout
    """
    query = {
        'desc': 'preproc',
        'suffix': 'bold',
        'extension': ['.nii', '.nii.gz']
    }
    
    if participant_label:
        query['subject'] = '|'.join(participant_label)
    if run:
        query['run'] = '|'.join(run)
    if task:
        query['task'] = task
    if SPACE:
        query['space'] = '|'.join(SPACE)
    
    return query

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_condition_names_from_events(events_file):
    """
    Get condition names and create interesting contrasts from events file.
    
    This function loads an events file, processes it to group CS-, CSS, and CSR conditions,
    and creates interesting contrasts for the NARSAD analysis.
    
    Args:
        events_file (str): Path to the events CSV file
    
    Returns:
        tuple: (contrasts, cs_conditions, css_conditions, csr_conditions, other_conditions, condition_names)
    """

    from utils import read_csv_with_detection
    events_df = read_csv_with_detection(events_file)
    
    df_trial_info = events_df.copy()

    if df_trial_info is None:
        raise ValueError("df_trial_info is required")
    
    # Extract CS-, CSS, and CSR conditions with grouping
    df_with_conditions = extract_cs_conditions(df_trial_info)
    
    # Use the conditions column for contrast generation
    all_contrast_conditions = df_with_conditions['conditions'].unique().tolist()
    condition_names = all_contrast_conditions.copy()
    
    # Check which conditions actually have trials
    conditions_with_trials = {}
    for condition in all_contrast_conditions:
        trial_count = len(df_with_conditions[df_with_conditions['conditions'] == condition])
        conditions_with_trials[condition] = trial_count
        logger.info(f"Condition '{condition}': {trial_count} trials")
    
    # Define the interesting contrasts
    interesting_contrasts = [
        ("CS-_first_half > FIXATION_first_half", "first half Other CS- trials vs baseline"),
        ("CSS_first_half > FIXATION_first_half", "first half Other CSS trials vs baseline"),
        ("CSR_first_half > FIXATION_first_half", "first half Other CSR trials vs baseline"),
        ("CSS_first_half > CSR_first_half", "first half Other CSS trials vs Other CSR trials"),
        ("CSR_first_half > CSS_first_half", "first half Other CSR trials vs Other CSS trials"),
        ("CSS_first_half > CS-_first_half", "first half Other CSS trials vs Other CS- trials"),
        ("CSR_first_half > CS-_first_half", "first half Other CSR trials vs Other CS- trials"),
        ("CS-_first_half > CSS_first_half", "first half Other CS- trials vs Other CSS trials"),
        ("CS-_first_half > CSR_first_half", "first half Other CS- trials vs Other CSR trials"),
        ("CS-_second_half > FIXATION_second_half", "second half Other CS- trials vs baseline"),
        ("CSS_second_half > FIXATION_second_half ", "second half Other CSS trials vs baseline"),
        ("CSR_second_half > FIXATION_second_half", "second half Other CSR trials vs baseline"),
        ("CSS_second_half > CSR_second_half", "second half Other CSS trials vs Other CSR trials"),
        ("CSR_second_half > CSS_second_half", "second half Other CSR trials vs Other CSS trials"),
        ("CSS_second_half > CS-_second_half", "second half Other CSS trials vs Other CS- trials"),
        ("CSR_second_half > CS-_second_half", "second half Other CSR trials vs Other CS- trials"),
        ("CS-_second_half > CSS_second_half", "second half Other CS- trials vs Other CSS trials"),
        ("CS-_second_half > CSR_second_half", "second half Other CS- trials vs Other CSR trials"),
    ]
    
    contrasts = []
    
    for contrast_name, description in interesting_contrasts:
        # Parse the contrast name (e.g., "CS-_others > FIXATION")
        if ' > ' in contrast_name:
            condition1, condition2 = contrast_name.split(' > ')
            condition1 = condition1.strip()
            condition2 = condition2.strip()
            
            # Check if both conditions exist AND have trials
            if (condition1 in all_contrast_conditions and condition2 in all_contrast_conditions and
                conditions_with_trials.get(condition1, 0) > 0 and conditions_with_trials.get(condition2, 0) > 0):
                contrast = (contrast_name, 'T', [condition1, condition2], [1, -1])
                contrasts.append(contrast)
                logger.info(f"Added contrast: {contrast_name} - {description}")
            else:
                missing_conditions = []
                if condition1 not in all_contrast_conditions or conditions_with_trials.get(condition1, 0) == 0:
                    missing_conditions.append(condition1)
                if condition2 not in all_contrast_conditions or conditions_with_trials.get(condition2, 0) == 0:
                    missing_conditions.append(condition2)
                logger.warning(f"Contrast {contrast_name}: conditions {missing_conditions} missing or have no trials")
        else:
            logger.warning(f"Invalid contrast format: {contrast_name}")
    
    logger.info(f"Created {len(contrasts)} interesting contrasts")
    
    return contrasts, condition_names, df_with_conditions


def create_workflow_config():
    """
    Create consistent workflow configuration for all tasks.
    
    Returns:
        dict: Workflow configuration
    """
    config = {
        'use_smoothing': True,
        'fwhm': 6.0,
        'brightness_threshold': 1000,
        'high_pass_cutoff': 100,
        'use_derivatives': True,
        'model_serial_correlations': True
    }
    
    logger.info(f"Created workflow configuration: {config}")
    return config

def get_events_file_path(sub, task):
    """
    Get the appropriate events file path for a subject and task.
    
    Args:
        sub (str): Subject ID
        task (str): Task name
    
    Returns:
        str: Path to events file
    """
    # Handle special case for N202 phase3
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(BEHAV_DIR, 'task-NARSAD_phase-3_sub-202_half_events.csv')
    else:
        events_file = os.path.join(BEHAV_DIR, f'task-Narsad_{task}_half_events.csv')
    
    logger.info(f"Using events file: {events_file}")
    return events_file

def create_subject_inputs(sub, part, layout, query):
    """
    Create input dictionary for a subject.
    
    Args:
        sub (str): Subject ID
        part: BIDS entity object
        layout: BIDS layout object
        query (dict): Query dictionary
    
    Returns:
        dict: Input files dictionary for the subject
    """
    inputs = {sub: {}}
    base = {'subject', 'task'}.intersection(part.entities)
    subquery = {k: v for k, v in part.entities.items() if k in base}
    
    # Set basic inputs
    inputs[sub]['bold'] = part.path
    inputs[sub]['tr'] = part.entities['RepetitionTime']
    
    try:
        # Get mask file
        mask_files = layout.get(suffix='mask', return_type='file',
                               extension=['.nii', '.nii.gz'],
                               space=query['space'], **subquery)
        if not mask_files:
            raise IndexError("No mask files found")
        inputs[sub]['mask'] = mask_files[0]
        
        # Get regressors file
        regressor_files = layout.get(desc='confounds', return_type='file',
                                   extension=['.tsv'], **subquery)
        if not regressor_files:
            raise IndexError("No regressor files found")
        inputs[sub]['regressors'] = regressor_files[0]
        
    except IndexError as e:
        logger.error(f"Missing required file for subject {sub}: {e}")
        raise
    
    # Set events file
    task = part.entities['task']
    inputs[sub]['events'] = get_events_file_path(sub, task)
    
    logger.info(f"Created inputs for subject {sub}: {list(inputs[sub].keys())}")
    return inputs

# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def create_slurm_script(sub, inputs, work_dir, output_dir, task, container_path):
    """
    Generate SLURM script for a subject.
    
    Args:
        sub (str): Subject ID
        inputs (dict): Input files dictionary
        work_dir (str): Working directory
        output_dir (str): Output directory
        task (str): Task name
        container_path (str): Path to container image
    
    Returns:
        str: Path to generated SLURM script
    """
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=first_level_sub_{sub}
#SBATCH --account=fang
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel_timeEffect/{task}_sub_{sub}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel_timeEffect/{task}_sub_{sub}_%j.err

module load apptainer
apptainer exec \\
    -B /gscratch/fang:/data \\
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \\
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_time_effect:/app \\
    {container_path} \\
    python3 /app/create_1st_voxelWise.py --subject {sub} --task {task}
"""
    
    script_path = os.path.join(work_dir, f'sub_{sub}_slurm.sh')
    try:
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        logger.info(f"SLURM script created: {script_path}")
        return script_path
    except Exception as e:
        logger.error(f"Failed to create SLURM script: {e}")
        raise

# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

def run_subject_workflow(sub, inputs, work_dir, output_dir, task):
    """
    Run first-level workflow for a single subject.
    
    Args:
        sub (str): Subject ID
        inputs (dict): Input files dictionary
        work_dir (str): Working directory
        output_dir (str): Output directory
        task (str): Task name
    """
    try:
        # Import workflows
        from first_level_workflows import first_level_wf
        
        # Get workflow configuration
        config = create_workflow_config()
        
        # Get condition names and contrasts from events file
        events_file = inputs[sub]['events']
        contrasts, condition_names, df_with_conditions = get_condition_names_from_events(events_file)
        
        logger.info(f"Processing subject {sub}, task {task}")
        logger.info(f"Workflow config: {config}")
        
        # Create the workflow with processed DataFrame
        workflow = first_level_wf(
            in_files=inputs,
            output_dir=output_dir,
            condition_names=condition_names,
            contrasts=contrasts,
            fwhm=config['fwhm'],
            brightness_threshold=config['brightness_threshold'],
            high_pass_cutoff=config['high_pass_cutoff'],
            use_smoothing=config['use_smoothing'],
            use_derivatives=config['use_derivatives'],
            model_serial_correlations=config['model_serial_correlations'],
            df_conditions=df_with_conditions
        )
        
        # Set workflow base directory
        workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
        
        # Create output directory for this subject
        subject_output_dir = os.path.join(output_dir, 'firstLevel_timeEffect', task, f'sub-{sub}')
        Path(subject_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running workflow for subject {sub}, task {task}")
        logger.info(f"Workflow base directory: {workflow.base_dir}")
        logger.info(f"Output directory: {subject_output_dir}")
        
        # Run the workflow
        workflow.run(**PLUGIN_SETTINGS)
        
        logger.info(f"Workflow completed successfully for subject {sub}, task {task}")
        
    except ImportError as e:
        logger.error(f"Could not import workflows from first_level_workflows.py: {e}")
        logger.error("Make sure first_level_workflows.py is in the Python path")
        raise
    except Exception as e:
        logger.error(f"Error running workflow for subject {sub}, task {task}: {e}")
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def process_single_subject(args, layout, query):
    """
    Process a single subject with specified task.
    
    Args:
        args: Command line arguments
        layout: BIDS layout object
        query (dict): Query dictionary
    """
    found = False
    for part in layout.get(invalid_filters='allow', **query):
        if (part.entities['subject'] == args.subject and 
            (not args.task or part.entities['task'] == args.task)):
            
            found = True
            entities = part.entities
            sub = entities['subject']
            task = entities['task']
            
            # Create working directory
            work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/firstLevel_timeEffect/{task}')
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                # Create subject inputs
                inputs = create_subject_inputs(sub, part, layout, query)
                
                logger.info(f"Running first-level analysis for subject {sub}, task {task}")
                run_subject_workflow(sub, inputs, work_dir, OUTPUT_DIR, task)
                
            except Exception as e:
                logger.error(f"Failed to process subject {sub}: {e}")
                raise
            
            break
    
    if not found:
        error_msg = f"Subject {args.subject} with task {args.task} not found in preprocessed BOLD files"
        logger.error(error_msg)
        raise ValueError(error_msg)

def generate_slurm_scripts(layout, query):
    """
    Generate SLURM scripts for all subjects.
    
    Args:
        layout: BIDS layout object
        query (dict): Query dictionary
    """
    logger.info("Generating SLURM scripts for all subjects")
    
    for part in layout.get(invalid_filters='allow', **query):
        entities = part.entities
        sub = entities['subject']
        task = entities['task']
        
        # Create working directory
        work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/firstLevel_timeEffect/{task}')
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Create subject inputs
            inputs = create_subject_inputs(sub, part, layout, query)
            
            # Generate SLURM script
            script_path = create_slurm_script(sub, inputs, work_dir, OUTPUT_DIR, task, CONTAINER_PATH)
            logger.info(f"SLURM script created for subject {sub}, task {task}")
            
        except Exception as e:
            logger.error(f"Failed to generate SLURM script for subject {sub}: {e}")
            continue

def main():
    """Main execution function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run first-level fMRI analysis.")
    parser.add_argument('--subject', type=str, help="Specific subject ID to process")
    parser.add_argument('--task', type=str, help="Specific task to process (e.g., phase2, phase3)")
    args = parser.parse_args()
    
    try:
        # Initialize BIDS layout
        layout, subjects, sessions, runs = initialize_bids_layout()
        
        # Build query
        query = build_query(PARTICIPANT_LABEL, RUN, args.task)
        
        # Validate query returns results
        prepped_bold = layout.get(**query)
        if not prepped_bold:
            logger.error(f'No preprocessed files found under: {DERIVATIVES_DIR}')
            return 1
        
        logger.info(f"Found {len(prepped_bold)} preprocessed BOLD files")
        
        if args.subject:
            # Process single subject
            process_single_subject(args, layout, query)
        else:
            # Generate SLURM scripts for all subjects
            generate_slurm_scripts(layout, query)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
