#!/usr/bin/env python3
"""
Generic pre-group level fMRI analysis pipeline for NARSAD project.

This script prepares first-level fMRI data for group-level analysis by:
1. Collecting cope and varcope files from first-level analysis
2. Running data preparation workflows for each task and contrast
3. Organizing data for subsequent group-level statistical analysis

Supports flexible filtering and analysis configurations through command-line arguments.

Usage:
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo
    python run_pre_group_level.py --filter-column drug_condition --filter-value DrugA
    python run_pre_group_level.py --filter-column group --filter-value Patients
    python run_pre_group_level.py --filter-column guess --filter-value High
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo --include-columns group_id,drug_id
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo --output-dir /custom/path

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import shutil
import logging
import argparse
import glob
from pathlib import Path
from bids.layout import BIDSLayout
import pandas as pd
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from group_level_workflows import wf_data_prepare
from templateflow.api import get as tpl_get, templates as get_tpl_list

# Configure Nipype crash directory to a writable location
import nipype
import os

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set crash directory to a writable location
os.environ['NIPYPE_CRASH_DIR'] = '/tmp/nipype_crashes'
nipype.config.set('execution', 'crashfile_format', 'txt')
nipype.config.set('execution', 'crash_dir', '/tmp/nipype_crashes')
nipype.config.set('execution', 'remove_unnecessary_outputs', 'false')

def get_workflow_crash_dir(workflow_dir):
    """Get crash directory within the workflow directory."""
    crash_dir = os.path.join(workflow_dir, 'nipype_crashes')
    try:
        os.makedirs(crash_dir, exist_ok=True)
        # Test write access
        test_file = os.path.join(crash_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"Using workflow crash directory: {crash_dir}")
        return crash_dir
    except Exception as e:
        logger.warning(f"Failed to create crash directory in {workflow_dir}: {e}")
        # Fallback to workflow directory itself
        logger.info(f"Using workflow directory as crash directory: {workflow_dir}")
        return workflow_dir

# Set Nipype to use /tmp for crash directories to avoid read-only filesystem issues
nipype.config.set('execution', 'crash_dir', '/tmp/nipype_crashes')
nipype.config.set('execution', 'crashfile_format', 'txt')
nipype.config.set('execution', 'remove_unnecessary_outputs', 'false')

# Get the logger instance
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT AND PATH SETUP
# =============================================================================

# Set FSL environment variables
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Use environment variables for data paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
SCRUBBED_DIR = os.getenv('SCRUBBED_DIR', '/scrubbed_dir')
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_time_effect/narsad-fmri_1st_level_1.0.sif"

# Define standard reference image (MNI152 template from FSL)
GROUP_MASK = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))

# =============================================================================
# SUBJECT EXCLUSION LISTS
# =============================================================================

# Subjects without MRI data for each phase
SUBJECTS_NO_MRI = {
    'phase2': ['N102', 'N208'],
    'phase3': ['N102', 'N208', 'N120']
}

# =============================================================================
# BEHAVIORAL DATA CONFIGURATION
# =============================================================================

# Behavioral data paths
SCR_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'EDR')
DRUG_FILE = os.path.join(SCR_DIR, 'drug_order.csv')
ECR_FILE = os.path.join(SCR_DIR, 'ECR.csv')

# Analysis parameters
TASKS = ['phase2', 'phase3']

def get_contrast_range(task):
    """
    Get dynamic contrast range based on task.
    
    Args:
        task (str): Task name ('phase2' or 'phase3')
    
    Returns:
        list: Range of contrast numbers
    """
    if task == 'phase2':
        # Phase 2: 18 contrasts (all available contrasts)
        return list(range(1, 19))
    elif task == 'phase3':
        # Phase 3: 18 contrasts (all available contrasts)
        return list(range(1, 19))
    else:
        # Default fallback
        return list(range(1, 19))

# Default contrast range (will be overridden per task)
# This is a fallback - actual ranges are determined dynamically per task
CONTRAST_RANGE = list(range(1, 19))  # Contrasts 1-18 (fallback)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_behavioral_data(filter_column=None, filter_value=None, include_columns=None, data_source=None):
    """
    Load and prepare behavioral data for analysis with flexible filtering.
    
    Args:
        filter_column (str): Column name to filter on (e.g., 'Drug', 'group', 'guess')
        filter_value (str): Value to filter for (e.g., 'Placebo', 'Patients', 'High')
        include_columns (list): List of columns to include in group_info
        data_source (str): Data source type ('standard', 'placebo', 'guess')
    
    Returns:
        pandas.DataFrame: Filtered behavioral data with appropriate mappings
    """
    try:
        # Load drug order data with automatic separator detection
        from utils import read_csv_with_detection
        df_drug = read_csv_with_detection(DRUG_FILE)
        logger.info(f"Loaded drug data: {len(df_drug)} subjects, columns: {list(df_drug.columns)}")
        
        df_drug['group'] = df_drug['subID'].apply(
            lambda x: 'Patients' if x.startswith('N1') else 'Controls'
        )
        logger.info(f"Group mapping applied: {df_drug['group'].value_counts().to_dict()}")
        
        # Load ECR data with automatic separator detection
        df_ECR = read_csv_with_detection(ECR_FILE)
        logger.info(f"Loaded ECR data: {len(df_ECR)} subjects, columns: {list(df_ECR.columns)}")
        
        # Debug: Check subID overlap
        drug_subjects = set(df_drug['subID'])
        ecr_subjects = set(df_ECR['subID'])
        common_subjects = drug_subjects.intersection(ecr_subjects)
        logger.info(f"Subject overlap: {len(common_subjects)} common subjects out of {len(drug_subjects)} drug + {len(ecr_subjects)} ECR")
        
        # Merge behavioral data
        df_behav = df_drug.merge(df_ECR, how='left', left_on='subID', right_on='subID')
        logger.info(f"After merge: {len(df_behav)} subjects, columns: {list(df_behav.columns)}")
        
        # Apply data source filtering if specified
        if data_source and data_source != 'standard':
            logger.info(f"Applying data source filtering for '{data_source}'")
            logger.info(f"DataFrame before filtering: {len(df_behav)} subjects, columns: {list(df_behav.columns)}")
            
            drug_column = None
            # Prioritize the Drug column (string values) over drug_condition (numeric values)
            if 'Drug' in df_behav.columns:
                drug_column = 'Drug'
                logger.info(f"Using 'Drug' column for filtering (contains string values)")
            elif 'drug_condition' in df_behav.columns:
                drug_column = 'drug_condition'
                logger.info(f"Using 'drug_condition' column for filtering (contains numeric values)")
            else:
                logger.warning(f"Neither 'Drug' nor 'drug_condition' column found. Available columns: {list(df_behav.columns)}")
            
            if drug_column:
                logger.info(f"Drug column '{drug_column}' values: {df_behav[drug_column].value_counts().to_dict()}")
                
                if data_source == 'placebo':
                    # Filter for placebo subjects only
                    if drug_column == 'Drug':
                        # Drug column contains string values like 'Placebo', 'Oxytocin'
                        df_behav = df_behav[df_behav[drug_column] == 'Placebo']
                    elif drug_column == 'drug_condition':
                        # drug_condition column contains numeric values like 0, 1
                        # Assuming 0 = Placebo, 1 = Active (based on your data)
                        df_behav = df_behav[df_behav[drug_column] == 0]
                        logger.info(f"Filtering drug_condition == 0 (Placebo)")
                    
                    logger.info(f"Filtered by data source '{data_source}': {len(df_behav)} subjects remaining")
                    
                    # Check if we have enough subjects after filtering
                    if len(df_behav) == 0:
                        logger.warning(f"No subjects found after filtering for '{data_source}'.")
                        logger.warning(f"Available drug conditions: {df_behav[drug_column].unique()}")
                        logger.warning(f"Consider using --data-source standard instead")
                        
                elif data_source == 'guess':
                    # For guess analysis, we might want to filter by guess condition
                    # This could be customized based on your specific needs
                    logger.info(f"Data source '{data_source}' selected - no additional filtering applied")
            else:
                logger.warning(f"Drug column not found, cannot filter by data source '{data_source}'")
        
        # EXCLUDE Trans subjects (gender_code == 2) from all analyses to prevent matrix singularity
        # if 'gender_code' in df_behav.columns:
        #     before_count = len(df_behav)
        #     df_behav = df_behav[df_behav['gender_code'] != 2].copy()
        #     after_count = len(df_behav)
        #     if before_count != after_count:
        #         logger.info(f"EXCLUDED {before_count - after_count} Trans subjects (gender_code=2) from analysis to prevent matrix singularity")
        #         logger.info(f"Subjects remaining: {after_count}")
        # else:
        #     logger.warning("No gender_code column found - cannot exclude Trans subjects")
        
        # Apply additional filtering if specified
        if filter_column and filter_value:
            if filter_column not in df_behav.columns:
                raise ValueError(f"Filter column '{filter_column}' not found in behavioral data. "
                               f"Available columns: {list(df_behav.columns)}")
            
            # Apply filter
            df_behav = df_behav[df_behav[filter_column] == filter_value]
            logger.info(f"Filtered data by {filter_column}={filter_value}: {len(df_behav)} subjects remaining")
        
        # Create ID mappings for categorical variables
        group_levels = df_behav['group'].unique()
        group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
        df_behav['group_id'] = df_behav['group'].map(group_map)
        
        # Create drug condition mapping (handle both possible column names)
        drug_column = None
        if 'drug_condition' in df_behav.columns:
            drug_column = 'drug_condition'
        elif 'Drug' in df_behav.columns:
            drug_column = 'Drug'
        
        if drug_column:
            drug_levels = df_behav[drug_column].unique()
            drug_map = {level: idx + 1 for idx, level in enumerate(drug_levels)}
            df_behav['drug_id'] = df_behav[drug_column].map(drug_map)
            logger.info(f"Drug conditions: {drug_levels.tolist()}")
        
        # Create guess mapping if column exists
        if 'guess' in df_behav.columns:
            guess_levels = df_behav['guess'].unique()
            guess_map = {level: idx + 1 for idx, level in enumerate(guess_levels)}
            df_behav['guess_id'] = df_behav['guess'].map(guess_map)
            logger.info(f"Guess conditions: {guess_levels.tolist()}")
        
        # Validate include_columns with smart column mapping
        if include_columns:
            # Smart column name mapping for common variations
            column_mapping = {
                'gender_id': 'gender_code',  # Map gender_id to demo_sex_at_birth for initial data access
                'drug_id': 'drug_id',        # Keep drug_id as is
                'group_id': 'group_id',      # Keep group_id as is
                'subID': 'subID'             # Keep subID as is
            }
            
            # Map requested columns to actual column names in the data
            mapped_columns = []
            missing_columns = []
            
            for col in include_columns:
                if col in df_behav.columns:
                    mapped_columns.append(col)
                elif col in column_mapping and column_mapping[col] in df_behav.columns:
                    mapped_columns.append(column_mapping[col])
                    logger.info(f"Mapped column '{col}' to '{column_mapping[col]}' for data access")
                else:
                    missing_columns.append(col)
            
            if missing_columns:
                raise ValueError(f"Requested columns not found: {missing_columns}. "
                               f"Available columns: {list(df_behav.columns)}. "
                               f"Note: gender_id maps to demo_sex_at_birth for initial data access")
            
            # Use mapped columns for data processing, but keep original names for output
            data_columns = mapped_columns
            output_columns = include_columns  # Keep original column names for output
        else:
            # Default columns: always include subID and group_id, add others if available
            include_columns = ['subID', 'group_id']
            if 'drug_id' in df_behav.columns:
                include_columns.append('drug_id')
            if 'guess_id' in df_behav.columns:
                include_columns.append('guess_id')
        
        logger.info(f"Loaded behavioral data for {len(df_behav)} subjects")
        logger.info(f"Groups: {group_levels.tolist()}")
        
        if include_columns:
            logger.info(f"Data processing columns: {data_columns}")
            logger.info(f"Output columns: {output_columns}")
        else:
            logger.info(f"Default columns: {include_columns}")
        
        return df_behav, output_columns if include_columns else include_columns
        
    except Exception as e:
        logger.error(f"Failed to load behavioral data: {e}")
        raise

def load_first_level_data():
    """
    Load first-level analysis data and get subject list.
    
    Returns:
        tuple: (BIDSLayout, list of subject IDs)
    """
    try:
        firstlevel_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/firstLevel_timeEffect')
        glayout = BIDSLayout(firstlevel_dir, validate=False, config=['bids', 'derivatives'])
        sub_list = sorted(glayout.get_subjects())
        
        logger.info(f"Loaded first-level data for {len(sub_list)} subjects")
        return glayout, sub_list
        
    except Exception as e:
        logger.error(f"Failed to load first-level data: {e}")
        raise

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def collect_task_data(task, contrast, subject_list, glayout):
    """
    Collect cope and varcope files for a specific task and contrast.
    
    Args:
        task (str): Task name (e.g., 'phase2', 'phase3')
        contrast (int): Contrast number
        subject_list (list): List of subject IDs
        glayout (BIDSLayout): BIDS layout for first-level data
    
    Returns:
        tuple: (list of cope files, list of varcope files)
    """
    copes, varcopes = [], []
    
    for sub in subject_list:
        try:
            # Get cope file
            cope_file = glayout.get(
                subject=sub, 
                task=task, 
                desc=f'cope{contrast}',
                extension=['.nii', '.nii.gz'], 
                return_type='file'
            )
            
            # Get varcope file
            varcope_file = glayout.get(
                subject=sub, 
                task=task, 
                desc=f'varcope{contrast}',
                extension=['.nii', '.nii.gz'], 
                return_type='file'
            )
            
            if cope_file and varcope_file:
                copes.append(cope_file[0])
                varcopes.append(varcope_file[0])
            else:
                logger.warning(f"Missing files for task-{task}, sub-{sub}, cope{contrast}")
                
        except Exception as e:
            logger.error(f"Error collecting data for sub-{sub}, task-{task}, cope{contrast}: {e}")
            continue
    
    return copes, varcopes

def filter_subjects_for_task(subject_list, task, df_behav):
    """
    Filter subjects for a specific task, excluding those without MRI data.
    
    Args:
        subject_list (list): Full list of subject IDs
        task (str): Task name
        df_behav (pandas.DataFrame): Behavioral data
    
    Returns:
        pandas.DataFrame: Filtered behavioral data for the task
    """
    # Get subjects to exclude for this task
    subjects_to_exclude = SUBJECTS_NO_MRI.get(task, [])
    
    # Filter behavioral data
    filtered_df = df_behav.loc[
        df_behav['subID'].isin(subject_list) & 
        ~df_behav['subID'].isin(subjects_to_exclude)
    ]
    
    logger.info(f"Task {task}: {len(filtered_df)} subjects after filtering "
                f"(excluded {len(subjects_to_exclude)} subjects without MRI)")
    
    return filtered_df

# =============================================================================
# WORKFLOW EXECUTION FUNCTIONS
# =============================================================================

def run_data_preparation_workflow(task, contrast, group_info, copes, varcopes, 
                                 contrast_results_dir, contrast_workflow_dir, include_columns):
    """
    Run data preparation workflow for a specific task and contrast.
    
    Args:
        task (str): Task name
        contrast (int): Contrast number
        group_info (list): Group information for subjects
        copes (list): List of cope file paths
        varcopes (list): List of varcope file paths
        contrast_results_dir (str): Results directory for this contrast
        contrast_workflow_dir (str): Workflow directory for this contrast
        include_columns (list): List of columns included in group_info
    """
    try:
        # Create workflow
        prepare_wf = wf_data_prepare(
            output_dir=contrast_results_dir,
            contrast=contrast,
            name=f"data_prepare_{task}_cope{contrast}"
        )
        
        # Set workflow parameters
        prepare_wf.base_dir = contrast_workflow_dir
        
        # Set crash directory to /tmp to avoid read-only filesystem issues
        workflow_crash_dir = '/tmp/nipype_crashes'
        os.makedirs(workflow_crash_dir, exist_ok=True)
        prepare_wf.config['execution']['crash_dir'] = workflow_crash_dir
        
        prepare_wf.inputs.inputnode.in_copes = copes
        prepare_wf.inputs.inputnode.in_varcopes = varcopes
        prepare_wf.inputs.inputnode.group_info = group_info
        prepare_wf.inputs.inputnode.result_dir = contrast_results_dir
        prepare_wf.inputs.inputnode.group_mask = GROUP_MASK
        
        # Set analysis-specific parameters
        # Note: use_guess parameter removed as it's not needed for design generation
        
        # Clear Nipype cache to avoid file validation issues from stale cache files
        cache_dir = os.path.join(contrast_workflow_dir, prepare_wf.name)
        if os.path.exists(cache_dir):
            try:
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared Nipype cache: {cache_dir}")
            except Exception as e:
                logger.warning(f"Could not clear Nipype cache: {e}")
        
        # Also clear any individual node caches that might cause TraitError
        node_cache_dirs = [
            'resample_copes',
            'resample_varcopes', 
            'rename_copes',
            'rename_varcopes'
        ]
        for node_name in node_cache_dirs:
            node_cache_dir = os.path.join(cache_dir, node_name)
            if os.path.exists(node_cache_dir):
                try:
                    import shutil
                    shutil.rmtree(node_cache_dir)
                    logger.info(f"Cleared node cache: {node_cache_dir}")
                except Exception as e:
                    logger.warning(f"Could not clear node cache {node_name}: {e}")
        
        # Final crash directory setting to ensure it's correct
        prepare_wf.config['execution']['crash_dir'] = workflow_crash_dir
        
        # Force removal of unnecessary outputs to avoid cache conflicts
        prepare_wf.config['execution']['remove_unnecessary_outputs'] = True
        
        logger.info(f"Workflow crash directory set to: {workflow_crash_dir}")
        logger.info("Nipype cache cleared and configured to avoid file validation errors")
        
        logger.info(f"Running data preparation for task-{task}, contrast-{contrast}")
        prepare_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
        logger.info(f"Completed data preparation for task-{task}, contrast-{contrast}")
        
        # Copy results from workflow directory to final results directory
        logger.info(f"Copying results from workflow directory to final results directory")
        
        # Get the workflow output directory
        workflow_output_dir = os.path.join(contrast_workflow_dir, prepare_wf.name)
        
        if os.path.exists(workflow_output_dir):
            # Use the contrast results directory directly (no need to add 'whole_brain' again)
            final_results_dir = contrast_results_dir
            Path(final_results_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy all files from workflow output to final results
            import shutil
            import glob
            try:
                # Copy merged files
                for file_pattern in ['merged_cope*.nii.gz', 'merged_varcope*.nii.gz']:
                    for file_path in glob.glob(os.path.join(workflow_output_dir, file_pattern)):
                        filename = os.path.basename(file_path)
                        dest_path = os.path.join(final_results_dir, filename)
                        shutil.copy2(file_path, dest_path)
                        logger.info(f"Copied {filename} to {dest_path}")
                
                # Copy design files
                design_source_dir = os.path.join(workflow_output_dir, 'design_files')
                if os.path.exists(design_source_dir):
                    design_dest_dir = os.path.join(final_results_dir, 'design_files')
                    if os.path.exists(design_dest_dir):
                        shutil.rmtree(design_dest_dir)
                    shutil.copytree(design_source_dir, design_dest_dir)
                    logger.info(f"Copied design files to {design_dest_dir}")
                
                logger.info(f"Successfully copied all results to: {final_results_dir}")
                
            except Exception as e:
                logger.error(f"Failed to copy results: {e}")
                raise
        else:
            logger.warning(f"Workflow output directory not found: {workflow_output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to run data preparation workflow for task-{task}, contrast-{contrast}: {e}")
        raise

def cleanup_intermediate_directories(contrast_workflow_dir):
    """
    Clean up intermediate workflow directories to save space.
    
    Args:
        contrast_workflow_dir (str): Workflow directory to clean
    """
    intermediate_dirs = [
        'merge_copes', 'merge_varcopes', 
        'resample_copes', 'resample_varcopes'
    ]
    
    for dir_name in intermediate_dirs:
        dir_path = os.path.join(contrast_workflow_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.debug(f"Cleaned up intermediate directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {dir_path}: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        #description="Pre-group level fMRI analysis pipeline for NARSAD project. Creates merged COPE/VARCOPE files and design matrices for group-level analysis. Supports 2x2 and 2x2x2 factorial designs. Trans subjects (gender_code=3) are automatically excluded to prevent matrix singularity.",
        description="Pre-group level fMRI analysis pipeline for NARSAD project. Creates merged COPE/VARCOPE files and design matrices for group-level analysis. Supports 2x2 and 2x2x2 factorial designs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - all subjects, all phases
  python run_pre_group_voxelWise.py
  
  # Process specific phase only
  python run_pre_group_voxelWise.py --phase phase2
  
  # Process specific phase and contrast
  python run_pre_group_voxelWise.py --phase phase2 --cope 1
  
  # Standard 2x2 factorial design: Group × Drug
  python run_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id"
  
  # 2x2x2 factorial design: Group × Drug × Gender
  python run_pre_group_voxelWise.py --include-columns "subID,group_id,drug_id,gender_id"
  
  # Placebo-only analysis with 2x2 design
  python run_pre_group_voxelWise.py --data-source placebo --phase phase2 --include-columns "subID,group_id,drug_id"
  
  # Placebo-only analysis with 2x2x2 design (Trans subjects automatically excluded)
  python run_pre_group_voxelWise.py --data-source placebo --phase phase2 --include-columns "subID,group_id,drug_id,gender_id"
  
  # Guess condition analysis
  python run_pre_group_voxelWise.py --data-source guess --phase phase3 --include-columns "subID,group_id,guess_id"
  
  # Filter by specific drug condition
  python run_pre_group_voxelWise.py --filter-column Drug --filter-value Placebo --include-columns "subID,group_id,drug_id"
  
  # Filter by specific group
  python run_pre_group_voxelWise.py --filter-column group --filter-value Patients --include-columns "subID,group_id,drug_id"
  
  # Custom output directory
  python run_pre_group_voxelWise.py --output-dir /custom/path --include-columns "subID,group_id,drug_id"
  
  # Process specific subject
  python run_pre_group_voxelWise.py --subject N101 --phase phase2 --include-columns "subID,group_id,drug_id"
        """
    )
    
    parser.add_argument(
        '--filter-column',
        type=str,
        help='Column name to filter on (e.g., Drug, group, guess, drug_condition)'
    )
    
    parser.add_argument(
        '--filter-value',
        type=str,
        help='Value to filter for (e.g., Placebo, Patients, High, DrugA)'
    )
    
    parser.add_argument(
        '--include-columns',
        type=str,
        help='Comma-separated list of columns to include in group_info (default: auto-detect). '
             'Available columns: subID, group_id, drug_id, gender_id, guess_id. '
             'Note: gender_id maps to demo_sex_at_birth column in data. '
             #'Trans subjects (gender_code=3) are automatically excluded to prevent matrix singularity.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (overrides default)'
    )
    
    parser.add_argument(
        '--workflow-dir',
        type=str,
        help='Custom workflow directory (overrides default)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        help='Specific subject ID to process (e.g., sub-001)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['phase2', 'phase3'],
        help='Specific phase to process'
    )
    
    parser.add_argument(
        '--data-source',
        choices=['standard', 'placebo', 'guess'],
        default='standard',
        help='Data source to process: standard (default), placebo (Drug==Placebo only), or guess (guess condition analysis)'
    )
    
    parser.add_argument(
        '--cope',
        type=int,
        help='Specific cope number to process (e.g., 1, 2, 3)'
    )
    
    args = parser.parse_args()
    
    # Debug: Log all received arguments
    logger.info(f"Received arguments: {vars(args)}")
    
    # Validate arguments
    if args.filter_column and not args.filter_value:
        parser.error("--filter-column requires --filter-value")
    if args.filter_value and not args.filter_column:
        parser.error("--filter-value requires --filter-column")
    
    try:
        # Parse include_columns if provided
        include_columns = None
        if args.include_columns:
            include_columns = [col.strip() for col in args.include_columns.split(',')]
        
        # Create analysis description
        filtering_applied = []
        
        # Check for explicit column filtering
        if args.filter_column and args.filter_value:
            filtering_applied.append(f"{args.filter_column}={args.filter_value}")
        
        # Check for data source filtering
        if args.data_source and args.data_source != 'standard':
            if args.data_source == 'placebo':
                filtering_applied.append("placebo subjects only")
            elif args.data_source == 'guess':
                filtering_applied.append("guess condition filtering")
            else:
                filtering_applied.append(f"data source: {args.data_source}")
        
        if filtering_applied:
            analysis_desc = f"filtered by {', '.join(filtering_applied)}"
        else:
            analysis_desc = "all subjects (no filtering)"
        
        logger.info(f"Starting pre-group level analysis pipeline: {analysis_desc}")
        
        # Set up directories based on data source
        if args.output_dir:
            # Use provided output_dir as base, but still add data source components
            base_results_dir = args.output_dir
            logger.info(f"Using custom output directory as base: {base_results_dir}")
        else:
            # Standard base: groupLevel/whole_brain
            base_results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel_timeEffect/whole_brain')
            logger.info(f"Using default base directory: {base_results_dir}")
        
        # Always add data source components to the base directory
        if args.data_source and args.data_source != 'standard':
            # Check if base_results_dir already contains 'whole_brain'
            if 'whole_brain' in base_results_dir:
                # Base already has whole_brain, just add data source
                results_dir = os.path.join(base_results_dir, args.data_source.capitalize())
            else:
                # Base doesn't have whole_brain, add it along with data source
                results_dir = os.path.join(base_results_dir, 'whole_brain', args.data_source.capitalize())
            logger.info(f"Using data source specific results directory: {results_dir}")
        else:
            # Check if base_results_dir already contains 'whole_brain'
            if 'whole_brain' in base_results_dir:
                # Base already has whole_brain, use as is
                results_dir = base_results_dir
            else:
                # Base doesn't have whole_brain, add it
                results_dir = os.path.join(base_results_dir, 'whole_brain')
            logger.info(f"Using standard results directory: {results_dir}")
        
        if args.workflow_dir:
            workflow_dir = args.workflow_dir
        else:
            # Standard base: groupLevel/whole_brain
            base_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows/groupLevel_timeEffect/whole_brain')
            
            # Add data source subdirectory if not 'standard'
            if args.data_source and args.data_source != 'standard':
                workflow_dir = os.path.join(base_workflow_dir, args.data_source.capitalize())
                logger.info(f"Using data source specific workflow directory: {workflow_dir}")
            else:
                workflow_dir = base_workflow_dir
                logger.info(f"Using standard workflow directory: {workflow_dir}")
        
        # Create workflow directory (use temporary location to avoid read-only issues)
        Path(workflow_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Workflow directory: {workflow_dir}")
        
        # Load data
        df_behav, final_include_columns = load_behavioral_data(
            args.filter_column, args.filter_value, include_columns, args.data_source
        )
        glayout, subject_list = load_first_level_data()
        
        if len(df_behav) == 0:
            logger.error("No subjects found after filtering. Check your filter criteria.")
            return 1
        
        # Determine which tasks to process
        if args.phase:
            tasks_to_process = [args.phase]
            logger.info(f"Processing specific phase: {args.phase}")
        else:
            tasks_to_process = TASKS
            logger.info("Processing all phases")
        
        # Process each task
        for task in tasks_to_process:
            logger.info(f"Processing task: {task}")
            
            # Filter subjects for this task
            task_group_info_df = filter_subjects_for_task(subject_list, task, df_behav)
            
            # If specific subject requested, filter to that subject
            if args.subject:
                if args.subject in task_group_info_df['subID'].values:
                    task_group_info_df = task_group_info_df[task_group_info_df['subID'] == args.subject]
                    logger.info(f"Processing single subject: {args.subject}")
                else:
                    logger.warning(f"Subject {args.subject} not found in task {task}, skipping")
                    continue
            
            # Use the correct column names for data processing
            processing_columns = final_include_columns
            if args.include_columns:
                # If include_columns was specified, we need to map back to the actual column names in the data
                column_mapping = {
                    'gender_id': 'gender_code',  # Map gender_id to demo_sex_at_birth for initial data access
                    'drug_id': 'drug_id',        # Keep drug_id as is
                    'group_id': 'group_id',      # Keep group_id as is
                    'subID': 'subID'             # Keep subID as is
                }
                # Map output column names back to data column names for processing
                processing_columns = []
                for col in final_include_columns:
                    if col in column_mapping:
                        # Use the actual column name in the data
                        processing_columns.append(column_mapping[col])
                    else:
                        # Keep the column name as is
                        processing_columns.append(col)
                logger.info(f"Processing with columns: {processing_columns}")
            
            # GENDER PROCESSING: If gender_id is requested, create proper gender_id column for 2×2 factorial design
            if args.include_columns and 'gender_id' in args.include_columns:
                logger.info("Processing gender_id for 2×2 factorial design")
                
                # GENDER LEVEL RECODING: Recode gender levels from (0,1) to (1,2) for 2×2 factorial design
                # This prevents the 6-column design matrix issue (2 groups × 3 genders = 6 columns)
                logger.info("Recoding gender levels from (0,1) to (1,2) for 2×2 factorial design")
                task_group_info_df['gender_id'] = task_group_info_df['gender_code'].map({0: 1, 1: 2})
                logger.info("Gender level recoding complete: 0→1 (Female), 1→2 (Male)")
                
                # Update processing_columns to use gender_id instead of demo_sex_at_birth for the final group_info
                # This ensures we use the recoded values (1,2) instead of the original (0,1)
                # if 'demo_sex_at_birth' in processing_columns:
                #     processing_columns = [col if col != 'demo_sex_at_birth' else 'gender_id' for col in processing_columns]
                #     logger.info(f"Updated processing_columns to use recoded gender_id: {processing_columns}")
            
            group_info = list(task_group_info_df[processing_columns].itertuples(index=False, name=None))
            expected_subjects = len(group_info)
            
            if expected_subjects == 0:
                logger.warning(f"No subjects found for task {task}, skipping")
                continue
            
            # Create task directories
            task_results_dir = os.path.join(results_dir, f'task-{task}')
            task_workflow_dir = os.path.join(workflow_dir, f'task-{task}')
            
            Path(task_results_dir).mkdir(parents=True, exist_ok=True)
            
            # Create workflow directory
            Path(task_workflow_dir).mkdir(parents=True, exist_ok=True)
            
            # Get dynamic contrast range for this task
            task_contrast_range = get_contrast_range(task)
            
            # If specific cope requested, filter to that cope only
            logger.info(f"Debug: args.cope = {args.cope}, type = {type(args.cope)}")
            if args.cope:
                if args.cope in task_contrast_range:
                    task_contrast_range = [args.cope]
                    logger.info(f"Processing specific cope: {args.cope}")
                else:
                    logger.warning(f"Cope {args.cope} not found in task {task}, skipping")
                    continue
            else:
                logger.info(f"Task {task}: Processing contrasts {task_contrast_range[0]}-{task_contrast_range[-1]} (total: {len(task_contrast_range)})")
            
            # Process each contrast
            for contrast in task_contrast_range:
                logger.info(f"Processing contrast {contrast}")
                
                # Create contrast directories
                contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
                contrast_workflow_dir = os.path.join(task_workflow_dir, f'cope{contrast}')
                
                Path(contrast_results_dir).mkdir(parents=True, exist_ok=True)
                Path(contrast_workflow_dir).mkdir(parents=True, exist_ok=True)
                
                # Check if workflow directory is writable
                try:
                    test_file = os.path.join(contrast_workflow_dir, 'test_write.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    logger.warning(f"Workflow directory {contrast_workflow_dir} is not writable: {e}")
                    # Use a fallback directory
                    contrast_workflow_dir = os.path.join('/tmp', f'workflow_{task}_cope{contrast}')
                    Path(contrast_workflow_dir).mkdir(parents=True, exist_ok=True)
                    logger.info(f"Using fallback workflow directory: {contrast_workflow_dir}")
                
                # Collect data for this contrast
                copes, varcopes = collect_task_data(
                    task, contrast, [info[0] for info in group_info], glayout
                )
                
                # Check if we have complete data
                if len(copes) != expected_subjects or len(varcopes) != expected_subjects:
                    logger.warning(f"Skipping contrast {contrast}: Expected {expected_subjects} subjects, "
                                  f"got copes={len(copes)}, varcopes={len(varcopes)}")
                    continue
                
                # Run data preparation workflow
                run_data_preparation_workflow(
                    task, contrast, group_info, copes, varcopes,
                    contrast_results_dir, contrast_workflow_dir, final_include_columns
                )
                

        
        logger.info(f"Pre-group level analysis pipeline completed successfully: {analysis_desc}")
        
    except Exception as e:
        logger.error(f"Pre-group level analysis pipeline failed: {e}")
        raise

if __name__ == "__main__":
    exit(main())
