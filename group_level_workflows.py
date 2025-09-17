from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import Merge, FLIRT, ExtractROI, FLAMEO, Randomise, ImageMaths, SmoothEstimate, Threshold, Cluster
from nipype import DataSink
import os
import shutil
import glob
import subprocess
import pandas as pd
import numpy as np

# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# =============================================================================
# WORKFLOW DEFINITIONS
# =============================================================================

def wf_data_prepare(output_dir, contrast, name="wf_data_prepare"):
    """Workflow for data preparation and merging (renamed from data_prepare_wf)."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['in_copes', 'in_varcopes', 'group_info', 'result_dir', 'group_mask']),
                     name='inputnode')

    # Design generation
    design_gen = Node(Function(input_names=['group_info', 'output_dir'],
                               output_names=['design_file', 'grp_file', 'con_file'],
                               function=create_dummy_design_files,
                               imports=['import os', 'import numpy as np', 'import pandas as pd']),
                      name='design_gen')
    design_gen.inputs.output_dir = output_dir

    # Merge nodes
    merge_copes = Node(Merge(dimension='t', output_type='NIFTI_GZ'), name='merge_copes')
    merge_varcopes = Node(Merge(dimension='t', output_type='NIFTI_GZ'), name='merge_varcopes')

    # Resample nodes with explicit output file specification
    resample_copes = Node(FLIRT(apply_isoxfm=2), name='resample_copes')
    resample_varcopes = Node(FLIRT(apply_isoxfm=2), name='resample_varcopes')

    # Rename nodes
    rename_copes = Node(Function(input_names=['in_file', 'output_dir', 'contrast', 'file_type'],
                                 output_names=['out_file'],
                                 function=rename_file),
                        name='rename_copes')
    rename_copes.inputs.output_dir = output_dir
    rename_copes.inputs.contrast = contrast
    rename_copes.inputs.file_type = 'cope'

    rename_varcopes = Node(Function(input_names=['in_file', 'output_dir', 'contrast', 'file_type'],
                                    output_names=['out_file'],
                                    function=rename_file),
                           name='rename_varcopes')
    rename_varcopes.inputs.output_dir = output_dir
    rename_varcopes.inputs.contrast = contrast
    rename_varcopes.inputs.file_type = 'varcope'

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir, parameterization=False), name="datasink")

    # Workflow connections
    wf.connect([
        (inputnode, design_gen, [('group_info', 'group_info')]),
        (inputnode, merge_copes, [('in_copes', 'in_files')]),
        (inputnode, merge_varcopes, [('in_varcopes', 'in_files')]),
        (inputnode, resample_copes, [('group_mask', 'reference')]),
        (inputnode, resample_varcopes, [('group_mask', 'reference')]),
        (merge_copes, resample_copes, [('merged_file', 'in_file')]),
        (merge_varcopes, resample_varcopes, [('merged_file', 'in_file')]),
        (resample_copes, rename_copes, [('out_file', 'in_file')]),
        (resample_varcopes, rename_varcopes, [('out_file', 'in_file')]),
        (rename_copes, datasink, [('out_file', 'merged_copes')]),
        (rename_varcopes, datasink, [('out_file', 'merged_varcopes')]),
        (design_gen, datasink, [('design_file', 'design_files.design_file'),
                                ('grp_file', 'design_files.grp_file'),
                                ('con_file', 'design_files.con_file')])
    ])

    return wf


def wf_roi_extract(output_dir, roi_dir="/Users/xiaoqianxiao/tool/parcellation/ROIs", name="wf_roi_extract"):
    """Workflow to extract ROI beta values and PSC from fMRI data.
    
    This workflow extracts mean beta values and percent signal change (PSC) 
    from ROI masks. It takes cope files as input and computes statistics
    within each ROI region.
    
    WHEN TO USE THIS WORKFLOW:
    
    1. DESCRIPTIVE ROI ANALYSIS:
       - When you want to extract mean values from ROIs
       - For descriptive statistics (mean, std, etc.)
       - When you need ROI values for further analysis
       - For data exploration and visualization
    
    2. VALUE EXTRACTION:
       - When you need beta values for each subject in each ROI
       - When you need PSC values for each subject in each ROI
       - For creating ROI-based datasets
       - For exporting data to other analysis tools (R, SPSS, etc.)
    
    3. BASELINE COMPARISON:
       - When you have a baseline condition to compare against
       - For computing percent signal change
       - When you want to normalize effects relative to baseline
       - For cross-study comparisons
    
    4. DATA PREPARATION:
       - When preparing data for statistical analysis in other software
       - For creating summary tables and figures
       - When you need ROI values for correlation analysis
       - For meta-analysis or systematic reviews
    
    5. CLINICAL INTERPRETATION:
       - When you need interpretable values (PSC in %)
       - For communicating results to clinicians
       - When you want to compare effect sizes across studies
       - For patient-specific analysis
    
    6. EXPLORATORY ANALYSIS:
       - When you want to examine ROI values without statistical testing
       - For data quality checks
       - For identifying outliers
       - For understanding effect sizes before formal testing
    
    OUTPUTS:
    - beta_csv: Mean beta values for each ROI across subjects
    - psc_csv: Percent signal change values for each ROI across subjects
    - Individual text files for each ROI with subject-wise values
    
    Args:
        output_dir (str): Output directory
        roi_dir (str): Directory containing ROI mask files
        name (str): Workflow name
    """
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(IdentityInterface(fields=['cope_file', 'baseline_file', 'result_dir']),
                     name='inputnode')

    # Node to get ROI files
    roi_node = Node(Function(input_names=['roi_dir'], output_names=['roi_files'], function=get_roi_files),
                    name='roi_node')
    roi_node.inputs.roi_dir = roi_dir

    # MapNode to extract values for each ROI
    roi_extract = MapNode(Function(input_names=['cope_file', 'roi_mask', 'baseline_file', 'output_dir'],
                                   output_names=['beta_file', 'psc_file'],
                                   function=extract_roi_values),
                          iterfield=['roi_mask'],
                          name='roi_extract')
    roi_extract.inputs.output_dir = os.path.join(output_dir, 'roi_temp')  # Temporary directory

    # Node to combine values into CSV
    roi_combine = Node(Function(input_names=['beta_files', 'psc_files', 'output_dir'],
                                output_names=['beta_csv', 'psc_csv'],
                                function=combine_roi_values),
                       name='roi_combine')
    roi_combine.inputs.output_dir = output_dir

    # Output node
    outputnode = Node(IdentityInterface(fields=['beta_csv', 'psc_csv']),
                      name='outputnode')

    # DataSink
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections
    wf.connect([
        # Inputs to roi_extract
        (inputnode, roi_extract, [('cope_file', 'cope_file'),
                                  ('baseline_file', 'baseline_file')]),
        (roi_node, roi_extract, [('roi_files', 'roi_mask')]),

        # Combine results
        (roi_extract, roi_combine, [('beta_file', 'beta_files'),
                                    ('psc_file', 'psc_files')]),

        # Outputs to outputnode
        (roi_combine, outputnode, [('beta_csv', 'beta_csv'),
                                   ('psc_csv', 'psc_csv')]),

        # Outputs to DataSink
        (outputnode, datasink, [('beta_csv', 'roi_results.@beta_csv'),
                                ('psc_csv', 'roi_results.@psc_csv')])
    ])

    return wf





def wf_flameo(output_dir, name="wf_flameo"):
    """Workflow for group-level analysis with FLAMEO and clustering (GRF with dlh)."""
    wf = Workflow(name=name, base_dir=output_dir)

    inputnode = Node(IdentityInterface(fields=['cope_file', 'var_cope_file', 'mask_file',
                                               'design_file', 'grp_file', 'con_file', 'result_dir']),
                     name='inputnode')

    flameo = Node(FLAMEO(run_mode='flame1'), name='flameo')  # flame1 for mixed effects

    # Smoothness estimation for GRF clustering
    smoothness = MapNode(SmoothEstimate(),
                         iterfield=['zstat_file'],  # Only zstat_file iterates
                         name='smoothness')

    # Clustering node with dlh for GRF-based correction
    clustering = MapNode(Cluster(threshold=2.3,  # Z-threshold (e.g., 2.3 or 3.1)
                                 connectivity=26,  # 3D connectivity
                                 out_threshold_file=True,
                                 out_index_file=True,
                                 out_localmax_txt_file=True,  # Local maxima text file
                                 pthreshold=0.05),  # Cluster-level FWE threshold
                         iterfield=['in_file', 'dlh', 'volume'],
                         name='clustering')

    outputnode = Node(IdentityInterface(fields=['zstats', 'cluster_thresh', 'cluster_index', 'cluster_peaks']),
                      name='outputnode')

    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    wf.connect([
        # Inputs to FLAMEO
        (inputnode, flameo, [('cope_file', 'cope_file'),
                             ('var_cope_file', 'var_cope_file'),
                             ('mask_file', 'mask_file'),
                             ('design_file', 'design_file'),
                             ('grp_file', 'cov_split_file'),
                             ('con_file', 't_con_file')]),

        # Smoothness estimation
        (flameo, smoothness, [(('zstats', flatten_stats), 'zstat_file')]),
        (inputnode, smoothness, [('mask_file', 'mask_file')]),  # Single mask, no iteration

        # Clustering with dlh
        (flameo, clustering, [(('zstats', flatten_stats), 'in_file')]),
        (smoothness, clustering, [('volume', 'volume')]),
        (smoothness, clustering, [('dlh', 'dlh')]),

        # Outputs to outputnode
        (flameo, outputnode, [('zstats', 'zstats')]),
        (clustering, outputnode, [('threshold_file', 'cluster_thresh'),
                                  ('index_file', 'cluster_index'),
                                  ('localmax_txt_file', 'cluster_peaks')]),

        # Outputs to DataSink
        (outputnode, datasink, [('zstats', 'stats.@zstats'),
                                ('cluster_thresh', 'cluster_results.@thresh'),
                                ('cluster_index', 'cluster_results.@index'),
                                ('cluster_peaks', 'cluster_results.@peaks')])
    ])
    return wf

def wf_randomise(output_dir, name="wf_randomise"):
    """Workflow for group-level analysis with Randomise and TFCE."""
    wf = Workflow(name=name, base_dir=output_dir)
    inputnode = Node(IdentityInterface(fields=['cope_file', 'mask_file', 'design_file', 'con_file']),
                     name='inputnode')
    randomise = Node(Randomise(num_perm=5000,  # Number of permutations
                               tfce=True),      # Use TFCE
                     name='randomise')
    outputnode = Node(IdentityInterface(fields=['tstat_files', 'tfce_corr_p_files']),
                      name='outputnode')
    datasink = Node(DataSink(base_directory=output_dir, parameterization=False), name='datasink')
    # Workflow connections
    wf.connect([
        # Inputs to Randomise
        (inputnode, randomise, [('cope_file', 'in_file'),
                                ('mask_file', 'mask'),
                                ('design_file', 'design_mat'),
                                ('con_file', 'tcon')]),
        # Outputs to outputnode
        (randomise, outputnode, [('tstat_files', 'tstat_files'),
                                 ('t_corrected_p_files', 'tfce_corr_p_files')]),
        # Outputs to DataSink
        (outputnode, datasink, [('tstat_files', 'stats.@tstats'),
                                ('tfce_corr_p_files', 'stats.@tfce_corr_p')])
    ])
    return wf

# =============================================================================
# UNIFIED GROUP-LEVEL ANALYSIS WORKFLOW
# =============================================================================

def create_group_analysis_workflow(output_dir, method='flameo', subjects=None, 
                                 group_coding='1/0', contrast_type='standard',
                                 analysis_type='whole_brain', roi_dir=None, **kwargs):
    """
    Create a complete group analysis workflow with design matrix generation.
    
    This function can create both whole-brain and ROI-based analysis workflows.
    
    Args:
        output_dir (str): Output directory
        method (str): 'flameo' or 'randomise'
        subjects (list): List of subject IDs
        group_coding (str): '1/0' or '1/-1' for group coding
        contrast_type (str): 'standard' or 'minimal' for contrasts
        analysis_type (str): 'whole_brain' or 'roi' - type of analysis to perform
        roi_dir (str): Directory containing ROI mask files (required if analysis_type='roi')
        **kwargs: Additional arguments for specific workflows
    
    Returns:
        tuple: (workflow, design_file, con_file)
    """
    # Create design matrix and contrasts
    if subjects is not None:
        design, contrasts = create_flexible_design_matrix(
            subjects, group_coding=group_coding, contrast_type=contrast_type
        )
        
        # Save design files
        design_file = os.path.join(output_dir, 'design.mat')
        con_file = os.path.join(output_dir, 'contrast.con')
        save_vest_file(design, design_file)
        save_vest_file(np.array(contrasts), con_file)
    else:
        design_file = None
        con_file = None

    # Create workflow based on analysis type and method
    if analysis_type == 'roi':
        if roi_dir is None:
            raise ValueError("roi_dir must be specified for ROI-based analysis")
        
        # For ROI analysis, we need to create a custom workflow since the standalone workflows
        # are designed for whole-brain analysis
        wf = Workflow(name=f"roi_analysis_{method}", base_dir=output_dir)
        
        # Input node for ROI analysis
        if method == 'flameo':
            inputnode = Node(IdentityInterface(fields=['cope_files', 'var_cope_files', 'mask_file',
                                                       'design_file', 'con_file', 'result_dir']),
                             name='inputnode')
        else:  # randomise
            inputnode = Node(IdentityInterface(fields=['cope_files', 'mask_file',
                                                       'design_file', 'con_file']),
                             name='inputnode')
        
        # ROI node to fetch ROI files
        roi_node = Node(Function(input_names=['roi_dir'], output_names=['roi_files'],
                                 function=get_roi_files),
                        name='roi_node')
        roi_node.inputs.roi_dir = roi_dir
        
        # Masking for copes and varcopes
        mask_copes = MapNode(ImageMaths(op_string='-mul'), 
                             iterfield=['in_file2'], 
                             name='mask_copes')
        
        if method == 'flameo':
            mask_varcopes = MapNode(ImageMaths(op_string='-mul'),
                                    iterfield=['in_file2'],
                                    name='mask_varcopes')
        
        # Analysis node for each ROI
        if method == 'flameo':
            analysis_node = MapNode(FLAMEO(run_mode='flame1'),
                                   iterfield=['cope_file', 'var_cope_file', 'mask_file'],
                                   name='flameo')
        else:  # randomise
            analysis_node = MapNode(Randomise(num_perm=10000, tfce=True, vox_p_values=True),
                                   iterfield=['in_file', 'mask', 'design_file', 'tcon'],
                                   name='randomise')
        
        # Statistical thresholding
        if method == 'flameo':
            fdr_ztop = MapNode(ImageMaths(op_string='-ztop', suffix='_pval'),
                               iterfield=['in_file'],
                               name='fdr_ztop')
            smoothness = MapNode(SmoothEstimate(),
                                 iterfield=['zstat_file', 'mask_file'],
                                 name='smoothness')
            fwe_thresh = MapNode(Threshold(thresh=0.05, direction='above'),
                                 iterfield=['in_file'],
                                 name='fwe_thresh')
        else:  # randomise
            tfce_ztop = MapNode(ImageMaths(op_string='-ztop', suffix='_zstat'),
                                iterfield=['in_file'],
                                name='tfce_ztop')
        
        # Output node
        if method == 'flameo':
            outputnode = Node(IdentityInterface(fields=['zstats', 'fdr_thresh', 'fwe_thresh']),
                              name='outputnode')
        else:  # randomise
            outputnode = Node(IdentityInterface(fields=['tstat_files', 'tfce_corr_p_files', 'z_thresh_files']),
                              name='outputnode')
        
        # DataSink
        datasink = Node(DataSink(base_directory=output_dir), name='datasink')
        
        # ROI workflow connections
        if method == 'flameo':
            wf.connect([
                (inputnode, roi_node, [('cope_files', 'cope_files')]),
                (roi_node, mask_copes, [('roi_files', 'in_file2')]),
                (roi_node, mask_varcopes, [('roi_files', 'in_file2')]),
                (roi_node, analysis_node, [('roi_files', 'mask_file')]),
                (roi_node, smoothness, [('roi_files', 'mask_file')]),
                
                (inputnode, mask_copes, [('cope_files', 'in_file')]),
                (inputnode, mask_varcopes, [('var_cope_files', 'in_file')]),
                
                (mask_copes, analysis_node, [('out_file', 'cope_file')]),
                (mask_varcopes, analysis_node, [('out_file', 'var_cope_file')]),
                
                (inputnode, analysis_node, [('design_file', 'design_file'),
                                           ('con_file', 't_con_file')]),
                
                (analysis_node, fdr_ztop, [(('zstats', flatten_zstats), 'in_file')]),
                (analysis_node, smoothness, [(('zstats', flatten_zstats), 'zstat_file')]),
                (analysis_node, fwe_thresh, [(('zstats', flatten_zstats), 'in_file')]),
                
                (analysis_node, outputnode, [('zstats', 'zstats')]),
                (fdr_ztop, outputnode, [('out_file', 'fdr_thresh')]),
                (fwe_thresh, outputnode, [('out_file', 'fwe_thresh')]),
                (roi_node, outputnode, [('roi_files', 'roi_files')]),
                
                (outputnode, datasink, [('zstats', 'zstats'),
                                        ('fdr_thresh', 'fdr_thresh'),
                                        ('fwe_thresh', 'fwe_thresh'),
                                        ('roi_files', 'roi_files')])
            ])
        else:  # randomise
            wf.connect([
                (inputnode, roi_node, [('cope_files', 'cope_files')]),
                (roi_node, mask_copes, [('roi_files', 'in_file2')]),
                (roi_node, analysis_node, [('roi_files', 'mask')]),
                
                (inputnode, mask_copes, [('cope_files', 'in_file')]),
                
                (mask_copes, analysis_node, [('out_file', 'in_file')]),
                
                (inputnode, analysis_node, [('design_file', 'design_file'),
                                           ('con_file', 'tcon')]),
                
                (analysis_node, tfce_ztop, [('t_corrected_p_files', 'in_file')]),
                
                (analysis_node, outputnode, [('tstat_files', 'tstat_files'),
                                            ('t_corrected_p_files', 'tfce_corr_p_files')]),
                (tfce_ztop, outputnode, [('out_file', 'z_thresh_files')]),
                (roi_node, outputnode, [('roi_files', 'roi_files')]),
                
                (outputnode, datasink, [('tstat_files', 'tstats'),
                                        ('t_corrected_p_files', 'tfce_p'),
                                        ('z_thresh_files', 'zscores'),
                                        ('roi_files', 'roi_files')])
            ])
    
    else:  # whole_brain
        # For whole-brain analysis, use the standalone workflows directly
        if method == 'flameo':
            wf = wf_flameo(output_dir, **kwargs)
        elif method == 'randomise':
            wf = wf_randomise(output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'flameo' or 'randomise'")

    return wf, design_file, con_file

def run_group_analysis(cope_files, var_cope_files=None, mask_file=None, 
                      subjects=None, output_dir=None, method='flameo',
                      group_coding='1/0', contrast_type='standard', 
                      analysis_type='whole_brain', roi_dir=None, **kwargs):
    """
    Run complete group-level analysis with automatic setup.
    
    This function can perform both whole-brain and ROI-based analysis.
    
    Args:
        cope_files (list): List of cope file paths
        var_cope_files (list): List of var_cope file paths (required for FLAMEO)
        mask_file (str): Mask file path
        subjects (list): List of subject IDs for design matrix
        output_dir (str): Output directory
        method (str): 'flameo' or 'randomise'
        group_coding (str): Group coding scheme
        contrast_type (str): Contrast type
        analysis_type (str): 'whole_brain' or 'roi' - type of analysis to perform
        roi_dir (str): Directory containing ROI mask files (required if analysis_type='roi')
        **kwargs: Additional workflow arguments
    
    Returns:
        Workflow: Configured and ready-to-run workflow
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create workflow
    wf, design_file, con_file = create_group_analysis_workflow(
        output_dir, method, subjects, group_coding, contrast_type, 
        analysis_type, roi_dir, **kwargs
    )
    
    # Set inputs
    wf.inputs.inputnode.cope_files = cope_files
    wf.inputs.inputnode.mask_file = mask_file
    
    if design_file:
        wf.inputs.inputnode.design_file = design_file
    if con_file:
        wf.inputs.inputnode.con_file = con_file
    
    # Set method-specific inputs
    if method == 'flameo':
        if var_cope_files is None:
            raise ValueError("var_cope_files required for FLAMEO method")
        wf.inputs.inputnode.var_cope_files = var_cope_files
        wf.inputs.inputnode.result_dir = output_dir
    elif method == 'randomise':
        # Randomise doesn't need var_cope_files
        pass
    
    return wf

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_dummy_design_files(group_info, output_dir, column_names=None, contrast_type='auto'):
    """
    Create design.mat, design.grp, and contrast.con for fMRI group-level analysis.
    
    This function is now generic and can handle any combination of columns in group_info.
    For 2x2 factorial designs, it creates 6 complete contrasts covering all main effects
    and interactions in both directions.
    
    Args:
        group_info (pandas.DataFrame or list): DataFrame where each column represents a condition/factor,
                                             or list of tuples where each tuple represents a subject's factors.
                                             Rows are subjects, columns are factors (e.g., group, drug, guess)
        output_dir (str): Output directory for design files
        column_names (list): Names of columns to use as factors (if None, uses all columns)
        contrast_type (str): Type of contrasts to generate ('auto', 'main_effects', 'interactions', 'custom')
    
    Examples:
        # Two-group comparison
        group_info = pd.DataFrame({
            'group': [1, 2, 1, 2],
            'subject': ['sub1', 'sub2', 'sub3', 'sub4']
        })
        column_names = ['group']  # Only use 'group' column
        
        # 2x2 factorial design
        group_info = pd.DataFrame({
            'group': [1, 1, 2, 2],
            'drug': ['A', 'B', 'A', 'B'],
            'subject': ['sub1', 'sub2', 'sub3', 'sub4']
        })
        column_names = ['group', 'drug']  # Use both factors
        
        # 2x2x2 factorial design
        group_info = pd.DataFrame({
            'group': [1, 1, 1, 1, 2, 2, 2, 2],
            'drug': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
            'guess': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'subject': ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8']
        })
        column_names = ['group', 'drug', 'guess']  # Use all three factors
    """
    # === Prepare output paths ===
    design_dir = os.path.join(output_dir, 'design_files')
    os.makedirs(design_dir, exist_ok=True)
    design_f = os.path.join(design_dir, 'design.mat')
    grp_f    = os.path.join(design_dir, 'design.grp')
    con_f    = os.path.join(design_dir, 'contrast.con')

    # === Generic design matrix generation ===
    n = len(group_info)
    
    # Handle both DataFrame and list inputs
    if hasattr(group_info, 'columns'):
        # Input is a pandas DataFrame
        if column_names is None:
            # Use all columns except 'subject' if it exists
            all_columns = list(group_info.columns)
            if 'subject' in all_columns:
                column_names = [col for col in all_columns if col != 'subject']
            else:
                column_names = all_columns
        
        # Trans subjects (gender_id==3) are already filtered out in run_pre_group_voxelWise.py
        # No need for additional safety checks here
    else:
        # Input is a list of tuples, convert to DataFrame
        import pandas as pd
        if column_names is None:
            # Default column names based on the data structure
            if len(group_info) > 0 and len(group_info[0]) == 4:
                column_names = ['subID', 'group_id', 'drug_id', 'guess_id']
            elif len(group_info) > 0 and len(group_info[0]) == 3:
                column_names = ['subID', 'group_id', 'drug_id']
            elif len(group_info) > 0 and len(group_info[0]) == 2:
                column_names = ['subID', 'group_id']
            else:
                column_names = [f'factor_{i}' for i in range(len(group_info[0]))]
        
        # Convert list of tuples to DataFrame
        group_info = pd.DataFrame(group_info, columns=column_names)
    
    # Trans subjects (gender_id==3) are already filtered out in run_pre_group_voxelWise.py
    # No need for additional safety checks here
    
    # Extract factor columns (exclude 'subID' if present)
    factor_columns = [col for col in column_names if col != 'subID']
    n_factors = len(factor_columns)
    
    # Get unique levels for each factor FROM THE FILTERED DATA
    # This ensures we don't include levels that were filtered out (like gender_id=3)
    factor_levels = {}
    for factor_name in factor_columns:
        factor_values = group_info[factor_name].values
        factor_levels[factor_name] = sorted(set(factor_values))
        print(f"Factor '{factor_name}' levels: {factor_levels[factor_name]} (from filtered data)")
    
    # Create design matrix - self-contained version
    if n_factors == 1:
        # Single factor (e.g., two-group comparison)
        factor_name = list(factor_levels.keys())[0]
        levels = factor_levels[factor_name]
        n_levels = len(levels)
        
        # Create design matrix (one column per level)
        design_matrix = []
        for _, row in group_info.iterrows():
            design_row = [0] * n_levels
            factor_value = row[factor_name]
            level_idx = levels.index(factor_value)
            design_row[level_idx] = 1
            design_matrix.append(design_row)
        
        # Create contrasts
        contrasts = []
        if n_levels == 2:
            # Two-group comparison
            contrasts = [
                [1, -1],  # Level 1 > Level 2
                [1, 0],   # Level 1 mean
                [0, 1],   # Level 2 mean
            ]
        else:
            # Multi-level factor
            for i in range(n_levels):
                for j in range(i+1, n_levels):
                    contrast = [0] * n_levels
                    contrast[i] = 1
                    contrast[j] = -1
                    contrasts.append(contrast)
                    
    elif n_factors == 2:
        # Two factors (e.g., 2x2 factorial) - self-contained version
        factor_names = list(factor_levels.keys())
        factor1_name = factor_names[0]
        factor2_name = factor_names[1]
        
        n_levels1 = len(factor_levels[factor1_name])
        n_levels2 = len(factor_levels[factor2_name])
        
        print(f"Creating 2-factor design matrix: {n_levels1} × {n_levels2} = {n_levels1 * n_levels2} columns")
        print(f"Expected design matrix shape: {len(group_info)} subjects × {n_levels1 * n_levels2} columns")
        
        # Create design matrix using cell-means coding
        design_matrix = []
        for _, row in group_info.iterrows():
            design_row = [0] * (n_levels1 * n_levels2)
            factor1_value = row[factor1_name]
            factor2_value = row[factor2_name]
            
            # Find the cell index
            level1_idx = factor_levels[factor1_name].index(factor1_value)
            level2_idx = factor_levels[factor2_name].index(factor2_value)
            cell_idx = level1_idx * n_levels2 + level2_idx
            
            design_row[cell_idx] = 1
            design_matrix.append(design_row)
        
        # Create contrasts for 2x2 factorial design
        contrasts = []
        if n_levels1 == 2 and n_levels2 == 2:
            print(f"Creating 6 contrasts for 2×2 factorial design (4 columns)")
        else:
            print(f"WARNING: Non-2×2 design detected: {n_levels1} × {n_levels2} = {n_levels1 * n_levels2} columns")
            print("This may cause matrix singularity issues!")
        
        if n_levels1 == 2 and n_levels2 == 2:
            # Contrast 1: Factor1 Level1 > Factor1 Level2
            contrast1 = [1, 1, -1, -1]  # (Cell 0+1) vs (Cell 2+3)
            contrasts.append(contrast1)
            
            # Contrast 2: Factor1 Level1 < Factor1 Level2 (reverse of Contrast 1)
            contrast2 = [-1, -1, 1, 1]  # (Cell 2+3) vs (Cell 0+1)
            contrasts.append(contrast2)
            
            # Contrast 3: Factor2 Level1 > Factor2 Level2
            contrast3 = [1, -1, 1, -1]  # (Cell 0+2) vs (Cell 1+3)
            contrasts.append(contrast3)
            
            # Contrast 4: Factor2 Level1 < Factor2 Level2 (reverse of Contrast 3)
            contrast4 = [-1, 1, -1, 1]  # (Cell 1+3) vs (Cell 0+2)
            contrasts.append(contrast4)
            
            # Contrast 5: Interaction
            contrast5 = [1, -1, -1, 1]  # (Cell 0) - (Cell 1) - (Cell 2) + (Cell 3)
            contrasts.append(contrast5)
            
            # Contrast 6: Opposite interaction (reverse of Contrast 5)
            contrast6 = [-1, 1, 1, -1]  # -(Cell 0) + (Cell 1) + (Cell 2) - (Cell 3)
            contrasts.append(contrast6)
        else:
            # For non-2x2 designs, use simple main effects
            for i in range(n_levels1):
                for j in range(i+1, n_levels1):
                    contrast = [0] * (n_levels1 * n_levels2)
                    # Main effect of factor 1
                    for k in range(n_levels2):
                        idx1 = i * n_levels2 + k
                        idx2 = j * n_levels2 + k
                        contrast[idx1] = 1
                        contrast[idx2] = -1
                    contrasts.append(contrast)
    else:
        # For 3+ factors, use a simple approach
        # Create a simple design matrix with one column per factor level
        design_matrix = []
        for i in range(n):
            row = []
            for factor_name in factor_columns:
                factor_value = group_info[factor_name].iloc[i]
                # Create dummy coding (1 for first level, 0 for others)
                if factor_value == factor_levels[factor_name][0]:
                    row.append(1)
                else:
                    row.append(0)
            design_matrix.append(row)
        
        # Create simple contrasts (main effects)
        contrasts = []
        for i, factor_name in enumerate(factor_columns):
            contrast = [0] * n_factors
            contrast[i] = 1
            contrasts.append(contrast)
    
    # Write design files
    num_evs = len(design_matrix[0])
    
    # Write design.mat
    with open(design_f, 'w') as f:
        f.write(f"/NumWaves {num_evs}\n/NumPoints {n}\n/Matrix\n")
        for row in design_matrix:
            f.write(" ".join(map(str, row)) + "\n")
    
    # Write design.grp (variance groups)
    with open(grp_f, 'w') as f:
        f.write("/NumWaves 1\n")
        f.write(f"/NumPoints {n}\n/Matrix\n")
        for _ in range(n):
            f.write("1\n")
    
    # Write contrast.con
    with open(con_f, 'w') as f:
        f.write(f"/NumWaves {num_evs}\n")
        f.write(f"/NumContrasts {len(contrasts)}\n/Matrix\n")
        for contrast in contrasts:
            f.write(" ".join(map(str, contrast)) + "\n")
    
    return design_f, grp_f, con_f

def create_single_factor_design(group_info, factor_levels, column_names):
    """Create design matrix for single factor (e.g., two-group comparison)."""
    factor_name = list(factor_levels.keys())[0]
    levels = factor_levels[factor_name]
    n_levels = len(levels)
    
    # Create design matrix (one column per level)
    design_matrix = []
    for _, row in group_info.iterrows():
        design_row = [0] * n_levels
        factor_value = row[factor_name]
        level_idx = levels.index(factor_value)
        design_row[level_idx] = 1
        design_matrix.append(design_row)
    
    # Create contrasts
    contrasts = []
    if n_levels == 2:
        # Two-group comparison
        contrasts = [
            [1, -1],  # Level 1 > Level 2
            [1, 0],   # Level 1 mean
            [0, 1],   # Level 2 mean
        ]
    else:
        # Multi-level factor
        for i in range(n_levels):
            for j in range(i+1, n_levels):
                contrast = [0] * n_levels
                contrast[i] = 1
                contrast[j] = -1
                contrasts.append(contrast)
    
    return design_matrix, contrasts

def create_two_factor_design(group_info, factor_levels, column_names, contrast_type):
    """Create design matrix for two-factor design with complete factorial contrasts."""
    factor_names = list(factor_levels.keys())
    levels1 = factor_levels[factor_names[0]]
    levels2 = factor_levels[factor_names[1]]
    n_levels1 = len(levels1)
    n_levels2 = len(levels2)
    n_cells = n_levels1 * n_levels2
    
    # Create design matrix
    design_matrix = []
    for _, row in group_info.iterrows():
        design_row = [0] * n_cells
        value1 = row[factor_names[0]]  # First factor
        value2 = row[factor_names[1]]  # Second factor
        cell_idx = levels1.index(value1) * n_levels2 + levels2.index(value2)
        design_row[cell_idx] = 1
        design_matrix.append(design_row)
    
    # Create contrasts
    contrasts = []
    if contrast_type in ['auto', 'main_effects', 'interactions']:
        # For 2x2 factorial design, create complete set of 6 contrasts
        if n_levels1 == 2 and n_levels2 == 2:
            # Contrast 1: Factor1 Level1 > Factor1 Level2
            contrast1 = [1, 1, -1, -1]  # (Cell 0+1) vs (Cell 2+3)
            contrasts.append(contrast1)
            
            # Contrast 2: Factor1 Level1 < Factor1 Level2 (reverse of Contrast 1)
            contrast2 = [-1, -1, 1, 1]  # (Cell 2+3) vs (Cell 0+1)
            contrasts.append(contrast2)
            
            # Contrast 3: Factor2 Level1 > Factor2 Level2
            contrast3 = [1, -1, 1, -1]  # (Cell 0+2) vs (Cell 1+3)
            contrasts.append(contrast3)
            
            # Contrast 4: Factor2 Level1 < Factor2 Level2 (reverse of Contrast 3)
            contrast4 = [-1, 1, -1, 1]  # (Cell 1+3) vs (Cell 0+2)
            contrasts.append(contrast4)
            
            # Contrast 5: Interaction
            contrast5 = [1, -1, -1, 1]  # (Cell 0) - (Cell 1) - (Cell 2) + (Cell 3)
            contrasts.append(contrast5)
            
            # Contrast 6: Opposite interaction (reverse of Contrast 5)
            contrast6 = [-1, 1, 1, -1]  # -(Cell 0) + (Cell 1) + (Cell 2) - (Cell 3)
            contrasts.append(contrast6)
            
        else:
            # For non-2x2 designs, use the original logic
            # Main effects
            for i in range(n_levels1):
                for j in range(i+1, n_levels1):
                    contrast = [0] * n_cells
                    for k in range(n_levels2):
                        idx1 = i * n_levels2 + k
                        idx2 = j * n_levels2 + k
                        contrast[idx1] = 1
                        contrast[idx2] = -1
                    contrasts.append(contrast)
            
            for i in range(n_levels2):
                for j in range(i+1, n_levels2):
                    contrast = [0] * n_cells
                    for k in range(n_levels1):
                        idx1 = k * n_levels2 + i
                        idx2 = k * n_levels2 + j
                        contrast[idx1] = 1
                        contrast[idx2] = -1
                    contrasts.append(contrast)
            
            if contrast_type in ['auto', 'interactions']:
                # Interaction effects (simplified)
                if n_levels1 == 2 and n_levels2 == 2:
                    contrast = [1, -1, -1, 1]  # Interaction
                    contrasts.append(contrast)
    
    return design_matrix, contrasts

def create_three_factor_design(group_info, factor_levels, column_names, contrast_type):
    """Create design matrix for three-factor design."""
    factor_names = list(factor_levels.keys())
    levels1 = factor_levels[factor_names[0]]
    levels2 = factor_levels[factor_names[1]]
    levels3 = factor_levels[factor_names[2]]
    n_levels1 = len(levels1)
    n_levels2 = len(levels2)
    n_levels3 = len(levels3)
    n_cells = n_levels1 * n_levels2 * n_levels3
    
    # Create design matrix
    design_matrix = []
    for _, row in group_info.iterrows():
        design_row = [0] * n_cells
        value1 = row[factor_names[0]]  # First factor
        value2 = row[factor_names[1]]  # Second factor
        value3 = row[factor_names[2]]  # Third factor
        cell_idx = (levels1.index(value1) * n_levels2 * n_levels3 + 
                   levels2.index(value2) * n_levels3 + 
                   levels3.index(value3))
        design_row[cell_idx] = 1
        design_matrix.append(design_row)
    
    # Create basic contrasts (main effects)
    contrasts = []
    if contrast_type in ['auto', 'main_effects']:
        # Main effects for each factor
        for i in range(n_levels1):
            for j in range(i+1, n_levels1):
                contrast = [0] * n_cells
                for k in range(n_levels2):
                    for l in range(n_levels3):
                        idx1 = i * n_levels2 * n_levels3 + k * n_levels3 + l
                        idx2 = j * n_levels2 * n_levels3 + k * n_levels3 + l
                        contrast[idx1] = 1
                        contrast[idx2] = -1
                contrasts.append(contrast)
    
    return design_matrix, contrasts

def create_general_factorial_design(group_info, factor_levels, column_names, contrast_type):
    """Create design matrix for general factorial design."""
    factor_names = list(factor_levels.keys())
    n_factors = len(factor_names)
    
    # Calculate total number of cells
    n_cells = 1
    for factor_name in factor_names:
        n_cells *= len(factor_levels[factor_name])
    
    # Create design matrix
    design_matrix = []
    for _, row in group_info.iterrows():
        design_row = [0] * n_cells
        cell_idx = calculate_cell_index(row, factor_levels, factor_names)
        design_row[cell_idx] = 1
        design_matrix.append(design_row)
    
    # Create basic contrasts (main effects for first two factors)
    contrasts = []
    if contrast_type in ['auto', 'main_effects'] and n_factors >= 2:
        # Main effect for first factor
        factor1_levels = factor_levels[factor_names[0]]
        factor2_levels = factor_levels[factor_names[1]]
        n_levels1 = len(factor1_levels)
        n_levels2 = len(factor2_levels)
        
        for i in range(n_levels1):
            for j in range(i+1, n_levels1):
                contrast = [0] * n_cells
                # This is a simplified version - would need more complex logic for general case
                contrasts.append(contrast)
    
    return design_matrix, contrasts

def calculate_cell_index(row, factor_levels, factor_names):
    """Calculate cell index for a given combination of factor levels."""
    cell_idx = 0
    multiplier = 1
    
    for i, factor_name in enumerate(factor_names):
        factor_value = row[factor_name]
        level_idx = factor_levels[factor_name].index(factor_value)
        cell_idx += level_idx * multiplier
        
        # Update multiplier for next factor
        if i < len(factor_names) - 1:
            next_factor = factor_names[i + 1]
            multiplier *= len(factor_levels[next_factor])
    
    return cell_idx


def test_dataframe_design():
    """
    Test function to demonstrate the new DataFrame functionality.
    
    This function shows how to use create_dummy_design_files with pandas DataFrames.
    """
    import pandas as pd
    import tempfile
    import os
    
    print("Testing DataFrame-based design matrix generation...")
    
    # Test 1: Two-group comparison
    print("\n1. Two-group comparison:")
    group_info = pd.DataFrame({
        'group': [1, 2, 1, 2, 1, 2],
        'subject': ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6']
    })
    print(f"Input DataFrame:\n{group_info}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        design_f, grp_f, con_f = create_dummy_design_files(
            group_info, temp_dir, column_names=['group']
        )
        print(f"Generated files: {design_f}, {grp_f}, {con_f}")
    
    # Test 2: 2x2 factorial design
    print("\n2. 2x2 factorial design:")
    group_info = pd.DataFrame({
        'group': [1, 1, 2, 2, 1, 1, 2, 2],
        'drug': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'subject': ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8']
    })
    print(f"Input DataFrame:\n{group_info}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        design_f, grp_f, con_f = create_dummy_design_files(
            group_info, temp_dir, column_names=['group', 'drug']
        )
        print(f"Generated files: {design_f}, {grp_f}, {con_f}")
    
    # Test 3: Auto-detect columns
    print("\n3. Auto-detect columns:")
    group_info = pd.DataFrame({
        'group': [1, 2, 1, 2],
        'drug': ['A', 'A', 'B', 'B'],
        'subject': ['sub1', 'sub2', 'sub3', 'sub4']
    })
    print(f"Input DataFrame:\n{group_info}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        design_f, grp_f, con_f = create_dummy_design_files(
            group_info, temp_dir  # column_names=None for auto-detection
        )
        print(f"Generated files: {design_f}, {grp_f}, {con_f}")
    
    print("\nAll tests completed successfully!")




def check_file_exists(in_file):
    """Check if a file exists and raise an error if not."""
    print(f"DEBUG: Checking file existence: {in_file}")
    import os
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"File {in_file} does not exist!")
    return in_file


def rename_file(in_file, output_dir, contrast, file_type):
    """Rename the merged file to a simpler name with error checking."""
    print(f"DEBUG: Received in_file: {in_file}, contrast: {contrast}, file_type: {file_type}")
    import shutil
    import os
    try:
        contrast_str = str(int(contrast))
    except (ValueError, TypeError):
        print(f"Warning: Invalid contrast value '{contrast}', defaulting to 'unknown'")
        contrast_str = "unknown"

    new_name = f"merged_{file_type}.nii.gz"
    out_file = os.path.join(output_dir, new_name)

    if os.path.exists(in_file):
        shutil.move(in_file, out_file)
        print(f"Renamed {in_file} -> {out_file}")
    else:
        raise FileNotFoundError(f"Input file {in_file} does not exist!")

    return out_file



def flatten_zstats(zstats):
    """Flatten a potentially nested list of z-stat file paths into a single list."""
    if not zstats:  # Handle empty input
        return []
    if isinstance(zstats, str):  # If it's a single string, wrap it in a list
        return [zstats]
    if isinstance(zstats[0], list):  # If it's a nested list, flatten it
        return [item for sublist in zstats for item in sublist]
    return zstats  # Already a flat list of strings

def flatten_stats(stats):
    """Flatten a potentially nested list of stat file paths into a single list."""
    if not stats:
        return []
    if isinstance(stats, str):
        return [stats]
    if isinstance(stats[0], list):
        return [item for sublist in stats for item in sublist]
    return stats

def flatten_list(nested):
    """Flatten a nested list of files into a 1D list."""
    return [f for sub in nested for f in sub]



def get_roi_files(roi_dir):
    """Retrieve list of ROI mask files from directory."""
    import os, glob
    roi_files = sorted(glob.glob(os.path.join(roi_dir, '*.nii.gz')))
    if not roi_files:
        raise ValueError(f"No ROI files found in {roi_dir}")
    return roi_files


def extract_roi_values(cope_file, roi_mask, baseline_file=None, output_dir=None):
    """Extract mean beta values (and PSC if baseline provided) for a single ROI across subjects."""
    from nipype.interfaces.fsl import ImageStats
    import os
    import numpy as np

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Compute mean beta values within ROI for each subject
    stats = ImageStats(in_file=cope_file, op_string='-k %s -m', mask_file=roi_mask)
    result = stats.run()
    beta_values = np.array(result.outputs.out_stat)  # Mean beta per subject

    # If baseline_file is provided, compute PSC
    if baseline_file:
        baseline_stats = ImageStats(in_file=baseline_file, op_string='-k %s -m', mask_file=roi_mask)
        baseline_result = baseline_stats.run()
        baseline_values = np.array(baseline_result.outputs.out_stat)
        # Ensure baseline matches cope_file in length
        if len(baseline_values) != len(beta_values):
            raise ValueError("Baseline file subject count does not match cope file")
        # Compute PSC: (beta / baseline) * 100
        psc_values = (beta_values / baseline_values) * 100
    else:
        psc_values = None  # No PSC without baseline

    # Save to text files
    roi_name = os.path.basename(roi_mask).replace('.nii.gz', '')
    beta_file = os.path.join(output_dir, f'beta_{roi_name}.txt')
    np.savetxt(beta_file, beta_values, fmt='%.6f')

    if psc_values is not None:
        psc_file = os.path.join(output_dir, f'psc_{roi_name}.txt')
        np.savetxt(psc_file, psc_values, fmt='%.6f')
        return beta_file, psc_file
    return beta_file, None


def combine_roi_values(beta_files, psc_files, output_dir):
    """Combine beta and PSC values from all ROIs into CSV files."""
    import pandas as pd
    import os

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Combine beta values
    beta_data = {}
    for beta_file in beta_files:
        roi_name = os.path.basename(beta_file).replace('beta_', '').replace('.txt', '')
        beta_values = np.loadtxt(beta_file)
        beta_data[roi_name] = beta_values
    beta_df = pd.DataFrame(beta_data)
    beta_df.index.name = 'subject'
    beta_csv = os.path.join(output_dir, 'beta_all_rois.csv')
    beta_df.to_csv(beta_csv)

    # Combine PSC values if available
    if psc_files and all(f is not None for f in psc_files):
        psc_data = {}
        for psc_file in psc_files:
            roi_name = os.path.basename(psc_file).replace('psc_', '').replace('.txt', '')
            psc_values = np.loadtxt(psc_file)
            psc_data[roi_name] = psc_values
        psc_df = pd.DataFrame(psc_data)
        psc_df.index.name = 'subject'
        psc_csv = os.path.join(output_dir, 'psc_all_rois.csv')
        psc_df.to_csv(psc_csv)
        return beta_csv, psc_csv
    return beta_csv, None# =============================================================================
# UNIFIED GROUP-LEVEL ANALYSIS WORKFLOW
# =============================================================================

def create_flexible_design_matrix(subjects, group_coding='1/0', contrast_type='standard'):
    """
    Create flexible design matrix for group-level analysis.
    
    Args:
        subjects (list): List of subject IDs
        group_coding (str): '1/0' for dummy coding (1=patients, 0=controls) 
                           or '1/-1' for effect coding (1=patients, -1=controls)
        contrast_type (str): 'standard' for all contrasts, 'minimal' for basic contrasts
    
    Returns:
        tuple: (design_matrix, list_of_contrasts)
    """
    n_subjects = len(subjects)
    
    # Determine group coding
    if group_coding == '1/0':
        # Dummy coding: 1 for patients, 0 for controls
        group_indicator = np.array([1 if sub.startswith('1') else 0 for sub in subjects])
    elif group_coding == '1/-1':
        # Effect coding: 1 for patients, -1 for controls
        group_indicator = np.array([1 if sub.startswith('1') else -1 for sub in subjects])
    else:
        raise ValueError(f"Unknown group_coding: {group_coding}. Use '1/0' or '1/-1'")
    
    # Create design matrix: [intercept, group_indicator]
    design = np.column_stack([np.ones(n_subjects), group_indicator])
    
    # Create contrasts based on type
    if contrast_type == 'minimal':
        if group_coding == '1/0':
            contrasts = [
                np.array([0, 1]),   # patients > controls
                np.array([0, -1]),  # patients < controls
            ]
        else:  # 1/-1 coding
            contrasts = [
                np.array([0, 1]),   # patients > controls
                np.array([0, -1]),  # patients < controls
            ]
    else:  # standard - all contrasts
        if group_coding == '1/0':
            contrasts = [
                np.array([0, 1]),   # patients > controls
                np.array([0, -1]),  # patients < controls
                np.array([1, 1]),   # mean effect in patients
                np.array([1, 0]),   # mean effect in controls
            ]
        else:  # 1/-1 coding
            contrasts = [
                np.array([0, 1]),   # patients > controls
                np.array([0, -1]),  # patients < controls
                np.array([1, 1]),   # mean effect in patients
                np.array([1, -1]),  # mean effect in controls
            ]
    
    return design, contrasts

def save_vest_file(data, filename):
    """Save data in VEST format for FSL."""
    with open(filename, 'w') as f:
        f.write(f"/NumWaves\t{data.shape[1]}\n")
        f.write(f"/NumPoints\t{data.shape[0]}\n")
        f.write("/Matrix\n")
        for row in data:
            f.write("\t".join([str(val) for val in row]) + "\n")



# =============================================================================
# UTILITY FUNCTIONS FOR GROUP ANALYSIS
# =============================================================================

def create_two_group_analysis(subjects, output_dir, method='flameo', 
                            group_coding='1/0', **kwargs):
    """
    Create a two-group analysis (e.g., patients vs controls).
    
    Args:
        subjects (list): List of subject IDs
        output_dir (str): Output directory
        method (str): Analysis method
        group_coding (str): Group coding scheme
        **kwargs: Additional arguments
    
    Returns:
        tuple: (workflow, design_file, con_file)
    """
    return create_group_analysis_workflow(
        output_dir, method, subjects, group_coding, 'standard', **kwargs
    )

def extract_subject_ids_from_files(file_paths):
    """
    Extract subject IDs from file paths.
    
    Args:
        file_paths (list): List of file paths
    
    Returns:
        list: List of subject IDs
    """
    subjects = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.startswith('sub-'):
            subject_id = filename.split('_')[0].replace('sub-', '')
            subjects.append(subject_id)
    return sorted(list(set(subjects)))

def validate_group_analysis_inputs(cope_files, var_cope_files=None, 
                                 subjects=None, method='flameo'):
    """
    Validate inputs for group analysis.
    
    Args:
        cope_files (list): List of cope files
        var_cope_files (list): List of var_cope files
        subjects (list): List of subject IDs
        method (str): Analysis method
    
    Returns:
        bool: True if valid
    """
    if not cope_files:
        raise ValueError("No cope files provided")
    
    if method == 'flameo' and not var_cope_files:
        raise ValueError("var_cope_files required for FLAMEO method")
    
    if method == 'flameo' and len(cope_files) != len(var_cope_files):
        raise ValueError("Number of cope and var_cope files must match")
    
    if subjects and len(subjects) != len(cope_files):
        raise ValueError("Number of subjects must match number of cope files")
    
    return True

def get_group_summary(subjects):
    """
    Get summary of group composition.
    
    Args:
        subjects (list): List of subject IDs
    
    Returns:
        dict: Group summary
    """
    patients = [s for s in subjects if s.startswith('1')]
    controls = [s for s in subjects if s.startswith('2')]
    
    return {
        'total_subjects': len(subjects),
        'patients': len(patients),
        'controls': len(controls),
        'patient_ids': patients,
        'control_ids': controls
    }

# =============================================================================
# WORKFLOW SUMMARY AND DOCUMENTATION
# =============================================================================

def get_workflow_summary():
    """
    Summary of all available group-level workflows.
    
    WORKFLOW TYPES:
    
    1. STANDARD GROUP-LEVEL WORKFLOWS:
       - wf_data_prepare(): Data preparation and merging
       - wf_roi_extract(): ROI value extraction (cope files → beta/PSC)
       - wf_flameo(): FLAMEO workflow with optional clustering
       - wf_randomise(): Randomise workflow with TFCE
    
    2. ROI-BASED WORKFLOWS:
       - wf_roi_psc_analysis(): ROI-based analysis using PSC (converts cope to PSC first)
       - wf_roi_extract(): ROI value extraction (cope files → beta/PSC)
    
    3. UNIFIED ANALYSIS WORKFLOWS:
       - create_group_analysis_workflow(): Unified function for whole-brain or ROI analysis
       - run_group_analysis(): One-stop analysis function with automatic setup
    
    4. CONVENIENCE FUNCTIONS:
       - create_two_group_analysis(): Two-group comparison setup
    
    DATA TYPES:
    - All workflows use cope files as primary input
    - FLAMEO workflows require var_cope files
    - Randomise workflows only need cope files
    - ROI workflows extract beta values and PSC from cope files
    
    ROI WORKFLOW COMPARISON:
    
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │ Workflow        │ Purpose         │ Output          │ When to Use     │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ create_group_   │ Statistical     │ Statistical     │ Formal testing, │
    │ analysis_       │ testing         │ maps (z-stats,  │ group compari-  │
    │ workflow(roi)   │                 │ p-values)       │ sons, publica-  │
    │                 │                 │                 │ tion results    │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ wf_roi_psc_     │ PSC-based       │ Statistical     │ Clinical inter- │
    │ analysis        │ testing         │ maps (PSC %)    │ pretation,      │
    │                 │                 │                 │ cross-study     │
    │                 │                 │                 │ comparisons     │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ wf_roi_extract  │ Value           │ CSV files with  │ Descriptive     │
    │                 │ extraction      │ ROI values      │ analysis, data  │
    │                 │                 │ (beta & PSC)    │ export, explo-  │
    │                 │                 │                 │ ratory analysis │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    """
    workflows = {
        'standard': {
            'wf_data_prepare': 'Data preparation and merging',
            'wf_roi_extract': 'ROI value extraction (cope files → beta/PSC)',
            'wf_flameo': 'FLAMEO workflow with optional clustering',
            'wf_randomise': 'Randomise workflow with TFCE'
        },
        'roi': {
            'wf_roi_psc_analysis': 'ROI-based analysis using PSC (converts cope to PSC first)',
            'wf_roi_extract': 'ROI value extraction (cope files → beta/PSC)'
        },
        'unified': {
            'create_group_analysis_workflow': 'Unified function for whole-brain or ROI analysis',
            'run_group_analysis': 'One-stop analysis function with automatic setup'
        },
        'convenience': {
            'create_two_group_analysis': 'Two-group comparison setup'
        }
    }
    return workflows

def get_workflow_usage_examples():
    """
    Usage examples for different workflow types.
    
    Returns:
        dict: Dictionary with usage examples
    """
    examples = {
        'basic_wholeBrain': '''
# Basic whole-brain analysis with FLAMEO (standalone)
from group_level_workflows import wf_flameo
wf = wf_flameo(output_dir='output')
wf.inputs.inputnode.cope_files = cope_files
wf.inputs.inputnode.var_cope_files = var_cope_files
wf.inputs.inputnode.mask_file = mask_file
wf.inputs.inputnode.design_file = design_file
wf.inputs.inputnode.con_file = con_file
wf.run()

# Or using the unified function
from group_level_workflows import create_group_analysis_workflow
wf, design_file, con_file = create_group_analysis_workflow(
    output_dir='output', method='flameo', analysis_type='whole_brain'
)
        ''',
        
        'wholeBrain_randomise': '''
# Whole-brain analysis with Randomise (standalone)
from group_level_workflows import wf_randomise
wf = wf_randomise(output_dir='output')
wf.inputs.inputnode.cope_files = cope_files
wf.inputs.inputnode.mask_file = mask_file
wf.inputs.inputnode.design_file = design_file
wf.inputs.inputnode.con_file = con_file
wf.run()

# Or using the unified function
from group_level_workflows import create_group_analysis_workflow
wf, design_file, con_file = create_group_analysis_workflow(
    output_dir='output', method='randomise', analysis_type='whole_brain'
)
        ''',
        
        'roi_analysis': '''
# ROI-based analysis using the unified function
from group_level_workflows import create_group_analysis_workflow

# ROI analysis with FLAMEO
wf, design_file, con_file = create_group_analysis_workflow(
    output_dir='output', method='flameo', analysis_type='roi', roi_dir=roi_directory
)
wf.inputs.inputnode.cope_files = cope_files
wf.inputs.inputnode.var_cope_files = var_cope_files

# ROI analysis with Randomise
wf, design_file, con_file = create_group_analysis_workflow(
    output_dir='output', method='randomise', analysis_type='roi', roi_dir=roi_directory
)
wf.inputs.inputnode.cope_files = cope_files

# ROI-based analysis with PSC (converts cope to PSC first)
from group_level_workflows import wf_roi_psc_analysis
wf = wf_roi_psc_analysis(output_dir='output', method='flameo')
wf.inputs.inputnode.cope_files = cope_files
wf.inputs.inputnode.baseline_cope_file = baseline_cope_file
wf.inputs.inputnode.var_cope_files = var_cope_files
wf.inputs.inputnode.roi = roi_directory
        ''',
        
        'roi_extraction': '''
# ROI value extraction
from group_level_workflows import wf_roi_extract
wf = wf_roi_extract(output_dir='output')
wf.inputs.inputnode.cope_file = cope_file
wf.inputs.inputnode.baseline_file = baseline_file
        ''',
        
        'complete_analysis': '''
# Complete analysis with automatic setup
from group_level_workflows import run_group_analysis
wf = run_group_analysis(
    cope_files=cope_files,
    var_cope_files=var_cope_files,
    mask_file=mask_file,
    subjects=subjects,
    output_dir='output',
    method='flameo',
    group_coding='1/0'
)
wf.run()
        '''
    }
    return examples


# =============================================================================
# ROI PSC ANALYSIS WORKFLOW
# =============================================================================

def wf_roi_psc_analysis(output_dir, name="wf_roi_psc_analysis", method='flameo', baseline_condition='baseline'):
    """
    Workflow for ROI-based analysis using PSC (Percent Signal Change).
    
    This workflow converts cope files to PSC before performing group-level analysis.
    It's useful when you want to analyze percent signal change rather than raw contrast values.
    
    WHEN TO USE THIS WORKFLOW:
    
    1. PSC-BASED STATISTICAL ANALYSIS:
       - When you want to perform statistical testing on percent signal change
       - For analyzing relative changes from baseline
       - When you have a well-defined baseline condition
       - For clinical interpretation of results
    
    2. CLINICAL INTERPRETATION:
       - When you need results in interpretable units (%)
       - For communicating with clinicians and patients
       - When you want to compare effect sizes across studies
       - For patient-specific analysis and diagnosis
    
    3. CROSS-STUDY COMPARISONS:
       - When comparing results across different studies
       - When you want normalized effect sizes
       - For meta-analysis preparation
       - When baseline conditions differ between studies
    
    4. BASELINE-REFERENCED ANALYSIS:
       - When you want to analyze changes relative to baseline
       - For resting-state vs task comparisons
       - When baseline is a meaningful reference point
       - For longitudinal studies with baseline measurements
    
    5. PERCENT CHANGE INTERPRETATION:
       - When you prefer percent change over raw contrast values
       - For educational and training purposes
       - When presenting results to non-experts
       - For grant applications and reports
    
    ADVANTAGES OF PSC ANALYSIS:
    - More intuitive interpretation (% change)
    - Comparable across different studies
    - Easier to communicate to non-experts
    - Normalized relative to baseline
    
    DISADVANTAGES OF PSC ANALYSIS:
    - Requires baseline condition
    - May amplify noise in low-signal regions
    - Division by zero issues in baseline regions
    - Less standard in neuroimaging literature
    
    Args:
        output_dir (str): Output directory
        name (str): Workflow name
        method (str): 'flameo' for parametric analysis or 'randomise' for non-parametric
        baseline_condition (str): Name of baseline condition for PSC calculation
    """
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node for PSC-based ROI workflow
    if method == 'flameo':
        inputnode = Node(IdentityInterface(fields=['roi', 'cope_files', 'baseline_cope_file', 'var_cope_files',
                                                   'design_file', 'grp_file', 'con_file', 'result_dir']),
                         name='inputnode')
    else:  # randomise
        inputnode = Node(IdentityInterface(fields=['roi', 'cope_files', 'baseline_cope_file',
                                                   'design_file', 'con_file']),
                         name='inputnode')

    # ROI node to fetch ROI files
    roi_node = Node(Function(input_names=['roi'], output_names=['roi_files'],
                             function=get_roi_files),
                    name='roi_node')

    # Convert cope to PSC for each ROI
    cope_to_psc = MapNode(Function(input_names=['cope_file', 'baseline_cope_file', 'roi_mask'],
                                   output_names=['psc_file'],
                                   function=convert_cope_to_psc),
                          iterfield=['cope_file', 'roi_mask'],
                          name='cope_to_psc')

    # Analysis node for each ROI
    if method == 'flameo':
        analysis_node = MapNode(FLAMEO(run_mode='flame1'),
                               iterfield=['cope_file', 'var_cope_file', 'mask_file'],
                               name='flameo')
    else:  # randomise
        analysis_node = MapNode(Randomise(num_perm=10000, tfce=True, vox_p_values=True),
                               iterfield=['in_file', 'mask', 'design_file', 'tcon'],
                               name='randomise')

    # Output node (method-dependent)
    if method == 'flameo':
        outputnode = Node(IdentityInterface(fields=['zstats', 'fdr_thresh', 'fwe_thresh']),
                          name='outputnode')
    else:  # randomise
        outputnode = Node(IdentityInterface(fields=['tstat_files', 'tfce_corr_p_files', 'z_thresh_files']),
                          name='outputnode')

    # DataSink for ROI analysis outputs
    datasink = Node(DataSink(base_directory=output_dir), name='datasink')

    # Workflow connections (method-dependent)
    if method == 'flameo':
        wf.connect([
            (inputnode, roi_node, [('roi', 'roi')]),
            (inputnode, cope_to_psc, [('cope_files', 'cope_file'),
                                      ('baseline_cope_file', 'baseline_cope_file')]),
            (roi_node, cope_to_psc, [('roi_files', 'roi_mask')]),
            (roi_node, analysis_node, [('roi_files', 'mask_file')]),
            (cope_to_psc, analysis_node, [('psc_file', 'cope_file')]),
            (inputnode, analysis_node, [('var_cope_files', 'var_cope_file'),
                                       ('design_file', 'design_file'),
                                       ('grp_file', 'cov_split_file'),
                                       ('con_file', 't_con_file')]),
            (analysis_node, outputnode, [('zstats', 'zstats')]),
            (roi_node, outputnode, [('roi_files', 'roi_files')]),
            (outputnode, datasink, [('zstats', 'zstats'),
                                    ('roi_files', 'roi_files')])
        ])
    else:  # randomise
        wf.connect([
            (inputnode, roi_node, [('roi', 'roi')]),
            (inputnode, cope_to_psc, [('cope_files', 'cope_file'),
                                      ('baseline_cope_file', 'baseline_cope_file')]),
            (roi_node, cope_to_psc, [('roi_files', 'roi_mask')]),
            (roi_node, analysis_node, [('roi_files', 'mask')]),
            (cope_to_psc, analysis_node, [('psc_file', 'in_file')]),
            (inputnode, analysis_node, [('design_file', 'design_file'),
                                       ('con_file', 'tcon')]),
            (analysis_node, outputnode, [('tstat_files', 'tstat_files'),
                                        ('t_corrected_p_files', 'tfce_corr_p_files')]),
            (roi_node, outputnode, [('roi_files', 'roi_files')]),
            (outputnode, datasink, [('tstat_files', 'tstats'),
                                    ('tfce_corr_p_files', 'tfce_p'),
                                    ('roi_files', 'roi_files')])
        ])

    return wf

def convert_cope_to_psc(cope_file, baseline_cope_file, roi_mask):
    """
    Convert cope file to PSC using baseline condition.
    
    Args:
        cope_file (str): Path to cope file
        baseline_cope_file (str): Path to baseline cope file
        roi_mask (str): Path to ROI mask
    
    Returns:
        str: Path to PSC file
    """
    import os
    import numpy as np
    import nibabel as nib
    from nipype.interfaces.fsl import ImageMaths
    
    # Load images
    cope_img = nib.load(cope_file)
    baseline_img = nib.load(baseline_cope_file)
    mask_img = nib.load(roi_mask)
    
    # Get data
    cope_data = cope_img.get_fdata()
    baseline_data = baseline_img.get_fdata()
    mask_data = mask_img.get_fdata()
    
    # Apply mask
    cope_masked = cope_data * mask_data
    baseline_masked = baseline_data * mask_data
    
    # Calculate PSC: (cope - baseline) / baseline * 100
    # Add small constant to avoid division by zero
    epsilon = 1e-6
    psc_data = np.where(baseline_masked != 0,
                        (cope_masked - baseline_masked) / (np.abs(baseline_masked) + epsilon) * 100,
                        0)
    
    # Create new image
    psc_img = nib.Nifti1Image(psc_data, cope_img.affine, cope_img.header)
    
    # Save PSC file
    psc_file = cope_file.replace('.nii.gz', '_psc.nii.gz')
    nib.save(psc_img, psc_file)
    
    return psc_file

