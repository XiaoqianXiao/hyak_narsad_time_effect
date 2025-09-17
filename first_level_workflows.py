#Nipype v1.10.0.
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _dict_ds_lss
from utils import _bids2nipypeinfo
from utils import _bids2nipypeinfo_from_df
from utils import _bids2nipypeinfo_lss
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design, FEATModel
import logging

# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class DerivativesDataSink(BIDSDerivatives):
    """Custom data sink for first-level analysis outputs."""
    out_path_base = 'firstLevel_timeEffect'

DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']
DATA_ITEMS_LSS = ['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']

def extract_cs_conditions(df_trial_info):
    """
    Extract and group CS-, CSS, and CSR conditions from a pandas DataFrame.
    
    This function adds a 'conditions' column to the DataFrame that groups trials:
    - First trial of each CS type becomes 'CS-_first', 'CSS_first', 'CSR_first'
    - Remaining trials of each type become 'CS-_others', 'CSS_others', 'CSR_others'
    - All other trials keep their original trial_type as conditions value
    
    Args:
        df_trial_info (pandas.DataFrame): DataFrame with columns 'trial_type', 'onset', 'duration'.
                                        The 'trial_type' column contains condition names,
                                        and 'onset' column is used for chronological sorting.
    
    Returns:
        tuple: (df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions)
            - df_with_conditions: DataFrame with added 'conditions' column
            - cs_conditions: dict with 'first' and 'other' keys for CS- conditions
            - css_conditions: dict with 'first' and 'other' keys for CSS conditions  
            - csr_conditions: dict with 'first' and 'other' keys for CSR conditions
            - other_conditions: List of non-CS/CSS/CSR conditions
    """
    import pandas as pd
    
    # Validate DataFrame input
    if not isinstance(df_trial_info, pd.DataFrame):
        raise ValueError("df_trial_info must be a pandas DataFrame")
    
    if df_trial_info.empty:
        raise ValueError("DataFrame cannot be empty")
    
    required_columns = ['trial_type', 'onset']
    missing_columns = [col for col in required_columns if col not in df_trial_info.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying original
    df_work = df_trial_info.copy()
    
    # Initialize conditions column with trial_type values
    df_work['conditions'] = df_work['trial_type'].copy()
    
    logger.info(f"Using DataFrame input with {len(df_work)} trials")
    logger.info(f"DataFrame columns: {list(df_work.columns)}")
    
    return df_work


# =============================================================================
# CORE WORKFLOW FUNCTIONS
# =============================================================================

def first_level_wf(in_files, output_dir, condition_names=None, contrasts=None, 
                   fwhm=6.0, brightness_threshold=1000, high_pass_cutoff=100,
                   use_smoothing=True, use_derivatives=True, model_serial_correlations=True,
                   df_conditions=None):
    """
    Generic first-level workflow for fMRI analysis.
    
    Args:
        in_files (dict): Input files dictionary
        output_dir (str): Output directory path
        condition_names (list): List of condition names (auto-detected if None)
        contrasts (list): List of contrast tuples (auto-generated if None)
        fwhm (float): Smoothing FWHM
        brightness_threshold (float): SUSAN brightness threshold
        high_pass_cutoff (float): High-pass filter cutoff
        use_smoothing (bool): Whether to apply smoothing
        use_derivatives (bool): Whether to use temporal derivatives
        model_serial_correlations (bool): Whether to model serial correlations
    
    Returns:
        pe.Workflow: Configured first-level workflow
    """
    if not in_files:
        raise ValueError("in_files cannot be empty")
    
    workflow = pe.Workflow(name='wf_1st_level')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    # Data source
    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    # Use processed DataFrame if provided, otherwise use original events file
    if df_conditions is not None:
        runinfo = pe.Node(niu.Function(
            input_names=['in_file', 'df_conditions', 'regressors_file', 'regressors_names'],
            function=_bids2nipypeinfo_from_df, output_names=['info', 'realign_file']),
            name='runinfo')
    else:
        runinfo = pe.Node(niu.Function(
            input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
            function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
            name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')
    
    # Optional smoothing
    if use_smoothing:
        susan = pe.Node(SUSAN(), name='susan')
        susan.inputs.fwhm = fwhm
        susan.inputs.brightness_threshold = brightness_threshold
        preproc_output = susan
    else:
        preproc_output = apply_mask

    # Model specification
    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=high_pass_cutoff
    ), name='l1_spec')


    # Level 1 model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': use_derivatives}},
        model_serial_correlations=model_serial_correlations,
        contrasts=contrasts
    ), name='l1_model')

    # FEAT model specification
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    
    # FEAT fitting
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    
    # Select output files
    n_contrasts = len(contrasts)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, n_contrasts + 1)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, n_contrasts + 1)}
    }), name='feat_select')

    # Data sinks for copes and varcopes
    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'cope{i}'),
            name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'varcope{i}'),
            name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, n_contrasts + 1)
    ]

    # Build workflow connections
    connections = _build_workflow_connections(
        datasource, apply_mask, runinfo, l1_spec, l1_model, 
        feat_spec, feat_fit, feat_select, preproc_output, use_smoothing, df_conditions
    )
    
    # Add data sink connections
    for i in range(1, n_contrasts + 1):
        connections.extend([
            (datasource, ds_copes[i - 1], [('bold', 'source_file')]),
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')]),
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')]),
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
        ])

    workflow.connect(connections)
    return workflow

def _build_workflow_connections(datasource, apply_mask, runinfo, l1_spec, l1_model, 
                              feat_spec, feat_fit, feat_select, preproc_output, use_smoothing, df_conditions=None):
    """
    Build workflow connections based on smoothing configuration.
    
    Args:
        datasource: Data source node
        apply_mask: Mask application node
        runinfo: Run info node
        l1_spec: Level 1 specification node
        l1_model: Level 1 model node
        feat_spec: FEAT specification node
        feat_fit: FEAT fitting node
        feat_select: FEAT selection node
        preproc_output: Preprocessing output node
        use_smoothing: Whether smoothing is used
    
    Returns:
        list: List of workflow connections
    """
    connections = [
        (datasource, apply_mask, [('bold', 'in_file'), ('mask', 'mask_file')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (feat_spec, feat_fit, [('design_file', 'design_file'), ('con_file', 'tcon_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
    ]
    
    # Add runinfo connections based on whether df_conditions is provided
    if df_conditions is not None:
        # Use processed DataFrame
        connections.append((datasource, runinfo, [('regressors', 'regressors_file')]))
        # Add df_conditions as a static input
        runinfo.inputs.df_conditions = df_conditions
    else:
        # Use original events file
        connections.append((datasource, runinfo, [('events', 'events_file'), ('regressors', 'regressors_file')]))
    
    # Add smoothing connections if used
    if use_smoothing:
        connections.extend([
            (apply_mask, preproc_output, [('out_file', 'in_file')]),
            (preproc_output, l1_spec, [('smoothed_file', 'functional_runs')]),
            (preproc_output, runinfo, [('smoothed_file', 'in_file')]),
            (preproc_output, feat_fit, [('smoothed_file', 'in_file')])
        ])
    else:
        connections.extend([
            (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
            (apply_mask, runinfo, [('out_file', 'in_file')]),
            (apply_mask, feat_fit, [('out_file', 'in_file')])
        ])
    
    # Add runinfo connections
    connections.extend([
        (runinfo, l1_spec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')])
    ])
    
    return connections