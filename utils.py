# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

import pandas as pd

def _get_tr(in_dict):
    return in_dict.get('RepetitionTime')


def _len(inlist):
    return len(inlist)


def _dof(inlist):
    return len(inlist) - 1


def _neg(val):
    return -val

def _dict_ds(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr']):
    return tuple([in_dict[sub][k] for k in order])

def _dict_ds_lss(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']):
    return tuple([in_dict[sub][k] for k in order])

def _bids2nipypeinfo_from_df(in_file, df_conditions, regressors_file,
                             regressors_names=None,
                             motion_columns=None,
                             decimals=3, amplitude=1.0):
    """
    Convert processed DataFrame from extract_cs_conditions() to FSL-compatible format.
    
    This function uses the pre-processed DataFrame that already has the 'conditions' column
    created by extract_cs_conditions(), ensuring consistent processing throughout the pipeline.
    
    Args:
        in_file (str): Path to the BOLD data file
        df_conditions (pandas.DataFrame): DataFrame with 'conditions' column from extract_cs_conditions()
        regressors_file (str): Path to the regressors file
        regressors_names (list): List of regressor names
        motion_columns (list): List of motion parameter column names
        decimals (int): Number of decimal places for rounding
        amplitude (float): Default amplitude value
    
    Returns:
        nipype.interfaces.base.support.Bunch: FSL-compatible session info
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Validate input DataFrame
    if not isinstance(df_conditions, pd.DataFrame):
        raise ValueError("df_conditions must be a pandas DataFrame")
    
    if 'conditions' not in df_conditions.columns:
        raise ValueError("DataFrame must have 'conditions' column from extract_cs_conditions()")
    
    required_columns = ['conditions', 'onset', 'duration']
    missing_columns = [col for col in required_columns if col not in df_conditions.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    print("=== DEBUG: Using processed DataFrame from extract_cs_conditions() ===")
    print(f"DataFrame shape: {df_conditions.shape}")
    print(f"DataFrame columns: {list(df_conditions.columns)}")
    print(f"Processed conditions: {sorted(df_conditions['conditions'].unique().tolist())}")

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    regress_data = read_csv_with_detection(regressors_file)
    
    # Handle motion columns gracefully
    try:
        # Convert motion columns to numeric, handling any string values
        motion_data = regress_data[motion_columns].apply(pd.to_numeric, errors='coerce')
        # Fill any NaN values with 0
        motion_data = motion_data.fillna(0)
        np.savetxt(out_motion, motion_data.values, '%g')
    except KeyError as e:
        print(f"Warning: Motion columns not found: {e}")
        # Create empty motion file
        np.savetxt(out_motion, np.zeros((len(regress_data), 6)), '%g')
        print("Created empty motion file")
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    # Get unique conditions from the processed DataFrame
    conditions = sorted(df_conditions['conditions'].unique().tolist())
    print(f"Using processed conditions: {conditions}")
    
    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        **{k: [] for k in bunch_fields})

    # Process each condition using the processed DataFrame
    for condition in runinfo.conditions:
        # Get all trials for this condition
        condition_trials = df_conditions[df_conditions['conditions'] == condition]
        
        if len(condition_trials) > 0:
            # Extract onsets, durations, and amplitudes
            onsets = condition_trials['onset'].values
            durations = condition_trials['duration'].values
            
            runinfo.onsets.append(np.round(onsets, 3).tolist())
            runinfo.durations.append(np.round(durations, 3).tolist())
            
            if 'amplitudes' in condition_trials.columns:
                amplitudes = condition_trials['amplitudes'].values
                runinfo.amplitudes.append(np.round(amplitudes, 3).tolist())
            else:
                runinfo.amplitudes.append([amplitude] * len(condition_trials))
                
            print(f"Condition '{condition}': {len(condition_trials)} trials at onsets {onsets.tolist()}")
        else:
            # Fallback if no trials found for this condition
            runinfo.onsets.append([])
            runinfo.durations.append([])
            runinfo.amplitudes.append([])
            print(f"Condition '{condition}': No trials found")

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        try:
            runinfo.regressors = regress_data[regressors_names]
        except KeyError:
            regressors_names = list(set(regressors_names).intersection(
                set(regress_data.columns)))
            runinfo.regressors = regress_data[regressors_names]
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)

def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Process the events file with automatic separator detection
    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    
    events = read_csv_with_detection(events_file)
    print("=== DEBUG: loaded event columns ===")
    print(events.columns.tolist())
    print(events.head())

    # Detect the condition column (try different possible names)
    condition_column = None
    possible_columns = ['trial_type', 'condition', 'event_type', 'type', 'stimulus', 'trial']
    for col in possible_columns:
        if col in events.columns:
            condition_column = col
            break
    
    if condition_column is None:
        # If no standard column found, try to use the first non-numeric column
        for col in events.columns:
            if not pd.api.types.is_numeric_dtype(events[col]):
                condition_column = col
                break
    
    if condition_column is None:
        raise ValueError(f"Could not find condition column in events file. Available columns: {events.columns.tolist()}")
    
    print(f"Using column '{condition_column}' for conditions")

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    regress_data = read_csv_with_detection(regressors_file)
    
    # Handle motion columns gracefully
    try:
        # Convert motion columns to numeric, handling any string values
        motion_data = regress_data[motion_columns].apply(pd.to_numeric, errors='coerce')
        # Fill any NaN values with 0
        motion_data = motion_data.fillna(0)
        np.savetxt(out_motion, motion_data.values, '%g')
    except KeyError as e:
        print(f"Warning: Motion columns not found: {e}")
        # Create empty motion file
        np.savetxt(out_motion, np.zeros((len(regress_data), 6)), '%g')
        print("Created empty motion file")
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    # Create conditions list with proper CS- splitting
    raw_conditions = list(events[condition_column].values)
    
    # Count CS- trials and create proper condition names
    cs_count = raw_conditions.count('CS-')
    if cs_count > 1:
        # Multiple CS- trials: split into CS-_first and CS-_others
        conditions = ['CS-_first', 'CS-_others']
        # Add other unique conditions (excluding CS-)
        other_conditions = [c for c in set(raw_conditions) if c != 'CS-']
        conditions.extend(other_conditions)
        print(f"Split {cs_count} CS- trials into CS-_first and CS-_others. Total conditions: {len(conditions)}")
    else:
        # Single or no CS- trials: use original logic
        conditions = list(set(raw_conditions))
        print(f"Using standard conditions: {len(conditions)} total")
    
    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        # Get all trials for this condition
        condition_trials = events[events[condition_column] == condition]
        
        if len(condition_trials) > 0:
            # Extract onsets, durations, and amplitudes
            onsets = condition_trials['onset'].values
            durations = condition_trials['duration'].values
            
            runinfo.onsets.append(np.round(onsets, 3).tolist())
            runinfo.durations.append(np.round(durations, 3).tolist())
            
            if 'amplitudes' in condition_trials.columns:
                amplitudes = condition_trials['amplitudes'].values
                runinfo.amplitudes.append(np.round(amplitudes, 3).tolist())
            else:
                runinfo.amplitudes.append([amplitude] * len(condition_trials))
                
            print(f"Condition '{condition}': {len(condition_trials)} trials at onsets {onsets.tolist()}")
        else:
            # Fallback if no trials found for this condition
            runinfo.onsets.append([])
            runinfo.durations.append([])
            runinfo.amplitudes.append([])
            print(f"Condition '{condition}': No trials found")

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        try:
            runinfo.regressors = regress_data[regressors_names]
        except KeyError:
            regressors_names = list(set(regressors_names).intersection(
                set(regress_data.columns)))
            runinfo.regressors = regress_data[regressors_names]
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)


def _bids2nipypeinfo_lss(in_file, events_file, regressors_file,
                          trial_ID,
                          regressors_names=None,
                          motion_columns=None,
                          decimals=3,
                          amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    # Load events and regressors with automatic separator detection
    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    
    events = read_csv_with_detection(events_file)
    print("LOADED EVENTS COLUMNS:", events.columns.tolist())
    print(events.head())
    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    regress_data = read_csv_with_detection(regressors_file)

    # Locate the trial of interest by ID
    trial = events[events['trial_ID'] == trial_ID]
    if trial.empty:
        raise ValueError(f"Trial ID {trial_ID} not found in events file.")
    if len(trial) > 1:
        raise ValueError(f"Trial ID {trial_ID} is not unique in events file.")

    other_trials = events[events['trial_ID'] != trial_ID]

    out_motion = Path('motion.par').resolve()
    # Convert motion columns to numeric, handling any string values
    motion_data = regress_data[motion_columns].apply(pd.to_numeric, errors='coerce')
    # Fill any NaN values with 0
    motion_data = motion_data.fillna(0)
    np.savetxt(out_motion, motion_data.values, '%g')

    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    # Build the subject_info Bunch
    conditions = ['trial', 'others']
    onsets = [
        np.round(trial['onset'].values.tolist(), decimals),
        np.round(other_trials['onset'].values.tolist(), decimals)
    ]
    durations = [
        np.round(trial['duration'].values.tolist(), decimals),
        np.round(other_trials['duration'].values.tolist(), decimals)
    ]
    amplitudes = [
        [amplitude] * len(onsets[0]),
        [amplitude] * len(onsets[1])
    ]

    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        onsets=onsets,
        durations=durations,
        amplitudes=amplitudes
    )

    if regressors_names:
        runinfo.regressor_names = regressors_names
        regress_subset = regress_data[regressors_names].fillna(0.0)
        runinfo.regressors = regress_subset.values.T.tolist()

    return [runinfo], str(out_motion)


def print_input_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().inputs
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")

    # 2) Capture help() output to find the "Mutually exclusive inputs" line
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        interface_class().help()
    finally:
        sys.stdout = old_stdout

    help_text = buf.getvalue().splitlines()
    mux_line = next((line for line in help_text if 'mutually_exclusive' in line), None)

    # 3) Parse and print mutually‐exclusive inputs if present
    if mux_line:
        # e.g. "Mutually exclusive inputs: subject_info, event_files, bids_event_file"
        _, fields = mux_line.split(':', 1)
        names = [n.strip() for n in fields.split(',')]
        print("\nMutually exclusive inputs:")
        for n in names:
            print(f"  - {n}")
    else:
        print("\nNo mutually exclusive inputs found in help().")


def print_output_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().output_spec()  # same as inst.output_spec(), but bound
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")


def detect_csv_separator(file_path, sample_size=1024):
    """
    Automatically detect the separator used in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        sample_size (int): Number of characters to read for detection
    
    Returns:
        str: Detected separator ('\t' for tab, ',' for comma)
    """
    try:
        with open(file_path, 'r') as f:
            sample = f.read(sample_size)
        
        # Count occurrences of potential separators
        comma_count = sample.count(',')
        tab_count = sample.count('\t')
        
        # Determine the most likely separator
        if tab_count > comma_count:
            return '\t'
        else:
            return ','
    except Exception as e:
        # Default to comma if detection fails
        return ','

def read_csv_with_detection(file_path, **kwargs):
    """
    Read a CSV file with automatic separator detection.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        pandas.DataFrame: Loaded CSV data
    """
    try:
        # First try with comma separator (most common)
        df = pd.read_csv(file_path, sep=',', **kwargs)
        
        # If we got multiple columns, we're good
        if len(df.columns) > 1:
            # Convert numeric columns for comma-separated data too
            for col in df.columns:
                if col in ['onset', 'duration', 'trial_ID']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        
        # If we got a single column, check if it contains tab-separated values
        if len(df.columns) == 1:
            column_name = df.columns[0]
            
            # Check if it's tab-separated data in a single column
            if '\t' in str(column_name) or (len(df) > 0 and '\t' in str(df.iloc[0, 0])):
                # This is tab-separated data in a single column
                if '\t' in str(column_name):
                    # Header contains tabs - split it
                    header_row = str(column_name).replace('"', '').split('\t')
                    # Split all data rows
                    df = df[column_name].astype(str).str.replace('"', '').str.split('\t', expand=True)
                    df.columns = header_row
                else:
                    # Data contains tabs - use first row as header
                    df = df[column_name].astype(str).str.replace('"', '').str.split('\t', expand=True)
                    df.columns = df.iloc[0]
                    df = df.drop(df.index[0]).reset_index(drop=True)
                
                # Convert numeric columns
                for col in df.columns:
                    if col in ['onset', 'duration', 'trial_ID']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Also convert any column that looks numeric
                    elif col not in ['trial_type', 'condition', 'event_type', 'type', 'stimulus', 'trial']:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                
                return df
        
        # If comma didn't work, try tab separator
        df = pd.read_csv(file_path, sep='\t', **kwargs)
        
        # If we got multiple columns, we're good
        if len(df.columns) > 1:
            # Convert numeric columns for tab-separated data too
            for col in df.columns:
                if col in ['onset', 'duration', 'trial_ID']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        
        # If still single column, try other separators
        for sep in [';', '|', ' ']:
            try:
                df = pd.read_csv(file_path, sep=sep, **kwargs)
                if len(df.columns) > 1:
                    # Convert numeric columns for other separators too
                    for col in df.columns:
                        if col in ['onset', 'duration', 'trial_ID']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
            except:
                continue
        
        # If nothing worked, return the original
        return df
        
    except Exception as e:
        # Fallback to automatic separator detection
        separator = detect_csv_separator(file_path)
        df = pd.read_csv(file_path, sep=separator, **kwargs)
        # Convert numeric columns for fallback case too
        for col in df.columns:
            if col in ['onset', 'duration', 'trial_ID']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df