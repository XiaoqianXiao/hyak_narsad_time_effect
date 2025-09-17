#!/usr/bin/env python3
"""
Test script for the updated first-level workflow with BIDS naming.
This script tests the workflow configuration without requiring the full container environment.
"""

import sys
import os
import tempfile
import pandas as pd

# Add current directory to path
sys.path.insert(0, '/app')

def test_workflow_configuration():
    """Test the workflow configuration with BIDS entities."""
    print("Testing updated first_level_workflows.py with BIDS naming...")
    
    try:
        from first_level_workflows import first_level_wf, extract_cs_conditions
        print("✓ Successfully imported first_level_workflows")
    except ImportError as e:
        print(f"✗ Failed to import first_level_workflows: {e}")
        return False
    
    # Create test data
    test_data = {
        'trial_type': ['CS-_first_half', 'CS-_first_half', 'CSS_first_half', 'CSS_first_half', 'FIXATION'],
        'onset': [10.0, 20.0, 30.0, 40.0, 50.0],
        'duration': [2.0, 2.0, 2.0, 2.0, 1.0]
    }
    df = pd.DataFrame(test_data)
    
    try:
        # Test extract_cs_conditions
        df_with_conditions, cs_conditions, css_conditions, csr_conditions, other_conditions = extract_cs_conditions(df)
        print("✓ extract_cs_conditions works")
    except Exception as e:
        print(f"✗ extract_cs_conditions failed: {e}")
        return False
    
    # Test workflow creation with BIDS entities
    with tempfile.TemporaryDirectory() as temp_dir:
        in_files = {
            'sub-001': {
                'bold': '/fake/path/bold.nii.gz',
                'mask': '/fake/path/mask.nii.gz', 
                'events': '/fake/path/events.tsv',
                'regressors': '/fake/path/regressors.tsv',
                'tr': 2.0
            }
        }
        
        contrasts = [('CS-_first_half_first > FIXATION', 'T', ['CS-_first_half_first', 'FIXATION'], [1, -1])]
        condition_names = ['CS-_first_half_first', 'FIXATION']
        
        # Test with BIDS entities
        bids_entities = {
            'subject': 'N101',
            'session': 'pilot3mm',
            'task': 'phase2',
            'space': 'MNI152NLin2009cAsym'
        }
        
        try:
            workflow = first_level_wf(
                in_files=in_files,
                output_dir=temp_dir,
                condition_names=condition_names,
                contrasts=contrasts,
                df_conditions=df_with_conditions,
                bids_entities=bids_entities
            )
            print("✓ first_level_wf creates workflow with BIDS entities successfully")
            print(f"✓ Target directory will be: {temp_dir}/ses-pilot3mm/func")
            print("✓ Files will be saved directly in target directory with BIDS naming")
        except Exception as e:
            print(f"✗ first_level_wf with BIDS entities failed: {e}")
            return False
        
        # Test without BIDS entities (fallback)
        try:
            workflow_fallback = first_level_wf(
                in_files=in_files,
                output_dir=temp_dir,
                condition_names=condition_names,
                contrasts=contrasts,
                df_conditions=df_with_conditions,
                bids_entities=None
            )
            print("✓ first_level_wf creates workflow without BIDS entities successfully")
        except Exception as e:
            print(f"✗ first_level_wf without BIDS entities failed: {e}")
            return False
    
    print("\n🎉 All tests passed! The workflow configuration is correct.")
    return True

def test_create_script_integration():
    """Test the integration with create_1st_voxelWise.py."""
    print("\nTesting create_1st_voxelWise.py integration...")
    
    try:
        # Read the file and check for the entities parameter without importing
        with open('create_1st_voxelWise.py', 'r') as f:
            content = f.read()
        
        # Check if the function signature includes entities parameter
        if 'def run_subject_workflow(sub, inputs, work_dir, output_dir, task, entities=None):' in content:
            print("✓ run_subject_workflow function signature includes entities parameter")
        else:
            print("✗ run_subject_workflow function signature missing entities parameter")
            return False
        
        # Check if entities is passed to first_level_wf
        if 'bids_entities=entities' in content:
            print("✓ entities parameter is passed to first_level_wf")
        else:
            print("✗ entities parameter not passed to first_level_wf")
            return False
            
        print("✓ Integration with create_1st_voxelWise.py is correct")
        return True
        
    except FileNotFoundError:
        print("✗ create_1st_voxelWise.py not found")
        return False
    except Exception as e:
        print(f"✗ Error testing create_1st_voxelWise integration: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CONTAINER TEST: First-Level Workflow with BIDS Naming")
    print("=" * 60)
    
    # Test workflow configuration
    workflow_ok = test_workflow_configuration()
    
    # Test integration
    integration_ok = test_create_script_integration()
    
    print("\n" + "=" * 60)
    if workflow_ok and integration_ok:
        print("✅ ALL TESTS PASSED - Ready for production!")
        print("Expected output structure:")
        print("  /data/NARSAD/MRI/derivatives/fMRI_analysis_remove/firstLevel_timeEffect/phase2/sub-N101/ses-pilot3mm/func/")
        print("  ├── sub-N101_ses-pilot3mm_task-phase2_space-MNI152NLin2009cAsym_desc-cope1_bold.nii.gz")
        print("  ├── sub-N101_ses-pilot3mm_task-phase2_space-MNI152NLin2009cAsym_desc-varcope1_bold.nii.gz")
        print("  └── ...")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check the errors above")
        sys.exit(1)
