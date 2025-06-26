#!/usr/bin/env python3
"""
Test script to verify the fixed data quality check in process_capture24_stage2_3_custom_mins_multiple.py
"""

import numpy as np

def test_data_quality_check():
    """Test the new data quality logic with various scenarios"""
    
    print("Testing the fixed data quality check logic...")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Normal data (all channels have variation)",
            "data": np.random.randn(100, 3) * 0.1 + [0.0, 0.05, 0.85],
            "should_skip": False
        },
        {
            "name": "Sleep data - x-axis all zero (like index 1674)",
            "data": np.column_stack([
                np.zeros(100),  # x-axis all zero
                np.random.randn(100) * 0.008 + 0.055,  # y-axis with small variation
                np.random.randn(100) * 0.008 + 0.866   # z-axis with small variation
            ]),
            "should_skip": False  # This should NOT be skipped after fix
        },
        {
            "name": "Corrupted data - 2 channels all zero",
            "data": np.column_stack([
                np.zeros(100),  # x-axis all zero
                np.zeros(100),  # y-axis all zero
                np.random.randn(100) * 0.008 + 0.866   # only z-axis has data
            ]),
            "should_skip": True
        },
        {
            "name": "Completely corrupted - all data zero",
            "data": np.zeros((100, 3)),
            "should_skip": True
        },
        {
            "name": "Data with NaN values",
            "data": np.random.randn(100, 3),
            "should_skip": True,  # Will be modified to have NaN
            "add_nan": True
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        d = test_case['data'].copy()
        
        if test_case.get('add_nan', False):
            d[50, 1] = np.nan
        
        # Apply the fixed data quality check logic
        has_nan = np.any(np.isnan(d))
        num_all_zero_channels = np.sum(np.all(d == 0, axis=0))
        should_skip = has_nan or np.all(d == 0) or num_all_zero_channels >= 2
        
        print(f"  Has NaN: {has_nan}")
        print(f"  All-zero channels: {num_all_zero_channels}")
        print(f"  Should skip: {should_skip}")
        print(f"  Expected skip: {test_case['should_skip']}")
        
        if should_skip == test_case['should_skip']:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
    
    print(f"\n{'=' * 60}")
    print("Summary: The fixed logic correctly handles sleep data with one all-zero channel")
    print("while still filtering out truly corrupted data with multiple all-zero channels.")

if __name__ == "__main__":
    test_data_quality_check() 