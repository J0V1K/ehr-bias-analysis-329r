#!/usr/bin/env python3
"""
Quick test script for CSV data loading.

Run this to verify your data loader is working correctly.

Usage:
    python test_data_loader.py
"""

import sys
from pathlib import Path

# Add project root to path (parent of tests directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_for_analysis


def main():
    print("\n" + "="*70)
    print("DATA LOADER TEST")
    print("="*70 + "\n")

    print("Testing data loader with merged_file_sample=100k_section=dischargeinstructions.csv")
    print()

    try:
        # Load small sample to test
        df = load_for_analysis(sample_size=100)

        print("✓ Data loaded successfully!")
        print()
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

        if 'text' in df.columns:
            print()
            print("Sample text (first 300 chars):")
            print(df['text'].iloc[0][:300] + "...")

        if 'race_simplified' in df.columns:
            print()
            print("Race distribution in sample:")
            print(df['race_simplified'].value_counts())

        print("\n" + "="*70)
        print("✓ SUCCESS! Data loader is working correctly")
        print("="*70)
        print("\nYou can now use this in your notebooks:")
        print("  from src.data_loader import load_for_analysis")
        print("  df = load_for_analysis()")
        print()

        return 0

    except FileNotFoundError as e:
        print("❌ ERROR: Data file not found")
        print()
        print(str(e))
        print("\nMake sure you have the data file in the current directory:")
        print("  merged_file_sample=100k_section=dischargeinstructions.csv")
        print()
        return 1

    except Exception as e:
        print("❌ ERROR: Something went wrong")
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
