"""
Data loading utilities for MIMIC dataset from local CSV files.

This module provides functions to load the pre-processed MIMIC discharge
instruction data from the local CSV file.

Author: Javokhir Arifov
Date: November 2025
"""

import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


def load_mimic_data(
    filepath: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load MIMIC discharge instructions from CSV file.

    Parameters:
    -----------
    filepath : str, optional
        Path to CSV file. If None, uses default from current directory
    sample_size : int, optional
        Randomly sample this many records (for testing)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame : Loaded dataset

    Examples:
    ---------
    >>> # Load full dataset
    >>> df = load_mimic_data()

    >>> # Load with sampling
    >>> df = load_mimic_data(sample_size=1000)

    >>> # Load specific file
    >>> df = load_mimic_data(filepath="data/my_sample.csv")
    """
    # Default to the 100k discharge instructions file
    if filepath is None:
        filepath = "data/merged_file_sample=100k_section=dischargeinstructions.csv"

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n\n"
            f"Please ensure you have the MIMIC data file in the data/ directory.\n"
            f"Expected file: data/merged_file_sample=100k_section=dischargeinstructions.csv"
        )

    print("="*70)
    print("Loading MIMIC Discharge Instructions Data")
    print("="*70)
    print(f"File: {filepath}")
    print(f"Loading...")

    # Load CSV (using ; as separator based on the saved format)
    try:
        df = pd.read_csv(filepath, sep=';', low_memory=False)
    except:
        # Try comma separator if semicolon doesn't work
        df = pd.read_csv(filepath, low_memory=False)

    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"\nSampling {sample_size:,} records (random_state={random_state})...")
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"✓ Sampled to {len(df):,} records")

    print("="*70 + "\n")

    return df


def standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize demographic columns.

    Parameters:
    -----------
    df : pd.DataFrame
        MIMIC dataset

    Returns:
    --------
    pd.DataFrame : Dataset with standardized demographics
    """
    print("Standardizing demographics...")

    # Simplify race categories (as in paper)
    if 'race' in df.columns:
        df['race_simplified'] = df['race'].apply(simplify_race)
        print(f"  ✓ Created 'race_simplified' column")

        print(f"\nRace distribution:")
        race_counts = df['race_simplified'].value_counts()
        for race, count in race_counts.items():
            pct = count / len(df) * 100
            print(f"  {race:>12}: {count:>6,} ({pct:>5.1f}%)")

    # Standardize gender values
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.upper().str.strip()
        print(f"\nGender distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = count / len(df) * 100
            print(f"  {gender:>12}: {count:>6,} ({pct:>5.1f}%)")

    print()
    return df


def simplify_race(race: str) -> str:
    """
    Simplify detailed race categories into broad groups.

    As done in the paper: collapse to WHITE, BLACK, HISPANIC, ASIAN, OTHER

    Parameters:
    -----------
    race : str
        Original race/ethnicity string

    Returns:
    --------
    str : Simplified race category
    """
    if pd.isna(race):
        return 'UNKNOWN'

    race_upper = str(race).upper()

    if 'WHITE' in race_upper:
        return 'WHITE'
    elif 'BLACK' in race_upper or 'AFRICAN' in race_upper:
        return 'BLACK'
    elif 'HISPANIC' in race_upper or 'LATINO' in race_upper:
        return 'HISPANIC'
    elif 'ASIAN' in race_upper:
        return 'ASIAN'
    else:
        return 'OTHER'


def load_for_analysis(
    filepath: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline to load and prepare MIMIC data for analysis.

    This is the main function to use for your analysis pipelines.

    Parameters:
    -----------
    filepath : str, optional
        Path to CSV file
    sample_size : int, optional
        Randomly sample this many records (for testing)
    random_state : int
        Random seed for sampling
    standardize : bool
        Whether to standardize demographics

    Returns:
    --------
    pd.DataFrame : Ready-to-analyze dataset

    Examples:
    ---------
    >>> # Load full dataset
    >>> df = load_for_analysis()

    >>> # Load with sampling for testing
    >>> df = load_for_analysis(sample_size=1000)

    >>> # Load specific file
    >>> df = load_for_analysis(filepath="sample=8k.csv")
    """
    # 1. Load from CSV
    df = load_mimic_data(filepath, sample_size, random_state)

    # 2. Remove missing text
    if 'text' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['text'])
        removed = initial_len - len(df)
        if removed > 0:
            print(f"Removed {removed:,} records with missing text")

    # 3. Standardize demographics
    if standardize and ('race' in df.columns or 'gender' in df.columns):
        df = standardize_demographics(df)

    # 4. Report final statistics
    print("="*70)
    print("DATASET READY FOR ANALYSIS")
    print("="*70)
    print(f"Total records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("="*70 + "\n")

    return df


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MIMIC Data Loader - Test Suite")
    print("="*70 + "\n")

    # Test: Load dataset
    print("TEST: Loading Dataset")
    print("-"*70)

    try:
        # Try to load with small sample for testing
        df = load_for_analysis(sample_size=100)

        print("✓ Data loaded successfully!")
        print(f"\nShape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head(2))

        if 'text' in df.columns:
            print(f"\nSample text (first 300 chars):")
            print(df['text'].iloc[0][:300] + "...")

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("✓ Data loader is ready to use")
        print("="*70)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have the data file in the data/ directory:")
        print("  - data/merged_file_sample=100k_section=dischargeinstructions.csv")
        print("\nOr specify the correct path when calling load_for_analysis()")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()
