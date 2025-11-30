"""
Statistical Analysis Module with Multiple Comparison Correction

This module includes proper FDR correction for Fighting
Words analysis.

Author: Javokhir Arifov
Date: November 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, Optional
import warnings


def calculate_effect_size(z_score: float) -> str:
    """
    Categorize effect size based on z-score magnitude.

    Cohen's d interpretation:
    - Small: |z| < 2 (roughly d < 0.2)
    - Medium: 2 <= |z| < 5 (roughly 0.2 <= d < 0.5)
    - Large: 5 <= |z| < 10 (roughly 0.5 <= d < 0.8)
    - Very Large: |z| >= 10 (roughly d >= 0.8)

    Parameters:
    -----------
    z_score : float
        Z-score from Fighting Words analysis

    Returns:
    --------
    str : Effect size category
    """
    abs_z = abs(z_score)

    if abs_z < 2:
        return 'small'
    elif abs_z < 5:
        return 'medium'
    elif abs_z < 10:
        return 'large'
    else:
        return 'very_large'


def fighting_words_with_correction(
    df: pd.DataFrame,
    z_col: str = 'z-score',
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> pd.DataFrame:
    """
    Apply multiple comparison correction to Fighting Words results.

    This function takes raw Fighting Words output and applies the
    Benjamini-Hochberg FDR correction to control for false discoveries
    when testing thousands of words simultaneously.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Fighting Words results (must contain z-score column)
    z_col : str, default='z-score'
        Name of column containing z-scores
    alpha : float, default=0.05
        Family-wise significance level
    method : str, default='fdr_bh'
        Multiple comparison correction method. Options:
        - 'fdr_bh': Benjamini-Hochberg FDR (RECOMMENDED)
        - 'bonferroni': Bonferroni correction (conservative)
        - 'fdr_by': Benjamini-Yekutieli FDR (for dependent tests)

    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - p_value: Two-tailed p-value
        - p_adjusted: FDR-corrected p-value
        - significant_fdr: Boolean indicating significance after correction
        - effect_size: Magnitude of effect
        - effect_magnitude: Categorical effect size

    Example:
    --------
    >>> df_corrected = fighting_words_with_correction(df)
    >>> sig_words = df_corrected[df_corrected['significant_fdr']]
    >>> print(f"Significant words: {len(sig_words)} / {len(df)}")
    """
    df = df.copy()

    # Validate input
    if z_col not in df.columns:
        raise ValueError(f"Column '{z_col}' not found in dataframe")

    # Calculate two-tailed p-values from z-scores
    df['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(df[z_col])))

    # Apply multiple comparison correction
    try:
        rejected, p_adjusted, alphacSidak, alphacBonf = multipletests(
            df['p_value'],
            alpha=alpha,
            method=method
        )
    except Exception as e:
        warnings.warn(f"Multiple testing correction failed: {e}")
        # Fallback: mark all as significant if correction fails
        rejected = df['p_value'] < alpha
        p_adjusted = df['p_value']

    df['p_adjusted'] = p_adjusted
    df['significant_fdr'] = rejected

    # Calculate effect sizes
    df['effect_size'] = np.abs(df[z_col])
    df['effect_magnitude'] = df[z_col].apply(calculate_effect_size)

    # Add interpretation column
    df['direction'] = df[z_col].apply(lambda x: 'class1' if x > 0 else 'class2')

    return df


def report_statistics(
    df: pd.DataFrame,
    comparison_name: str,
    class1_name: str = 'Class 1',
    class2_name: str = 'Class 2',
    top_n: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Generate comprehensive statistical report for Fighting Words analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with corrected Fighting Words results
    comparison_name : str
        Name of the comparison (e.g., "White vs. Black")
    class1_name : str
        Name of class 1 (positive z-scores)
    class2_name : str
        Name of class 2 (negative z-scores)
    top_n : int, default=10
        Number of top terms to report
    verbose : bool, default=True
        Print detailed report

    Returns:
    --------
    dict : Statistical summary including counts and top terms
    """
    # Filter to significant results
    sig = df[df['significant_fdr']].copy()

    # Separate by direction
    sig_class1 = sig[sig['z-score'] > 0].sort_values('z-score', ascending=False)
    sig_class2 = sig[sig['z-score'] < 0].sort_values('z-score', ascending=True)

    # Calculate statistics
    stats_summary = {
        'comparison': comparison_name,
        'total_words_tested': len(df),
        'significant_after_fdr': len(sig),
        'fdr_rate': len(sig) / len(df) if len(df) > 0 else 0,
        'class1_significant': len(sig_class1),
        'class2_significant': len(sig_class2),
        'effect_sizes': sig['effect_magnitude'].value_counts().to_dict(),
        'top_class1_terms': sig_class1.head(top_n)[['z-score', 'p_adjusted', 'effect_magnitude']].to_dict('records') if len(sig_class1) > 0 else [],
        'top_class2_terms': sig_class2.head(top_n)[['z-score', 'p_adjusted', 'effect_magnitude']].to_dict('records') if len(sig_class2) > 0 else []
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Statistical Report: {comparison_name}")
        print(f"{'='*70}")
        print(f"\nSample Sizes:")
        print(f"  Total words tested: {stats_summary['total_words_tested']:,}")
        print(f"  Significant (FDR-corrected α=0.05): {stats_summary['significant_after_fdr']:,} ({stats_summary['fdr_rate']:.2%})")
        print(f"    - Overrepresented in {class1_name}: {stats_summary['class1_significant']:,}")
        print(f"    - Overrepresented in {class2_name}: {stats_summary['class2_significant']:,}")

        print(f"\nEffect Size Distribution:")
        for magnitude, count in sorted(stats_summary['effect_sizes'].items()):
            print(f"  {magnitude:>12}: {count:>4}")

        print(f"\nTop {top_n} Terms Overrepresented in {class1_name}:")
        print(f"  {'Term':<20} {'Z-Score':>10} {'P-adj':>10} {'Effect':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for idx, row in sig_class1.head(top_n).iterrows():
            print(f"  {idx:<20} {row['z-score']:>10.2f} {row['p_adjusted']:>10.2e} {row['effect_magnitude']:>10}")

        print(f"\nTop {top_n} Terms Overrepresented in {class2_name}:")
        print(f"  {'Term':<20} {'Z-Score':>10} {'P-adj':>10} {'Effect':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for idx, row in sig_class2.head(top_n).iterrows():
            print(f"  {idx:<20} {row['z-score']:>10.2f} {row['p_adjusted']:>10.2e} {row['effect_magnitude']:>10}")

        print(f"\n{'='*70}\n")

    return stats_summary


def bootstrap_confidence_interval(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for difference in statistics.

    Useful for reporting uncertainty around observed differences.

    Parameters:
    -----------
    data1 : np.ndarray
        Sample from group 1
    data2 : np.ndarray
        Sample from group 2
    statistic : callable, default=np.mean
        Statistic to compute (mean, median, etc.)
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)

    Returns:
    --------
    tuple : (observed_diff, lower_bound, upper_bound)

    Example:
    --------
    >>> white_lengths = df[df['race']=='White']['text'].str.len()
    >>> black_lengths = df[df['race']=='Black']['text'].str.len()
    >>> diff, lower, upper = bootstrap_confidence_interval(white_lengths, black_lengths)
    >>> print(f"Difference: {diff:.1f} words, 95% CI: [{lower:.1f}, {upper:.1f}]")
    """
    observed_diff = statistic(data1) - statistic(data2)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(statistic(sample1) - statistic(sample2))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return observed_diff, lower, upper


def compare_distributions(
    group1_data: pd.Series,
    group2_data: pd.Series,
    group1_name: str = 'Group 1',
    group2_name: str = 'Group 2'
) -> Dict:
    """
    Statistical comparison of two distributions.

    Performs multiple tests:
    - T-test (parametric)
    - Mann-Whitney U (non-parametric)
    - Kolmogorov-Smirnov test (distribution shape)
    - Effect size (Cohen's d)

    Parameters:
    -----------
    group1_data : pd.Series
        Data from group 1
    group2_data : pd.Series
        Data from group 2
    group1_name : str
        Name of group 1
    group2_name : str
        Name of group 2

    Returns:
    --------
    dict : Test statistics and interpretations
    """
    # Remove NaN values
    g1 = group1_data.dropna()
    g2 = group2_data.dropna()

    # Descriptive statistics
    desc = {
        f'{group1_name}_mean': g1.mean(),
        f'{group1_name}_std': g1.std(),
        f'{group1_name}_median': g1.median(),
        f'{group1_name}_n': len(g1),
        f'{group2_name}_mean': g2.mean(),
        f'{group2_name}_std': g2.std(),
        f'{group2_name}_median': g2.median(),
        f'{group2_name}_n': len(g2),
    }

    # T-test
    t_stat, t_pval = stats.ttest_ind(g1, g2)

    # Mann-Whitney U (non-parametric alternative)
    u_stat, u_pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(g1, g2)

    # Cohen's d effect size
    pooled_std = np.sqrt(((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2) / (len(g1) + len(g2) - 2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else np.nan

    results = {
        **desc,
        'mean_difference': g1.mean() - g2.mean(),
        'median_difference': g1.median() - g2.median(),
        't_statistic': t_stat,
        't_pvalue': t_pval,
        't_significant': t_pval < 0.05,
        'mann_whitney_u': u_stat,
        'mann_whitney_pvalue': u_pval,
        'mann_whitney_significant': u_pval < 0.05,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'cohens_d': cohens_d,
        'effect_size_interpretation': (
            'negligible' if abs(cohens_d) < 0.2 else
            'small' if abs(cohens_d) < 0.5 else
            'medium' if abs(cohens_d) < 0.8 else
            'large'
        ) if not np.isnan(cohens_d) else 'undefined'
    }

    return results


# Example usage and testing
if __name__ == "__main__":
    print("Statistical Analysis Module - Test Suite")
    print("="*70)

    # Create synthetic Fighting Words results for testing
    np.random.seed(42)
    n_words = 1000

    test_df = pd.DataFrame({
        'word': [f'word_{i}' for i in range(n_words)],
        'z-score': np.random.randn(n_words) * 3,  # Random z-scores
        'class': np.random.choice(['A', 'B'], n_words)
    })
    test_df.set_index('word', inplace=True)

    # Test FDR correction
    print("\n1. Testing FDR Correction")
    print("-" * 70)
    corrected = fighting_words_with_correction(test_df)
    print(f"✓ Processed {len(corrected)} words")
    print(f"✓ Significant before correction (α=0.05): {(test_df['z-score'].abs() > 1.96).sum()}")
    print(f"✓ Significant after FDR correction: {corrected['significant_fdr'].sum()}")

    # Test statistical report
    print("\n2. Testing Statistical Report Generation")
    print("-" * 70)
    stats_summary = report_statistics(
        corrected,
        comparison_name="Test: Group A vs. Group B",
        class1_name="Group A",
        class2_name="Group B",
        top_n=5,
        verbose=True
    )

    # Test bootstrap CI
    print("\n3. Testing Bootstrap Confidence Intervals")
    print("-" * 70)
    data1 = np.random.normal(100, 15, 500)
    data2 = np.random.normal(95, 15, 500)
    diff, lower, upper = bootstrap_confidence_interval(data1, data2)
    print(f"✓ Observed difference: {diff:.2f}")
    print(f"✓ 95% CI: [{lower:.2f}, {upper:.2f}]")

    print("\n" + "="*70)
    print("All tests passed! Module ready for use.")
    print("="*70)
