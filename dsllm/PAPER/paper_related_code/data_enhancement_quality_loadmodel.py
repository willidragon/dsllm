#!/usr/bin/env python
# coding: utf-8

# # Data Enhancement Quality Analysis
# 
# This notebook provides helper functions to evaluate the quality of data reconstructions by comparing upsampled data with ground truth high-resolution data.
# 
# The following metrics are computed:
# 1. Mean Squared Error (MSE)
# 2. Mean Absolute Error (MAE) 
# 3. Dynamic Time Warping (DTW)
# 4. Temporal Correlation (Pearson)
# 5. STFT Magnitude Error
# 6. Power Spectral Density (PSD) Similarity

# In[35]:


import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from scipy.signal import stft, welch
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from tqdm import tqdm
from fastdtw import fastdtw
import os

# --- Additional imports for new metrics ---
from scipy.stats import ks_2samp, anderson_ksamp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[36]:


from pathlib import Path

# Base directories
BASE_DIR = Path('/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm')
DATA_DIR = BASE_DIR / 'data'

# Data paths configuration
DATA_PATHS = {
    # Baseline (100DS)
    'baseline': {
        'train': {
            'data': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/train/capture24_train_data_stage2_300seconds_100DS.pkl',
            'labels': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/train/capture24_train_labels_stage2_300seconds_100DS.pkl'
        },
        'val': {
            'data': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/val/capture24_val_data_stage2_300seconds_100DS.pkl',
            'labels': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/val/capture24_val_labels_stage2_300seconds_100DS.pkl'
        },
        'test': {
            'data': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/test/capture24_test_data_stage2_300seconds_100DS.pkl',
            'labels': DATA_DIR / 'stage_2_compare_buffer/300seconds_100DS/test/capture24_test_labels_stage2_300seconds_100DS.pkl'
        }
    },
    
    # Comparison methods (upsampled to 100DS)
    'comparisons': {
        # SAITS upsampled to 100DS
        'SAITS': {
            'train': {
                'data': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/train/capture24_train_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/train/capture24_train_labels_stage2_300seconds_100DS_upsampled.pkl'
            },
            'val': {
                'data': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/val/capture24_val_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/val/capture24_val_labels_stage2_300seconds_100DS_upsampled.pkl'
            },
            'test': {
                'data': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/test/capture24_test_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled_saits/300seconds_100DS_upsampled_from_1000DS/test/capture24_test_labels_stage2_300seconds_100DS_upsampled.pkl'
            }
        },
        
        # Non-SAITS upsampled to 100DS
        'Non-SAITS': {
            'train': {
                'data': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/train/capture24_train_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/train/capture24_train_labels_stage2_300seconds_100DS_upsampled.pkl'
            },
            'val': {
                'data': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/val/capture24_val_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/val/capture24_val_labels_stage2_300seconds_100DS_upsampled.pkl'
            },
            'test': {
                'data': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/test/capture24_test_data_stage2_300seconds_100DS_upsampled.pkl',
                'labels': DATA_DIR / 'stage_2_upsampled/300seconds_100DS_upsampled_from_1000DS/test/capture24_test_labels_stage2_300seconds_100DS_upsampled.pkl'
            }
        }
    }
}

# Print available methods and check if files exist
print("Available methods:")
print("\nBaseline:")
for split in DATA_PATHS['baseline']:
    print(f"  {split}:")
    print(f"    data:   {DATA_PATHS['baseline'][split]['data']}")
    print(f"    labels: {DATA_PATHS['baseline'][split]['labels']}")
    print(f"    data exists: {DATA_PATHS['baseline'][split]['data'].exists()}")
    print(f"    labels exist: {DATA_PATHS['baseline'][split]['labels'].exists()}")

print("\nComparison methods:")
for method in DATA_PATHS['comparisons']:
    print(f"\n{method}:")
    for split in DATA_PATHS['comparisons'][method]:
        print(f"  {split}:")
        print(f"    data:   {DATA_PATHS['comparisons'][method][split]['data']}")
        print(f"    labels: {DATA_PATHS['comparisons'][method][split]['labels']}")
        print(f"    data exists: {DATA_PATHS['comparisons'][method][split]['data'].exists()}")
        print(f"    labels exist: {DATA_PATHS['comparisons'][method][split]['labels'].exists()}")


# In[37]:


import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from scipy.signal import stft, welch
from fastdtw import fastdtw

# --- Enhanced compute_metrics ---
def compute_metrics(original, enhanced):
    metrics = {}
    # Flatten for global metrics
    orig_flat = original.flatten()
    enh_flat = enhanced.flatten()
    # Basic error metrics
    metrics['mse'] = mean_squared_error(orig_flat, enh_flat)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(orig_flat, enh_flat)
    metrics['r2'] = r2_score(orig_flat, enh_flat)
    metrics['pearson_r'] = pearsonr(orig_flat, enh_flat)[0]
    metrics['mape'] = np.mean(np.abs((orig_flat - enh_flat) / (orig_flat + 1e-8))) * 100
    metrics['snr_db'] = 10 * np.log10(np.var(orig_flat) / np.var(orig_flat - enh_flat))
    # Per-axis metrics
    for axis, axis_name in enumerate(['x', 'y', 'z']):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        metrics[f'mse_{axis_name}'] = mean_squared_error(orig_1d, enh_1d)
        metrics[f'rmse_{axis_name}'] = np.sqrt(metrics[f'mse_{axis_name}'])
        metrics[f'mae_{axis_name}'] = mean_absolute_error(orig_1d, enh_1d)
        metrics[f'r2_{axis_name}'] = r2_score(orig_1d, enh_1d)
        metrics[f'pearson_r_{axis_name}'] = pearsonr(orig_1d, enh_1d)[0]
        metrics[f'mape_{axis_name}'] = np.mean(np.abs((orig_1d - enh_1d) / (orig_1d + 1e-8))) * 100
        metrics[f'snr_db_{axis_name}'] = 10 * np.log10(np.var(orig_1d) / np.var(orig_1d - enh_1d))
    # Temporal correlation (average across axes)
    correlations = []
    for axis in range(3):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        corr, _ = pearsonr(orig_1d, enh_1d)
        correlations.append(corr)
    metrics['temporal_correlation'] = np.mean(correlations)
    # DTW distance (average across axes)
    dtw_distances = []
    for axis in range(3):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        distance, _ = fastdtw(orig_1d, enh_1d, dist=lambda u, v: abs(u - v))
        dtw_distances.append(distance)
    metrics['dtw'] = np.mean(dtw_distances)
    # STFT magnitude error
    def compute_stft_error(x, y):
        x = x.flatten()
        y = y.flatten()
        _, _, Zxx_x = stft(x, fs=1, nperseg=64, noverlap=32)
        _, _, Zxx_y = stft(y, fs=1, nperseg=64, noverlap=32)
        return np.mean(np.abs(np.abs(Zxx_x) - np.abs(Zxx_y)))
    stft_errors = []
    for axis in range(3):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        error = compute_stft_error(orig_1d, enh_1d)
        stft_errors.append(error)
    metrics['stft_error'] = np.mean(stft_errors)
    # PSD similarity and frequency metrics
    def compute_psd_similarity(x, y):
        x = x.flatten()
        y = y.flatten()
        freqs, psd_x = welch(x, fs=1, nperseg=64, noverlap=32)
        _, psd_y = welch(y, fs=1, nperseg=64, noverlap=32)
        freq_corr = pearsonr(psd_x, psd_y)[0]
        high_freq_mask = freqs > 0.1 * np.max(freqs)
        high_freq_pres = np.sum(psd_y[high_freq_mask]) / (np.sum(psd_x[high_freq_mask]) + 1e-8)
        return np.mean(np.abs(psd_x - psd_y)), freq_corr, high_freq_pres
    psd_errors = []
    freq_corrs = []
    high_freq_preservs = []
    for axis in range(3):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        psd_err, freq_corr, high_freq_pres = compute_psd_similarity(orig_1d, enh_1d)
        psd_errors.append(psd_err)
        freq_corrs.append(freq_corr)
        high_freq_preservs.append(high_freq_pres)
    metrics['psd_error'] = np.mean(psd_errors)
    metrics['freq_corr'] = np.mean(freq_corrs)
    metrics['high_freq_pres'] = np.mean(high_freq_preservs)
    # Statistical tests (KS, AD, mean/std diff)
    for axis, axis_name in enumerate(['x', 'y', 'z']):
        orig_1d = original[:, axis].flatten()
        enh_1d = enhanced[:, axis].flatten()
        ks_stat, ks_p = ks_2samp(orig_1d, enh_1d)
        ad_stat, _, _ = anderson_ksamp([orig_1d, enh_1d])
        metrics[f'ks_stat_{axis_name}'] = ks_stat
        metrics[f'ks_p_{axis_name}'] = ks_p
        metrics[f'ad_stat_{axis_name}'] = ad_stat
        metrics[f'mean_diff_{axis_name}'] = np.mean(enh_1d) - np.mean(orig_1d)
        metrics[f'std_diff_{axis_name}'] = np.std(enh_1d) - np.std(orig_1d)
    return metrics


# In[38]:


def evaluate_enhancement(original_data_path, enhanced_data_path, num_samples=None):
    """Evaluate enhancement quality between original and enhanced datasets
    
    Args:
        original_data_path (str): Path to original high-res data pickle file
        enhanced_data_path (str): Path to enhanced data pickle file
        num_samples (int, optional): Number of samples to evaluate. If None, evaluate all.
    
    Returns:
        pd.DataFrame: DataFrame containing metrics for each sample
    """
    # Load data
    with open(original_data_path, 'rb') as f:
        original_data = pickle.load(f)
    with open(enhanced_data_path, 'rb') as f:
        enhanced_data = pickle.load(f)
        
    if num_samples is None:
        num_samples = len(original_data)
    else:
        num_samples = min(num_samples, len(original_data))
    
    # Compute metrics for each sample
    all_metrics = []
    for i in tqdm(range(num_samples), desc="Computing metrics"):
        metrics = compute_metrics(original_data[i], enhanced_data[i])
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    return df


# In[39]:


def plot_metrics_distribution(metrics_df):
    metrics_to_plot = [
        'mse', 'rmse', 'mae', 'r2', 'pearson_r', 'snr_db',
    ]
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.hist(metrics_df[metric], bins=20, alpha=0.7)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
    fig.tight_layout()
    return fig

def plot_example_comparison(original, enhanced, lowres=None, sample_idx=0, method=None):
    """Plot comparison between original, enhanced, and optionally low-res signals
    
    Args:
        original (np.ndarray): Original signal array (N, 3)
        enhanced (np.ndarray): Enhanced signal array (N, 3)
        lowres (np.ndarray, optional): Low-res input signal array (N, 3)
        sample_idx (int): Sample index for title
        method (str, optional): Method name for labeling
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    axes = axes.flatten()
    for i, (ax, axis_name) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
        ax.plot(original[:, i], label='Original', alpha=0.7)
        ax.plot(enhanced[:, i], label='Enhanced', alpha=0.7)
        if lowres is not None:
            stride = len(original) // len(lowres)
            lowres_indices = np.arange(len(lowres)) * stride
            ax.plot(lowres_indices, lowres[:, i], 'o', label='Low-res', alpha=0.5, markersize=3)
        ax.set_title(f'Axis {axis_name} - Sample {sample_idx} ({method})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Acceleration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# --- Add temporal metrics for all plotted examples and summary reporting ---
def compute_temporal_metrics(data1, data2):
    """Compute temporal similarity metrics for batches"""
    correlations = []
    trend_similarities = []
    for i in range(len(data1)):
        for axis in range(3):
            corr = np.corrcoef(data1[i][:, axis], data2[i][:, axis])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
            grad1 = np.gradient(data1[i][:, axis])
            grad2 = np.gradient(data2[i][:, axis])
            trend_corr = np.corrcoef(grad1, grad2)[0, 1]
            if not np.isnan(trend_corr):
                trend_similarities.append(trend_corr)
    return np.array(correlations), np.array(trend_similarities)


# In[40]:


# Example usage - evaluate SAITS upsampled data against baseline
baseline_data_path = DATA_PATHS['baseline']['train']['data']
enhanced_data_path = DATA_PATHS['comparisons']['SAITS']['train']['data']

print("Computing metrics for SAITS upsampled vs baseline...")
metrics_df = evaluate_enhancement(baseline_data_path, enhanced_data_path, num_samples=100)

print("\nMetrics Summary:")
print(metrics_df.describe())

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

# Save metrics DataFrame to CSV
metrics_csv_path = os.path.join(results_dir, "metrics_SAITS_vs_baseline.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved metrics DataFrame to {metrics_csv_path}")

print("\nPlotting distributions...")
fig1 = plot_metrics_distribution(metrics_df)
dist_plot_path = os.path.join(results_dir, "metrics_distribution_SAITS_vs_baseline.png")
fig1.savefig(dist_plot_path)
print(f"Saved metrics distribution plot to {dist_plot_path}")

# Plot first sample comparison
with open(baseline_data_path, 'rb') as f:
    original_data = pickle.load(f)
with open(enhanced_data_path, 'rb') as f:
    enhanced_data = pickle.load(f)

# Create subfolder for example comparisons
example_comparisons_dir = os.path.join(results_dir, "example_comparisons")
os.makedirs(example_comparisons_dir, exist_ok=True)

fig2 = plot_example_comparison(original_data[0], enhanced_data[0])
example_plot_path = os.path.join(example_comparisons_dir, "example_comparison_SAITS_vs_baseline.png")
fig2.savefig(example_plot_path)
print(f"Saved example comparison plot to {example_plot_path}")

# Show plots interactively if running in a notebook or interactive session
# plt.show()

if __name__ == "__main__":
    # Use TEST set for all evaluation and plotting
    baseline_data_path = DATA_PATHS['baseline']['test']['data']
    baseline_labels_path = DATA_PATHS['baseline']['test']['labels']
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load baseline labels (test set)
    with open(baseline_labels_path, 'rb') as f:
        baseline_labels = pickle.load(f)

    # Find two indices per unique activity_name in test set
    from collections import defaultdict
    activity_indices = defaultdict(list)
    for idx, label in enumerate(baseline_labels):
        activity = label['activity_name']
        if len(activity_indices[activity]) < 2:
            activity_indices[activity].append(idx)
    # activity_indices is now a dict: {activity_name: [idx1, idx2]}

    for method in ["SAITS", "Non-SAITS"]:
        enhanced_data_path = DATA_PATHS['comparisons'][method]['test']['data']
        print(f"Computing metrics for {method} upsampled vs baseline (TEST set only)...")
        # Compute metrics for ALL test samples
        metrics_df = evaluate_enhancement(baseline_data_path, enhanced_data_path, num_samples=None)

        print(f"\nMetrics Summary for {method} (TEST set):")
        print(metrics_df.describe())

        # Save metrics DataFrame to CSV
        metrics_csv_path = os.path.join(results_dir, f"metrics_{method}_vs_baseline.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics DataFrame to {metrics_csv_path}")

        print(f"\nPlotting distributions for {method} (TEST set)...")
        fig1 = plot_metrics_distribution(metrics_df)
        dist_plot_path = os.path.join(results_dir, f"metrics_distribution_{method}_vs_baseline.png")
        fig1.savefig(dist_plot_path)
        print(f"Saved metrics distribution plot to {dist_plot_path}")

        # Load data for plotting examples (for PNGs, use only 2 per activity from TEST set)
        with open(baseline_data_path, 'rb') as f:
            original_data = pickle.load(f)
        with open(enhanced_data_path, 'rb') as f:
            enhanced_data = pickle.load(f)
        # Create subfolder for example comparisons for this method
        example_comparisons_dir = os.path.join(results_dir, "example_comparisons", method)
        os.makedirs(example_comparisons_dir, exist_ok=True)
        # For each activity, plot two examples (PNGs only)
        example_metrics_rows = []
        for activity, indices in activity_indices.items():
            for i, idx in enumerate(indices):
                # Load low-res data for this method (test set)
                if method == "SAITS":
                    lowres_data_path = DATA_DIR / 'stage_2_compare_buffer/300seconds_1000DS/test/capture24_test_data_stage2_300seconds_1000DS.pkl'
                elif method == "Non-SAITS":
                    lowres_data_path = DATA_DIR / 'stage_2_compare_buffer/300seconds_1000DS/test/capture24_test_data_stage2_300seconds_1000DS.pkl'
                else:
                    lowres_data_path = None
                if lowres_data_path is not None and os.path.exists(lowres_data_path):
                    with open(lowres_data_path, 'rb') as f:
                        lowres_data = pickle.load(f)
                    lowres_sample = lowres_data[idx]
                else:
                    lowres_sample = None
                fig2 = plot_example_comparison(original_data[idx], enhanced_data[idx], lowres=lowres_sample, sample_idx=idx, method=method)
                example_plot_path = os.path.join(example_comparisons_dir, f"example_comparison_{activity.replace(' ', '_')}_{i}_{method}.png")
                fig2.savefig(example_plot_path)
                print(f"Saved example comparison plot to {example_plot_path}")

        # --- After all metrics are computed and saved, aggregate and output summary CSVs and Markdown report ---
        # Load the metrics DataFrame for all examples
        example_metrics_csv_path = os.path.join(example_comparisons_dir, "example_comparison_metrics.csv")
        metrics_df.to_csv(example_metrics_csv_path, index=False)
        print(f"Saved example comparison metrics table to {example_metrics_csv_path}")

        # --- Summary tables ---
        def summarize_metrics(df, groupby=None):
            import numpy as np
            numeric_df = df.select_dtypes(include=[np.number])
            if groupby:
                grouped = df.groupby(groupby)
                summary = grouped[numeric_df.columns].agg(['mean', 'std'])
            else:
                summary = numeric_df.agg(['mean', 'std'])
            return summary
        # Overall summary
        overall_summary = summarize_metrics(metrics_df) # Use metrics_df (all test samples) for summaries
        overall_summary.to_csv(os.path.join(example_comparisons_dir, "example_comparison_metrics_summary_overall.csv"))
        # By activity
        if 'activity_name' in metrics_df.columns: # Use metrics_df (all test samples) for summaries
            by_activity_summary = summarize_metrics(metrics_df, groupby='activity_name')
            by_activity_summary.to_csv(os.path.join(example_comparisons_dir, "example_comparison_metrics_summary_by_activity.csv"))
        # --- Markdown report ---
        def generate_markdown_report(overall_summary, by_activity_summary=None, out_path=None):
            lines = [f"# 🚀 Data Enhancement Quality - Performance Report ({method})\n"]
            lines.append("## Overall Metrics (mean ± std)\n")
            import pandas as pd
            for metric in overall_summary.index:
                try:
                    mean = overall_summary.loc[metric, 'mean']
                    std = overall_summary.loc[metric, 'std']
                except (KeyError, IndexError):
                    try:
                        mean = overall_summary.loc[metric][('mean',)] if ('mean',) in overall_summary.columns else overall_summary.loc[metric].xs('mean')
                        std = overall_summary.loc[metric][('std',)] if ('std',) in overall_summary.columns else overall_summary.loc[metric].xs('std')
                    except Exception:
                        mean = overall_summary.loc[metric].get('mean', float('nan'))
                        std = overall_summary.loc[metric].get('std', float('nan'))
                lines.append(f"- **{metric}**: {mean:.4f} ± {std:.4f}")
            if by_activity_summary is not None:
                lines.append("\n## By Activity (mean ± std)\n")
                lines.append(by_activity_summary.to_markdown())
            report = "\n".join(lines)
            if out_path:
                with open(out_path, 'w') as f:
                    f.write(report)
            return report
        # Generate and save report
        report_path = os.path.join(example_comparisons_dir, "example_comparison_metrics_report.md")
        generate_markdown_report(overall_summary, by_activity_summary if 'by_activity_summary' in locals() else None, out_path=report_path)
        print(f"Saved markdown report to {report_path}")

    # plt.show()  # Uncomment if running interactively

