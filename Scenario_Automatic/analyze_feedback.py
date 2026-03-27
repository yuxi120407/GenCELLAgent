import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import argparse
import sys


def load_all_metrics(output_dir):
    """
    Load all iteration_summary.json files from all image folders
    
    Args:
        output_dir: Root directory containing all image folders
    
    Returns:
        List of dicts with all metrics
    """
    output_dir = Path(output_dir)
    all_data = []
    
    # Find all iteration_summary.json files
    summary_files = list(output_dir.rglob("iteration_summary.json"))
    print(f"Found {len(summary_files)} summary files")
    
    for summary_path in sorted(summary_files):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        image_name = summary.get('image_name', summary_path.parent.name)
        evaluation_mode = summary.get('evaluation_mode', 'unknown')
        best_iter = summary.get('best_iteration', {})
        quality_progression = summary.get('quality_progression', {})
        
        # Extract best iteration metrics
        best_metrics = best_iter.get('final_metrics', {})
        
        if not best_metrics:
            print(f"  ⚠ No best metrics found for {image_name}")
            continue
        
        # Build record
        record = {
            # Image info
            'image_name': image_name,
            'evaluation_mode': evaluation_mode,
            'num_sam3_iterations': summary.get('num_sam3_iterations', 0),
            'stopped_reason': summary.get('stopped_reason', 'unknown'),
            
            # Best iteration info
            'best_iteration_num': best_iter.get('iteration_number', 0),
            'best_quality_score': best_iter.get('quality_score', 0),
            
            # Best metrics
            'dice': best_metrics.get('dice', 0),
            'iou': best_metrics.get('iou', 0),
            'precision': best_metrics.get('precision', 0),
            'recall': best_metrics.get('recall', 0),
            'f1': best_metrics.get('f1', 0),
            'accuracy': best_metrics.get('accuracy', 0),
            
            # Pixel stats
            'true_positive': best_metrics.get('pixel_stats', {}).get('true_positive', 0),
            'false_positive': best_metrics.get('pixel_stats', {}).get('false_positive', 0),
            'false_negative': best_metrics.get('pixel_stats', {}).get('false_negative', 0),
            'true_negative': best_metrics.get('pixel_stats', {}).get('true_negative', 0),
            
            # Quality progression
            'initial_quality': quality_progression.get('initial_score', 0),
            'final_quality': quality_progression.get('final_score', 0),
            'best_quality': quality_progression.get('best_score', 0),
            'quality_improvement': quality_progression.get('improvement', 0),
            'quality_scores_all': quality_progression.get('scores', []),
        }
        
        all_data.append(record)
        print(f"  ✓ Loaded: {image_name} | Best Iter: {record['best_iteration_num']} | "
              f"Dice: {record['dice']:.3f} | IoU: {record['iou']:.3f} | "
              f"Quality: {record['best_quality_score']:.2f}")
    
    return all_data


def analyze_metrics(all_data, verbose=True):
    """
    Analyze all metrics and print comprehensive statistics
    
    Args:
        all_data: List of metric dictionaries
        verbose: Whether to print detailed statistics
    """
    if not all_data:
        print("No data to analyze!")
        return None
    
    df = pd.DataFrame(all_data)
    
    if not verbose:
        return df
    
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*80)
    
    print(f"\nTotal images analyzed: {len(df)}")
    print(f"Evaluation mode: {df['evaluation_mode'].iloc[0]}")
    
    # ========================================================================
    # 1. OVERALL STATISTICS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("1. OVERALL STATISTICS (Best Iteration Per Image)")
    print(f"{'='*80}")
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']
    
    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-"*65)
    
    for metric in metrics:
        vals = df[metric]
        print(f"{metric.capitalize():<15} "
              f"{vals.mean():<10.4f} "
              f"{vals.std():<10.4f} "
              f"{vals.min():<10.4f} "
              f"{vals.max():<10.4f} "
              f"{vals.median():<10.4f}")
    
    # ========================================================================
    # 2. QUALITY SCORE STATISTICS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("2. QUALITY SCORE STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    
    q_metrics = ['best_quality_score', 'initial_quality', 'final_quality', 'quality_improvement']
    q_labels = ['Best Quality Score', 'Initial Quality', 'Final Quality', 'Quality Improvement']
    
    for metric, label in zip(q_metrics, q_labels):
        vals = df[metric]
        print(f"{label:<25} "
              f"{vals.mean():<10.4f} "
              f"{vals.std():<10.4f} "
              f"{vals.min():<10.4f} "
              f"{vals.max():<10.4f}")
    
    # ========================================================================
    # 3. QUALITY vs ACTUAL METRICS CORRELATION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("3. QUALITY SCORE vs ACTUAL METRICS CORRELATION")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<15} {'Pearson r':<12} {'Pearson p':<12} {'Spearman r':<12} {'Interpretation'}")
    print("-"*70)
    
    for metric in ['dice', 'iou', 'precision', 'recall']:
        q_vals = df['best_quality_score'].values
        m_vals = df[metric].values
        
        if len(q_vals) >= 3:
            p_r, p_p = pearsonr(q_vals, m_vals)
            s_r, s_p = spearmanr(q_vals, m_vals)
            
            # Interpret
            abs_r = abs(p_r)
            if abs_r >= 0.7:
                interp = "Strong"
            elif abs_r >= 0.5:
                interp = "Moderate"
            elif abs_r >= 0.3:
                interp = "Weak"
            else:
                interp = "Very weak"
            direction = "+" if p_r > 0 else "-"
            
            print(f"{metric.capitalize():<15} "
                  f"{p_r:<12.3f} "
                  f"{p_p:<12.4f} "
                  f"{s_r:<12.3f} "
                  f"{interp} {direction}")
    
    # ========================================================================
    # 4. BEST ITERATION DISTRIBUTION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("4. BEST ITERATION DISTRIBUTION")
    print(f"{'='*80}")
    
    iter_counts = df['best_iteration_num'].value_counts().sort_index()
    print(f"\n{'Iteration':<15} {'Count':<10} {'Percentage'}")
    print("-"*40)
    for iter_num, count in iter_counts.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct / 5)
        print(f"{iter_num:<15} {count:<10} {pct:.1f}% {bar}")
    
    # ========================================================================
    # 5. PER-IMAGE SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("5. PER-IMAGE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Image':<30} {'Best Iter':<10} {'Quality':<10} {'Dice':<10} {'IoU':<10} {'Precision':<10} {'Recall'}")
    print("-"*90)
    
    for _, row in df.sort_values('dice', ascending=False).iterrows():
        print(f"{row['image_name']:<30} "
              f"{row['best_iteration_num']:<10} "
              f"{row['best_quality_score']:<10.3f} "
              f"{row['dice']:<10.4f} "
              f"{row['iou']:<10.4f} "
              f"{row['precision']:<10.4f} "
              f"{row['recall']:.4f}")
    
    # ========================================================================
    # 6. PERFORMANCE TIERS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("6. PERFORMANCE TIERS (by Dice Score)")
    print(f"{'='*80}")
    
    tiers = [
        ('Excellent', 0.8, 1.0),
        ('Good', 0.6, 0.8),
        ('Fair', 0.4, 0.6),
        ('Poor', 0.0, 0.4)
    ]
    
    for tier_name, low, high in tiers:
        tier_df = df[(df['dice'] >= low) & (df['dice'] < high)]
        pct = len(tier_df) / len(df) * 100
        print(f"\n{tier_name} (Dice {low:.1f}-{high:.1f}): {len(tier_df)} images ({pct:.1f}%)")
        if len(tier_df) > 0:
            print(f"  Mean Dice: {tier_df['dice'].mean():.4f}")
            print(f"  Mean IoU:  {tier_df['iou'].mean():.4f}")
    
    return df




def plot_metrics(df, output_dir='.', iou_threshold=0.05):
    """
    Create comprehensive plots for all metrics
    Filter to only samples with IoU > threshold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter dataframe
    original_count = len(df)
    df_filtered = df[df['iou'] > iou_threshold].copy()
    filtered_count = len(df_filtered)
    
    print(f"\nFiltering data: IoU > {iou_threshold}")
    print(f"  Original samples: {original_count}")
    print(f"  After filtering: {filtered_count}")
    print(f"  Removed: {original_count - filtered_count}")
    
    if filtered_count == 0:
        print("Warning: No samples remaining after filtering!")
        return None
    
    # Use filtered dataframe for all plots
    df = df_filtered
    
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # -----------------------------------------------------------------------
    # Plot 1: Metrics Distribution (Boxplots)
    # -----------------------------------------------------------------------
    
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']
    metric_data = [df[m].values for m in metrics]
    
    bp = ax1.boxplot(metric_data, labels=[m.capitalize() for m in metrics],
                     patch_artist=True, notch=False)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title(f'Metrics Distribution (IoU > {iou_threshold}, n={filtered_count})', 
                  fontsize=14, weight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax1.legend()
    
    # -----------------------------------------------------------------------
    # Plot 2: Mean Metrics Bar Chart
    # -----------------------------------------------------------------------
    
    ax2 = fig.add_subplot(gs[0, 2])
    means = [df[m].mean() for m in metrics]
    stds = [df[m].std() for m in metrics]
    
    bars = ax2.bar([m.capitalize() for m in metrics], means, 
                   yerr=stds, capsize=5, color=colors, alpha=0.7)
    
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title(f'Mean ± Std (IoU > {iou_threshold})', fontsize=14, weight='bold')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim([0, 1.1])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # -----------------------------------------------------------------------
    # Plot 3: Quality Score vs Dice
    # -----------------------------------------------------------------------
    
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['best_quality_score'], df['dice'],
                         c=df['best_iteration_num'], cmap='viridis',
                         s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax3, label='Best Iteration')
    
    # Fit line
    if len(df) >= 3:
        z = np.polyfit(df['best_quality_score'], df['dice'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['best_quality_score'].min(), df['best_quality_score'].max(), 100)
        ax3.plot(x_line, p(x_line), 'r--', alpha=0.7)
        
        r, pval = pearsonr(df['best_quality_score'], df['dice'])
        ax3.set_title(f'Quality Score vs Dice\n(r={r:.3f}, p={pval:.4f}, n={filtered_count})', 
                     fontsize=12, weight='bold')
    else:
        ax3.set_title(f'Quality Score vs Dice (n={filtered_count})', fontsize=12, weight='bold')
    
    ax3.set_xlabel('Quality Score', fontsize=11)
    ax3.set_ylabel('Dice Score', fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # -----------------------------------------------------------------------
    # Plot 4: Quality Score vs IoU
    # -----------------------------------------------------------------------
    
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(df['best_quality_score'], df['iou'],
               c=df['best_iteration_num'], cmap='viridis',
               s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

    plt.colorbar(scatter, ax=ax3, label='Best Iteration')
    
    if len(df) >= 3:
        z = np.polyfit(df['best_quality_score'], df['iou'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['best_quality_score'].min(), df['best_quality_score'].max(), 100)
        ax4.plot(x_line, p(x_line), 'r--', alpha=0.7)
        
        r, pval = pearsonr(df['best_quality_score'], df['iou'])
        ax4.set_title(f'Quality Score vs IoU\n(r={r:.3f}, p={pval:.4f}, n={filtered_count})',
                     fontsize=12, weight='bold')
    else:
        ax4.set_title(f'Quality Score vs IoU (n={filtered_count})', fontsize=12, weight='bold')

    x_min, x_max = df['best_quality_score'].min(), df['best_quality_score'].max()
    x_padding = (x_max - x_min) * 0.1
    ax4.set_xlim([max(0, x_min - x_padding), min(1, x_max + x_padding)])
    ax4.set_xlabel('Quality Score', fontsize=11)
    ax4.set_ylabel('IoU Score', fontsize=11)
    ax4.grid(alpha=0.3)
    #ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    # Add IoU threshold line
    ax4.axhline(y=iou_threshold, color='red', linestyle='--', alpha=0.5, label=f'IoU > {iou_threshold}')
    ax4.legend(fontsize=8)
    
    # -----------------------------------------------------------------------
    # Plot 5: Best Iteration Distribution
    # -----------------------------------------------------------------------
    
    ax5 = fig.add_subplot(gs[1, 2])
    iter_counts = df['best_iteration_num'].value_counts().sort_index()
    ax5.bar(iter_counts.index.astype(str), iter_counts.values, 
            color='#2196F3', alpha=0.7, edgecolor='black')
    ax5.set_title(f'Best Iteration Distribution (n={filtered_count})', fontsize=12, weight='bold')
    ax5.set_xlabel('Iteration Number', fontsize=11)
    ax5.set_ylabel('Number of Images', fontsize=11)
    ax5.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(iter_counts.values):
        ax5.text(i, v + 0.1, str(v), ha='center', fontsize=10, weight='bold')
    
    # -----------------------------------------------------------------------
    # Plot 6: Per-Image Dice (sorted)
    # -----------------------------------------------------------------------
    
    ax6 = fig.add_subplot(gs[2, :])
    sorted_df = df.sort_values('dice', ascending=False).reset_index(drop=True)
    
    x = range(len(sorted_df))
    width = 0.25
    
    ax6.bar([i - width for i in x], sorted_df['dice'], width,
            label='Dice', color='#2196F3', alpha=0.8)
    ax6.bar(x, sorted_df['iou'], width,
            label='IoU', color='#4CAF50', alpha=0.8)
    ax6.bar([i + width for i in x], sorted_df['precision'], width,
            label='Precision', color='#FF9800', alpha=0.8)
    ax6.bar([i + 2*width for i in x], sorted_df['recall'], width,
            label='Recall', color='#E91E63', alpha=0.8)
    
    ax6.set_xticks(x)
    ax6.set_xticklabels([name[:20] for name in sorted_df['image_name']], 
                        rotation=45, ha='right', fontsize=8)
    ax6.set_title(f'Per-Image Metrics (Sorted by Dice, IoU > {iou_threshold})', 
                  fontsize=14, weight='bold')
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_ylim([0, 1])
    ax6.legend(loc='upper right')
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(y=df['dice'].mean(), color='blue', linestyle='--', 
                alpha=0.5, label=f'Mean Dice: {df["dice"].mean():.3f}')
    # Add IoU threshold line
    ax6.axhline(y=iou_threshold, color='red', linestyle='--', 
                alpha=0.3, linewidth=2)
    
    # -----------------------------------------------------------------------
    # Plot 7: Quality Improvement vs Final Dice
    # -----------------------------------------------------------------------
    
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.scatter(df['quality_improvement'], df['dice'],
               s=80, alpha=0.7, color='#9C27B0', edgecolors='black', linewidths=0.5)
    ax7.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax7.set_title(f'Quality Improvement vs Final Dice (n={filtered_count})', 
                  fontsize=12, weight='bold')
    ax7.set_xlabel('Quality Score Improvement', fontsize=11)
    ax7.set_ylabel('Best Dice Score', fontsize=11)
    ax7.grid(alpha=0.3)
    
    # -----------------------------------------------------------------------
    # Plot 8: Precision vs Recall
    # -----------------------------------------------------------------------
    
    ax8 = fig.add_subplot(gs[3, 1])
    scatter8 = ax8.scatter(df['recall'], df['precision'],
                          c=df['dice'], cmap='RdYlGn',
                          s=80, alpha=0.7, edgecolors='black', linewidths=0.5,
                          vmin=0, vmax=1)
    plt.colorbar(scatter8, ax=ax8, label='Dice Score')
    
    # F1 iso-curves
    recall_range = np.linspace(0.01, 1, 100)
    for f1_val in [0.3, 0.5, 0.7]:
        precision_curve = (f1_val * recall_range) / (2 * recall_range - f1_val)
        valid = (precision_curve > 0) & (precision_curve <= 1)
        ax8.plot(recall_range[valid], precision_curve[valid], '--', 
                alpha=0.3, label=f'F1={f1_val}')
    
    ax8.set_title(f'Precision vs Recall (n={filtered_count})', fontsize=12, weight='bold')
    ax8.set_xlabel('Recall', fontsize=11)
    ax8.set_ylabel('Precision', fontsize=11)
    ax8.set_xlim([0, 1])
    ax8.set_ylim([0, 1])
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    # -----------------------------------------------------------------------
    # Plot 9: Quality Score Distribution
    # -----------------------------------------------------------------------
    
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.hist(df['best_quality_score'], bins=10, color='#00BCD4', 
             alpha=0.7, edgecolor='black')
    ax9.axvline(x=df['best_quality_score'].mean(), color='red', 
                linestyle='--', label=f"Mean: {df['best_quality_score'].mean():.3f}")
    ax9.set_title(f'Quality Score Distribution (n={filtered_count})', 
                  fontsize=12, weight='bold')
    ax9.set_xlabel('Quality Score', fontsize=11)
    ax9.set_ylabel('Count', fontsize=11)
    ax9.set_xlim([0, 1])
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    plt.suptitle(f'ER Segmentation Analysis - Filtered (IoU > {iou_threshold}, n={filtered_count}/{original_count})', 
                 fontsize=16, weight='bold', y=1.01)
    
    # Save
    plot_path = output_dir / f'metrics_analysis_iou_gt_{iou_threshold}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved filtered analysis plot: {plot_path}")
    
    return plot_path



def save_metrics_csv(df, output_dir='.', prefix=''):
    """Save all metrics to CSV"""
    output_dir = Path(output_dir)
    
    # Save full metrics
    csv_filename = f'{prefix}all_metrics.csv' if prefix else 'all_metrics.csv'
    csv_path = output_dir / csv_filename
    df_save = df.drop(columns=['quality_scores_all'])  # Remove list column
    df_save.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics CSV: {csv_path}")
    
    # Save summary statistics
    stats_filename = f'{prefix}metrics_summary_stats.csv' if prefix else 'metrics_summary_stats.csv'
    stats_path = output_dir / stats_filename
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy', 'best_quality_score']
    stats_df = df[metrics].describe()
    stats_df.to_csv(stats_path)
    print(f"✓ Saved summary stats: {stats_path}")
    
    return csv_path



def analyze_selection_accuracy(df_filtered, input_dir, iou_threshold=0.0, metric='iou'):
    """
    Analyze how well quality score selection matches actual best metric
    Uses the SAME filtered dataframe as main analysis for consistency
    
    Args:
        df_filtered: Already filtered DataFrame from main analysis
        input_dir: Root directory containing all image folders
        iou_threshold: IoU threshold (for display only, df is already filtered)
        metric: Which metric to use for ranking ('iou' or 'dice')
    """
    print("\n" + "="*80)
    print(f"SELECTION ACCURACY ANALYSIS (Based on {metric.upper()})")
    print("="*80)
    
    if iou_threshold > 0:
        print(f"\nUsing pre-filtered data: Images with best IoU > {iou_threshold}")
    
    print(f"\nFor each image, checking if highest quality score iteration")
    print(f"matches the best {metric.upper()} iteration...\n")
    
    results = {
        'perfect_match': 0,
        'top_2': 0,
        'top_3': 0,
        'worse': 0,
        'total': 0,
        'filtered_out': 0,  # Not used when pre-filtering with df
        'details': [],
        'metric': metric
    }
    
    input_dir = Path(input_dir)
    
    # Iterate through filtered dataframe
    for _, row in df_filtered.iterrows():
        image_name = row['image_name']
        
        # Reconstruct path to iteration_summary.json
        summary_path = input_dir / image_name / 'iteration_summary.json'
        
        if not summary_path.exists():
            print(f"Warning: {summary_path} not found")
            continue
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        evaluation_mode = summary.get('evaluation_mode', 'boxes_only')
        iterations = summary.get('iterations', {})
        
        # Extract quality scores and metrics for each SAM3 iteration
        iteration_metrics = []
        
        for iter_key in sorted(iterations.keys()):
            if not iter_key.startswith('iteration_'):
                continue
            
            iter_num = int(iter_key.split('_')[1])
            iter_data = iterations[iter_key].get(evaluation_mode, {})
            
            # Skip feedback-only iterations
            if iter_data.get('feedback_only', False):
                continue
            
            quality = iter_data.get('quality_score', 0)
            metrics = iter_data.get('metrics', {})
            iou = metrics.get('iou', 0)
            dice = metrics.get('dice', 0)
            
            if quality > 0 or iou > 0:  # Valid iteration
                iteration_metrics.append({
                    'iteration': iter_num,
                    'quality_score': quality,
                    'iou': iou,
                    'dice': dice
                })
        
        if len(iteration_metrics) < 2:
            continue  # Need at least 2 iterations to compare
        
        # Sort by quality (descending) and by target metric (descending)
        sorted_by_quality = sorted(iteration_metrics, key=lambda x: x['quality_score'], reverse=True)
        sorted_by_metric = sorted(iteration_metrics, key=lambda x: x[metric], reverse=True)
        
        # Find which iteration quality selected
        quality_selected = sorted_by_quality[0]['iteration']
        quality_selected_metric = sorted_by_quality[0][metric]
        quality_selected_quality = sorted_by_quality[0]['quality_score']
        quality_selected_iou = sorted_by_quality[0]['iou']
        quality_selected_dice = sorted_by_quality[0]['dice']
        
        # Find actual best metric iteration
        best_metric_iter = sorted_by_metric[0]['iteration']
        best_metric_value = sorted_by_metric[0][metric]
        best_metric_quality = sorted_by_metric[0]['quality_score']
        best_metric_iou = sorted_by_metric[0]['iou']
        best_metric_dice = sorted_by_metric[0]['dice']
        
        # Find rank of quality-selected iteration in metric ranking
        metric_rank = next((i for i, x in enumerate(sorted_by_metric) if x['iteration'] == quality_selected), -1)
        
        # Categorize
        results['total'] += 1
        
        detail = {
            'image_name': image_name,
            'quality_selected_iter': quality_selected,
            'quality_selected_quality': quality_selected_quality,
            'quality_selected_metric': quality_selected_metric,
            'quality_selected_iou': quality_selected_iou,
            'quality_selected_dice': quality_selected_dice,
            'best_metric_iter': best_metric_iter,
            'best_metric_value': best_metric_value,
            'best_metric_quality': best_metric_quality,
            'best_metric_iou': best_metric_iou,
            'best_metric_dice': best_metric_dice,
            'metric_rank': metric_rank,
            'total_iterations': len(iteration_metrics)
        }
        
        if metric_rank == 0:
            results['perfect_match'] += 1
            detail['category'] = 'Perfect'
        elif metric_rank == 1:
            results['top_2'] += 1
            detail['category'] = 'Top-2'
        elif metric_rank == 2:
            results['top_3'] += 1
            detail['category'] = 'Top-3'
        else:
            results['worse'] += 1
            detail['category'] = 'Worse'
        
        results['details'].append(detail)
    
    # Print summary
    total = results['total']
    
    print(f"Images analyzed: {total}\n")
    
    if total == 0:
        print("No valid data for selection accuracy analysis")
        return results
    
    print(f"Selection Accuracy (based on {metric.upper()}):")
    print(f"  Perfect Match (selected actual best {metric.upper()}):  {results['perfect_match']:3d} ({results['perfect_match']/total*100:5.1f}%)")
    print(f"  Top-2 (selected 2nd best {metric.upper()}):            {results['top_2']:3d} ({results['top_2']/total*100:5.1f}%)")
    print(f"  Top-3 (selected 3rd best {metric.upper()}):            {results['top_3']:3d} ({results['top_3']/total*100:5.1f}%)")
    print(f"  Worse (selected 4th or worse):                         {results['worse']:3d} ({results['worse']/total*100:5.1f}%)")
    
    print(f"\nCumulative Accuracy:")
    top1_acc = results['perfect_match'] / total * 100
    top2_acc = (results['perfect_match'] + results['top_2']) / total * 100
    top3_acc = (results['perfect_match'] + results['top_2'] + results['top_3']) / total * 100
    
    print(f"  Top-1 Accuracy: {top1_acc:.1f}%")
    print(f"  Top-2 Accuracy: {top2_acc:.1f}%")
    print(f"  Top-3 Accuracy: {top3_acc:.1f}%")
    
    # Show examples of mismatches
    print(f"\n{'='*80}")
    print("EXAMPLES OF SELECTION MISMATCHES")
    print(f"{'='*80}")
    
    mismatches = [d for d in results['details'] if d['category'] != 'Perfect']
    mismatches.sort(key=lambda x: x['best_metric_value'] - x['quality_selected_metric'], reverse=True)
    
    if mismatches:
        print(f"\nWorst 10 mismatches (largest {metric.upper()} gap):")
        print(f"\n{'Image':<30} {'Quality→':<8} {metric.upper():<7} {'Best→':<8} {metric.upper():<7} {'Gap':<7} {'Rank'}")
        print("-"*85)
        
        for detail in mismatches[:10]:
            gap = detail['best_metric_value'] - detail['quality_selected_metric']
            print(f"{detail['image_name']:<30} "
                  f"Iter {detail['quality_selected_iter']:<6} "
                  f"{detail['quality_selected_metric']:<7.3f} "
                  f"Iter {detail['best_metric_iter']:<6} "
                  f"{detail['best_metric_value']:<7.3f} "
                  f"{gap:<7.3f} "
                  f"{detail['category']}")
    else:
        print("\nNo mismatches found!")
    
    # Show best matches
    matches = [d for d in results['details'] if d['category'] == 'Perfect']
    matches.sort(key=lambda x: x['best_metric_value'], reverse=True)
    
    if matches:
        print(f"\n{'='*80}")
        print("EXAMPLES OF PERFECT MATCHES")
        print(f"{'='*80}")
        print(f"\nTop 10 perfect matches (highest {metric.upper()}):")
        print(f"\n{'Image':<30} {'Iter':<6} {'Quality':<9} {metric.upper():<7}")
        print("-"*60)
        
        for detail in matches[:10]:
            print(f"{detail['image_name']:<30} "
                  f"{detail['quality_selected_iter']:<6} "
                  f"{detail['quality_selected_quality']:<9.3f} "
                  f"{detail['quality_selected_metric']:<7.3f}")
    
    return results


def save_selection_accuracy_plot(results, output_dir='.', iou_threshold=0.0):
    """
    Create visualization of selection accuracy
    """
    output_dir = Path(output_dir)
    
    if results['total'] == 0:
        print("No data to plot for selection accuracy")
        return None
    
    metric = results.get('metric', 'iou')
    metric_upper = metric.upper()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Selection accuracy pie chart
    ax1 = axes[0, 0]
    categories = ['Perfect\nMatch', 'Top-2', 'Top-3', 'Worse']
    values = [
        results['perfect_match'],
        results['top_2'],
        results['top_3'],
        results['worse']
    ]
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
    
    wedges, texts, autotexts = ax1.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    
    filter_text = f' (IoU > {iou_threshold})' if iou_threshold > 0 else ''
    ax1.set_title(f'Selection Accuracy (by {metric_upper}){filter_text}\n(n={results["total"]})', 
                  fontsize=14, weight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(12)
    
    # Plot 2: Cumulative accuracy bar chart
    ax2 = axes[0, 1]
    total = results['total']
    cum_values = [
        results['perfect_match'] / total * 100,
        (results['perfect_match'] + results['top_2']) / total * 100,
        (results['perfect_match'] + results['top_2'] + results['top_3']) / total * 100
    ]
    bars = ax2.bar(['Top-1', 'Top-2', 'Top-3'], cum_values, 
                   color=['#4CAF50', '#8BC34A', '#FFC107'])
    ax2.set_title(f'Cumulative Accuracy (by {metric_upper})', fontsize=14, weight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cum_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Plot 3: Quality vs Metric for selected iterations
    ax3 = axes[1, 0]
    details = results['details']
    
    x_vals = [d['quality_selected_quality'] for d in details]
    y_vals = [d['quality_selected_metric'] for d in details]  # ← FIXED
    colors_scatter = ['#4CAF50' if d['category'] == 'Perfect' else '#F44336' for d in details]
    
    ax3.scatter(x_vals, y_vals, c=colors_scatter, alpha=0.6, s=60, 
               edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Quality Score (Selected)', fontsize=12)
    ax3.set_ylabel(f'{metric_upper} (Selected)', fontsize=12)
    ax3.set_title(f'Quality Score vs {metric_upper} of Selected Iterations', 
                  fontsize=14, weight='bold')
    ax3.grid(alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Perfect Match'),
        Patch(facecolor='#F44336', label='Mismatch')
    ]
    ax3.legend(handles=legend_elements, loc='lower right')
    
    # Plot 4: Metric gap distribution
    ax4 = axes[1, 1]
    gaps = [d['best_metric_value'] - d['quality_selected_metric'] for d in details]  # ← FIXED
    
    ax4.hist(gaps, bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Selection')
    ax4.set_xlabel(f'{metric_upper} Gap (Best - Selected)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title(f'Distribution of Selection {metric_upper} Gap', fontsize=14, weight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add stats text
    mean_gap = np.mean(gaps)
    median_gap = np.median(gaps)
    ax4.text(0.95, 0.95, f'Mean: {mean_gap:.3f}\nMedian: {median_gap:.3f}',
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    title_text = f'Quality Score Selection Accuracy Analysis (by {metric_upper})'
    if iou_threshold > 0:
        title_text += f'\n(Filtered: Best IoU > {iou_threshold})'
    
    plt.suptitle(title_text, fontsize=16, weight='bold')
    plt.tight_layout()
    
    plot_filename = f'selection_accuracy_{metric}_iou_gt_{iou_threshold}.png' if iou_threshold > 0 else f'selection_accuracy_{metric}.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved selection accuracy plot: {plot_path}")
    
    return plot_path


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Analyze ER segmentation metrics from iteration_summary.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results in directory
  python analyze_metrics.py --input sam3_results --output results_analysis
  
  # Generate only plots
  python analyze_metrics.py -i sam3_results -o results_analysis --plot-only
  
  # Generate only CSV
  python analyze_metrics.py -i sam3_results -o results_analysis --csv-only
  
  # Quiet mode (no verbose output)
  python analyze_metrics.py -i sam3_results -o results_analysis --quiet
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing iteration_summary.json files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for analysis results (default: same as input)'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots, skip CSV output'
    )
    
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Only generate CSV, skip plots'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Skip CSV generation'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    
    parser.add_argument(
        '--plot-filename',
        type=str,
        default='metrics_analysis.png',
        help='Filename for plot output (default: metrics_analysis.png)'
    )
    
    parser.add_argument(
        '--csv-prefix',
        type=str,
        default='',
        help='Prefix for CSV filenames'
    )

    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.05,
        help='Minimum IoU threshold for filtering samples (default: 0.05)'
    )
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output if args.output else args.input
    
    # Validate conflicting arguments
    if args.plot_only and args.csv_only:
        print("Error: Cannot specify both --plot-only and --csv-only")
        sys.exit(1)
    
    if args.plot_only and args.no_plot:
        print("Error: Cannot specify both --plot-only and --no-plot")
        sys.exit(1)
    
    if args.csv_only and args.no_csv:
        print("Error: Cannot specify both --csv-only and --no-csv")
        sys.exit(1)
    
    # Determine what to generate
    generate_plot = not (args.csv_only or args.no_plot)
    generate_csv = not (args.plot_only or args.no_csv)
    
    # ========================================================================
    # Load metrics
    # ========================================================================
    
    print("="*80)
    print("ER SEGMENTATION METRICS ANALYSIS")
    print("="*80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    print("\nLoading all metrics...")
    all_data = load_all_metrics(args.input)
    
    if not all_data:
        print("\n✗ No data found!")
        sys.exit(1)
    
    # ========================================================================
    # Analyze metrics
    # ========================================================================
    
    print(f"\nAnalyzing {len(all_data)} images...")
    df = analyze_metrics(all_data, verbose=not args.quiet)


    original_count = len(df)
    df_filtered = df[df['iou'] > args.iou_threshold].copy()
    filtered_count = len(df_filtered)


    print(f"\n{'='*80}")
    print(f"FILTERING: IoU > {args.iou_threshold}")
    print(f"{'='*80}")
    print(f"Original samples: {original_count}")
    print(f"After filtering: {filtered_count}")
    print(f"Removed: {original_count - filtered_count}")
    print(f"{'='*80}")

    selection_results = analyze_selection_accuracy(
        df_filtered, 
        args.input,
        iou_threshold=args.iou_threshold  # ← Add this
    )
    
    # ========================================================================
    # Generate outputs
    # ========================================================================
    
    if generate_plot:
        print("\nGenerating plots...")
        plot_metrics(df, output_dir=output_dir, iou_threshold=args.iou_threshold)
        save_selection_accuracy_plot(
            selection_results, 
            output_dir=output_dir,
            iou_threshold=args.iou_threshold  # ← Add this
        )
    
    if generate_csv:
        print("\nSaving CSV...")
        save_metrics_csv(df, output_dir=output_dir, prefix=args.csv_prefix)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Overall statistics (unfiltered)
    print(f"\nTotal images: {len(df)}")
    print(f"Overall (all samples):")
    print(f"  Mean Dice:    {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"  Mean IoU:     {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"  Mean Quality: {df['best_quality_score'].mean():.4f}")
    
    # Metrics at different IoU thresholds
    print(f"\n{'='*80}")
    print("METRICS AT DIFFERENT IoU THRESHOLDS")
    print(f"{'='*80}")
    
    thresholds = [0.0, 0.005, 0.01, 0.015]
    
    print(f"\n{'Threshold':<12} {'Count':<8} {'%':<8} {'Mean Dice':<12} {'Mean IoU':<12} {'Mean Quality':<12}")
    print("-"*80)
    
    for threshold in thresholds:
        df_thresh = df[df['iou'] > threshold]
        count = len(df_thresh)
        pct = (count / len(df) * 100) if len(df) > 0 else 0
        
        if count > 0:
            mean_dice = df_thresh['dice'].mean()
            mean_iou = df_thresh['iou'].mean()
            mean_quality = df_thresh['best_quality_score'].mean()
            
            print(f"IoU > {threshold:<6.2f} {count:<8} {pct:<7.1f}% "
                  f"{mean_dice:<12.4f} {mean_iou:<12.4f} {mean_quality:<12.4f}")
        else:
            print(f"IoU > {threshold:<6.2f} {count:<8} {pct:<7.1f}% "
                  f"{'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    # Show detailed stats for the main threshold used in plots
    if args.iou_threshold > 0:
        df_filtered = df[df['iou'] > args.iou_threshold]
        print(f"\n{'='*80}")
        print(f"FILTERED STATISTICS (IoU > {args.iou_threshold})")
        print(f"{'='*80}")
        
        if len(df_filtered) > 0:
            print(f"Samples: {len(df_filtered)}/{len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
            print(f"Removed: {len(df) - len(df_filtered)}")
            
            print(f"\nMetrics:")
            print(f"  Dice:      {df_filtered['dice'].mean():.4f} ± {df_filtered['dice'].std():.4f}")
            print(f"  IoU:       {df_filtered['iou'].mean():.4f} ± {df_filtered['iou'].std():.4f}")
            print(f"  Precision: {df_filtered['precision'].mean():.4f} ± {df_filtered['precision'].std():.4f}")
            print(f"  Recall:    {df_filtered['recall'].mean():.4f} ± {df_filtered['recall'].std():.4f}")
            print(f"  Quality:   {df_filtered['best_quality_score'].mean():.4f} ± {df_filtered['best_quality_score'].std():.4f}")
            
            # Show what was filtered out
            df_removed = df[df['iou'] <= args.iou_threshold]
            if len(df_removed) > 0:
                print(f"\nRemoved samples (IoU ≤ {args.iou_threshold}):")
                print(f"  Count: {len(df_removed)}")
                print(f"  Mean IoU: {df_removed['iou'].mean():.4f}")
                print(f"  Mean Dice: {df_removed['dice'].mean():.4f}")
                
                # Show worst performers
                worst = df_removed.nsmallest(5, 'iou')
                print(f"\n  Worst 5 removed:")
                for idx, row in worst.iterrows():
                    print(f"    - {row['image_name']}: IoU={row['iou']:.4f}, Dice={row['dice']:.4f}")
        else:
            print(f"Warning: No samples remain after filtering with IoU > {args.iou_threshold}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    if generate_plot:
        print(f"Plots filtered with: IoU > {args.iou_threshold}")
    print("="*80)


if __name__ == "__main__":
    main()