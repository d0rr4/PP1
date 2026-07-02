import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_umap_summary(file_path):
    """Parses the weirdly formatted summary.tsv to extract UMAP2 metrics."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = df.columns.str.strip()
        df['space'] = df['space'].str.strip()
        df['type'] = df['type'].str.strip()
        
        metrics = {}
        unsupervised_row = df[df['space'].str.contains('UMAP2 - Full', na=False) | (df['type'] == 'unsupervised')]
        if not unsupervised_row.empty:
            metrics['recall_mean'] = unsupervised_row.iloc[0].get('recall_mean')
            metrics['trust_mean'] = unsupervised_row.iloc[0].get('trust_mean')
            metrics['cont_mean'] = unsupervised_row.iloc[0].get('cont_mean')
            
        supervised_row = df[(df['space'] == 'UMAP2') & (df['type'] == 'supervised')]
        if not supervised_row.empty:
            metrics['knn_acc_mean'] = supervised_row.iloc[0].get('knn_acc_mean')
            metrics['silhouette'] = supervised_row.iloc[0].get('silhouette')
            metrics['concordex'] = supervised_row.iloc[0].get('concordex')
            
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
def main_UMAP():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'robustness', 'toxins')
    all_results = []
    folder_pattern = re.compile(r'umap_nn(\d+)_md([\d\.]+)')
    
    print(f"Scanning directories in: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"Error: The directory {base_dir} does not exist.")
        return
    for folder_name in os.listdir(base_dir):
        run_folder_path = os.path.join(base_dir, folder_name)
        match = folder_pattern.match(folder_name)
        if match and os.path.isdir(run_folder_path):
            nn_val = int(match.group(1))
            md_val = float(match.group(2))
            tsv_path = os.path.join(run_folder_path, 'umap', 'eval', 'prott5', 'summary.tsv')
            
            if os.path.exists(tsv_path):
                metrics = parse_umap_summary(tsv_path)
                if metrics:
                    metrics['nn'] = nn_val
                    metrics['md'] = md_val
                    metrics['folder_name'] = folder_name
                    all_results.append(metrics)
            else:
                print(f"Warning: Missing summary file at {tsv_path}")

    if all_results:
        final_df = pd.DataFrame(all_results)
        
        metric_cols = ['recall_mean', 'trust_mean', 'cont_mean', 'knn_acc_mean', 'silhouette', 'concordex']
        for col in metric_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        columns_order = ['folder_name', 'nn', 'md'] + metric_cols
        final_df = final_df[columns_order]
        final_df = final_df.sort_values(by=['nn', 'md']).reset_index(drop=True)
        output_csv = os.path.join(base_dir, 'aggregated_umap_metrics.csv')
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved aggregated data to: {output_csv}")
        summary_stats = pd.DataFrame({
            'Mean': final_df[metric_cols].mean(),
            'Std Dev': final_df[metric_cols].std()
        })
        print("\n" + "="*45)
        print("   GLOBAL METRIC SUMMARY   ")
        print("="*45)
        print(summary_stats.round(4).to_string())
        print("="*45)
        
        stats_txt = os.path.join(base_dir, 'metrics_summary_stats.txt')
        with open(stats_txt, 'w') as f:
            f.write("GLOBAL METRIC SUMMARY STATISTICS:\n")
            f.write(f"Total processed runs: {len(final_df)}\n\n")
            f.write(summary_stats.to_string())
        print(f"Saved statistical summary to: {stats_txt}")
        
        try:
            plot_df = final_df.copy()
            for col in metric_cols:
                min_val = plot_df[col].min()
                max_val = plot_df[col].max()
                min_row = final_df.loc[final_df[col].idxmin()]
                print(f"Worst run for {col}: {min_row['folder_name']} (Value: {min_row[col]})")
            
                if max_val != min_val:
                    plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)
                else:
                    plot_df[col] = 1.0 
            
            melted_df = pd.melt(plot_df, id_vars=['folder_name'], value_vars=metric_cols,
                                var_name='Metric', value_name='Normalized Value (0 to 1)')
            
            plt.figure(figsize=(11, 6))
            sns.set_theme(style="whitegrid")
            
            ax = sns.boxplot(x='Metric', y='Normalized Value (0 to 1)', data=melted_df, 
                             palette='Set2', hue='Metric', legend=False)
            
            plt.xticks(rotation=15, ha='right')
            plt.title('Normalized Distribution of UMAP Metrics Across All Runs\n(Min-Max Scaled to [0, 1] per Metric)', 
                      fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Evaluation Metrics', fontsize=12, labelpad=10)
            plt.ylabel('Relative Variance Range [0, 1]', fontsize=12, labelpad=10)
            plt.tight_layout()
            
            plot_path = os.path.join(base_dir, 'metrics_distribution_boxplot_normalized.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved normalized distribution box plot to: {plot_path}")
            
        except Exception as plot_error:
            print(f"Could not generate box plot: {plot_error}")
        
    else:
        print("No valid evaluation metrics were found or parsed.")
        
if __name__ == "__main__":
    main_UMAP()