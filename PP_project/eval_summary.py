from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Define input path and load the dataset
input_path = Path("protspace_results/toxins/eval/prott5/summary.tsv")
df = pd.read_csv(input_path, sep="\t")

# 2. Process and split data into Unsupervised and Supervised sets
df_unsupervised = df[df["type"] == "unsupervised"].dropna(axis=1, how="all")
df_unsupervised.set_index("space", inplace=True)
df_unsupervised.drop(columns=["type"], inplace=True)

df_supervised = df[df["type"] == "supervised"].dropna(axis=1, how="all")
df_supervised.set_index("space", inplace=True)
df_supervised.drop(columns=["type"], inplace=True)


# 3. Define the visualization function
def generate_heatmap(dataframe, title, filename, figsize):
    # Perform column-wise Min-Max normalization for background colors
    df_normalized = (dataframe - dataframe.min()) / (
        dataframe.max() - dataframe.min()
    )

    # Setup the plot canvas
    fig, ax = plt.subplots(figsize=figsize)

    # Generate heatmap: color comes from normalized data, label text from raw data
    sns.heatmap(
        df_normalized,
        annot=dataframe,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    # Beautify labels and title
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_ylabel("Dimensionality Reduction Method", fontsize=11)
    ax.set_xlabel("Evaluation Metrics", fontsize=11)

    # Rotate x-axis ticks to avoid overlapping labels
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    # Save to file
    plt.savefig(filename, dpi=300)
    plt.close()


# 4. Derive dynamic output paths in the same directory
# e.g., summary.tsv -> summary_unsupervised.png and summary_supervised.png
output_unsupervised = input_path.with_name(f"{input_path.stem}_unsupervised.png")
output_supervised = input_path.with_name(f"{input_path.stem}_supervised.png")

# Generate and save both evaluation heatmaps
generate_heatmap(
    df_unsupervised,
    "Unsupervised Dimensionality Reduction Evaluation",
    output_unsupervised,
    figsize=(11, 8),
)

generate_heatmap(
    df_supervised,
    "Supervised Dimensionality Reduction Evaluation",
    output_supervised,
    figsize=(9, 6),
)

print(
    f"Heatmaps successfully generated and saved to:\n"
    f"  - {output_unsupervised}\n"
    f"  - {output_supervised}"
)
