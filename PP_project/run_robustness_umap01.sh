#!/bin/bash

# Define datasets
DATASETS=("toxins")
EMBEDDINGS=(
  "data/toxins.h5:prott5"
)

# Robustness Grid
N_NEIGHBORS=(2 5 10 20 30 40 50)
MIN_DISTS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

# Total number of random runs desired
TOTAL_RUNS=50

# Generate, shuffle, and slice using Python safely by parsing strings
SELECTED_RUNS=$(python3 -c "
import random
nn_list = '${N_NEIGHBORS[*]}'.split()
md_list = '${MIN_DISTS[*]}'.split()
combos = [f'{nn}:{md}' for nn in nn_list for md in md_list]
random.shuffle(combos)
print('\n'.join(combos[:$TOTAL_RUNS]))
")

# Loop through datasets
for i in "${!DATASETS[@]}"; do
  DS_NAME=${DATASETS[$i]}
  DS_EMBED=${EMBEDDINGS[$i]}
  
  echo "-----------------------------------------"
  echo "Processing Dataset: $DS_NAME"
  echo "-----------------------------------------"


  run_count=1

  # Read the Python output line-by-line using a standard while loop
  while read -r combo; do
    [ -z "$combo" ] && continue

    # Extract parameters from the combo string
    nn=${combo%%:*}
    md=${combo#*:}

    OUTPUT_DIR="robustness/${DS_NAME}/umap_nn${nn}_md${md}/umap"
    echo "--> [Run ${run_count}/$TOTAL_RUNS] Running UMAP with n_neighbors=$nn, min_dist=$md"
    mkdir -p "$OUTPUT_DIR"
    
    # Execute protspace with explicit hyperparameter overrides
     #-m pca2,umap2,tsne2 \
    protspace prepare \
        -i "$DS_EMBED" \
        -m umap2 \
        -o "$OUTPUT_DIR" \
        --refetch annotations,projections \
        --eval \
        --label protein_families \
        --filter 50 \
        --n-neighbors "$nn" \
        --min-dist "$md"

    ((run_count++))
  done <<< "$SELECTED_RUNS"
done