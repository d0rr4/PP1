#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- UNIREF50 ---
echo "Starting UNIREF50 - Length preparation..."
protspace prepare \
    -i embeddings/uniref50_under_2k_subset.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+tsne2 \
    -a interpro,uniprot \
    --background embeddings/uniref50_background_real.h5:real \
    --background embeddings/uniref50_background_random.h5:random \
    --background embeddings/uniref50_background_polyA.h5:polyA \
    --background embeddings/uniref50_background_shuffled.h5:shuffled \
    --background embeddings/uniref50_background_disprot.h5:disprot \
    -o protspace_results/length \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label continuous:length \
    --filter 50

# echo "Starting UNIREF50 - Robustness preparation..."
# protspace prepare \
#     -i embeddings/uniref50_under_2k_subset.h5:prott5 \
#     -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
#     -a interpro,uniprot \
#     --refetch annotations,projections \
#     -o protspace_results/robustness \
#     --eval \
#     --label protein_families \
#     --robustness 10 \
#     --filter 50

# --- UNIREF50 (ESM) ---
echo "Starting UNIREF50 (ESM) - Length preparation..."
protspace prepare \
    -i embeddings/uniref50_under_2k_subset_ESM.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+tsne2 \
    -a interpro,uniprot \
    --background embeddings/uniref50_background_real_ESM.h5:real \
    --background embeddings/uniref50_background_random_ESM.h5:random \
    --background embeddings/uniref50_background_polyA_ESM.h5:polyA \
    --background embeddings/uniref50_background_shuffled_ESM.h5:shuffled \
    --background embeddings/uniref50_background_disprot_ESM.h5:disprot \
    -o protspace_results/length_ESM \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label continuous:length \
    --filter 50

# echo "Starting UNIREF50 (ESM) - Robustness preparation..."
# protspace prepare \
#     -i embeddings/uniref50_under_2k_subset_ESM.h5:prott5 \
#     -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
#     -a interpro,uniprot \
#     --refetch annotations,projections \
#     -o protspace_results/robustness_ESM \
#     --eval \
#     --label protein_families \
#     --robustness 10 \
#     --filter 50

echo "All protspace jobs completed successfully!"