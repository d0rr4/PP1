#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting TOXINS - Removed Target Signal preparation..."
protspace prepare \
    -i embeddings/toxins_20_2000_signal_processed_puls_nonsignal.h5:prott5 \
    -o protspace_results/toxins_signal_target \
    -m pca2,umap2,tsne2 \
    -a interpro,biocentral,uniprot \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label ft_signal_bin \
    --robustness 50 \
    --filter 50


echo "Starting TOXINS - Signal preparation..."
protspace prepare \
    -i embeddings/toxins_20_2000.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/nontoxins_20_2000_5050_lengthmatched.h5:complement \
    --background embeddings/nontoxins_20_2000_5050_processed.h5:complement_singalp6 \
    --background embeddings/toxins_20_2000_5050_processed.h5:target_singalp6 \
    -o protspace_results/toxins_signal \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label ft_signal_bin \
    --robustness 50 \
    --filter 50


# echo "Starting TOXINS - Robustness preparation..."
# protspace prepare \
#     -i embeddings/toxins_20_2000.h5:prott5 \
#     -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
#     -a interpro,biocentral,uniprot \
#     -o protspace_results/toxins_robustness \
#     --refetch annotations,projections \
#     --eval \
#     --label protein_families \
#     --robustness 50 \
#     --filter 50

echo "All protspace jobs completed successfully!"