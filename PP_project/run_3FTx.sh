#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting 3FTx - Removed Target Signal preparation..."
protspace prepare \
    -i embeddings/3FTx_mature_prot_t5.h5:prott5 \
    -o protspace_results/3FTx_signal_target \
    -m pca2,umap2,tsne2 \
    -a interpro,biocentral,uniprot \
    -a datasets/3FTx_annotation.csv \
    --refetch annotations,projections \
    --eval \
    --label major_group \
    --label has_full \
    --filter 50 \
    --robustness 20 



echo "Starting 3FTx - Signal preparation..."
protspace prepare \
    -i embeddings/3FTx_full_fallback_prot_t5.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    -a datasets/3FTx_annotation.csv \
    --background embeddings/non3FTx_5050.h5:complement \
    --background embeddings/non3FTx_5050_processed.h5:complement_singalp6 \
    --background embeddings/3FTx_5050_processed.h5:target_singalp6 \
    -o protspace_results/3FTx_signal \
    --refetch annotations,projections \
    --eval \
    --label major_group \
    --label has_full \
    --filter 50 \
    --robustness 20 


echo "Starting 3FTx - Robustness preparation..."
protspace prepare \
    -i embeddings/3FTx_full_fallback_prot_t5.h5:prott5 \
    -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
    -a interpro,biocentral,uniprot \
    -a datasets/3FTx_annotation.csv \
    -o protspace_results/3FTx_robustness \
    --refetch annotations,projections \
    --eval \
    --label major_group \
    --filter 50 \
    --robustness 20 

echo "All protspace jobs completed successfully!"