#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting 3FTX - Signal preparation..."
protspace prepare \
    -i embeddings/3FTx_mature_prot_t5_clean.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+umap2 \
    -a datasets/3FTx_annotation.csv \
    --background embeddings/non3FTx_5050_lengthmatched.h5:5050_length \
    --background embeddings/non3FTx_signal_signalp6_merged_lengthmatched_prot_t5.h5:singalp6_length \
    --background embeddings/3FTx_signal_signalp6_merged_prot_t5.h5:signalp6_target \
    -o protspace_results/3FTx_signal \
    --refetch annotations,projections \
    --eval \
    --label major_group \
    --label signalp6.0 \
    --filter 50

echo "Starting 3FTX - Robustness preparation..."
protspace prepare \
    -i embeddings/3FTx_mature_prot_t5_clean.h5:prott5 \
    -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
    -a datasets/3FTx_annotation.csv \
    -o protspace_results/3FTx_robustness \
    --refetch annotations,projections \
    --eval \
    --label major_group \
    --robustness 10 \
    --filter 50

echo "Starting TOXINS - Signal preparation..."
protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide5050_lengthmatched.h5:5050 \
    --background embeddings/metazoa_nontoxin_signalpeptide_lengthmatched_signalp6_merged.h5:singalp6 \
    --background embeddings/toxins_signalpeptide_signalp6_merged_filtered.h5:singalp6_target \
    -o protspace_results/toxins_signal \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

echo "Starting TOXINS - Robustness preparation..."
protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,umap2,tsne2,pacmap2,densmap2,trimap2,phate2 \
    -a interpro,biocentral,uniprot \
    -o protspace_results/toxins_robustness \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --robustness 10 \
    --filter 50

echo "All protspace jobs completed successfully!"