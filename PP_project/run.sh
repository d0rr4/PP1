#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Starting protspace preparation pipeline..."

# ==========================================
# Command 1
# ==========================================
echo "Running job 1/6..."
protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide5050.h5 \
    -o protspace_results/toxins_5050 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

# ==========================================
# Command 2
# ==========================================
echo "Running job 2/6..."
protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide5050_lengthmatched.h5 \
    -o protspace_results/toxins_5050_length \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

# ==========================================
# Command 3
# ==========================================
echo "Running job 3/6..."
protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide_signalp6_merged_prot_t5.h5 \
    -o protspace_results/toxins_5050_signalp6 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

# ==========================================
# Command 4
# ==========================================
echo "Running job 4/6..."
protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide_lengthmatched_signalp6_merged.h5 \
    -o protspace_results/toxins_5050_signalp6_length \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

# ==========================================
# Command 5
# ==========================================
echo "Running job 5/6..."
protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/toxins_signalpeptide_signalp6_merged.h5 \
    -o protspace_results/toxins_5050_signalp6_target \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

# ==========================================
# Command 6
# ==========================================
echo "Running job 6/6..."
protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2 \
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+tsne2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/toxins_signalpeptide_signalp6_merged_filtered.h5 \
    -o protspace_results/toxins_5050_signalp6_target_filtered \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --label signal_peptide \
    --filter 50

echo "All protspace jobs completed successfully!"
