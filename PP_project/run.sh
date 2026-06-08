#!/bin/bash

echo "Starting SwissProt Prot-T5..."
protspace prepare \
    -i embeddings/per-protein.h5:prott5 \
    -m ,pca2,umap2,tsne2  \
    -o protspace_results/Prot-T5 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50

echo "Starting Multi Prot-T5..."
protspace prepare \
    -i embeddings/Prot-T5.h5:prott5 \
    -m rhopca2,pca2,umap2,tsne2  \
    -o protspace_results/Prot-T5 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50

echo "Starting Multi ESM..."
protspace prepare \
    -i embeddings/ESM.h5:esm \
    -m pca2,umap2,tsne2 \
    -o protspace_results/ESM \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50

echo "Starting Multi Prost-T5..."
protspace prepare \
    -i embeddings/Prost-T5.h5:prostt5 \
    -m pca2,umap2,tsne2 \
    -o protspace_results/Prost-T5 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50

echo "Starting Multi Ankh3-Large..."
protspace prepare \
    -i embeddings/Ankh3-Large.h5:ankh3-large \
    -m pca2,umap2,tsne2 \
    -o protspace_results/Ankh3-Large \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50

echo "All jobs completed!"