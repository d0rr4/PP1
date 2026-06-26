

protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide5050.h5 \
    -o protspace_results/toxins_5050 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50



protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalpeptide_signalp6_merged_prot_t5.h5 \
    -o protspace_results/toxins_5050_signalp6 \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50



protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalp6_merged_lengthmatched.h5 \
    -o protspace_results/toxins_5050_signalp6_length \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50


protspace prepare \
    -i embeddings/toxins.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/toxins_signalpeptide_signalp6_merged.h5 \
    -o protspace_results/toxins_5050_signalp6_target \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50


protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/metazoa_nontoxin_signalp6_merged_lengthmatched_delta_full_minus_signalp6.h5 \
    -o protspace_results/toxins_5050_signalp6_length_delta \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50




protspace prepare \
    -i embeddings/toxins_filtered_15_300.h5:prott5 \
    -m pca2,rhopca2,umap2\
    -m rhopca64+umap2,rhopca128+umap2,rhopca256+umap2,rhopca512+umap2,rhopca1024+umap2 \
    -m tsne2,rhopca64+tsne2,rhopca128+tsne2,rhopca256+tsne2,rhopca512+tsne2,rhopca1024+umap2 \
    -a interpro,biocentral,uniprot \
    --background embeddings/toxins_signalpeptide_signalp9_removed_delta_symmetric_background_by_uniprot_id.h5 \
    -o protspace_results/toxins_5050_signalp6_target_delta \
    --refetch annotations,projections \
    --eval \
    --label protein_families \
    --filter 50