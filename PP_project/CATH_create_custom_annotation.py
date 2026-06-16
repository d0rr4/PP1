import pandas as pd
import argparse


def cath_txt_to_csv(input_file, output_file):
    # 'domain_id' is changed to 'uniprot_kb_id' so ProtSpace natively pairs 
    # it with your H5 keys during the join.
    # 'class' is renamed to 'cath_class' to stop it from colliding with 
    # ProtSpace's internal taxonomy keywords.
    columns = [
        "Entry",  
        "cath_class",     
        "architecture",
        "topology",
        "homology",
        "S35_cluster",
        "S60_cluster",
        "S95_cluster",
        "S100_cluster",
        "S100_count",
        "domain_length",
        "resolution"
    ]

    df = pd.read_csv(
        input_file,
        comment="#",
        sep=r"\s+",
        header=None,
        names=columns
    )

    # Convert numeric levels to descriptive string categories for ProtSpace's color engine
    df["cath_class"] = "Class_" + df["cath_class"].astype(str)
    df["architecture"] = "Arch_" + df["architecture"].astype(str)
    df["topology"] = "Topo_" + df["topology"].astype(str)
    df["homology"] = "Homol_" + df["homology"].astype(str)

    # Optional: Do the same for clusters if you want to color-code by cluster groups
    df["S35_cluster"] = "Cluster35_" + df["S35_cluster"].astype(str)

    df.to_csv(output_file, index=False)
    print(f"Saved CSV to {output_file}")


cath_txt_to_csv("datasets/cath-domain-list-v4_3_0.txt", "datasets/cath_annotation.csv")