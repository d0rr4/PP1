import pandas as pd
import argparse



def cath_txt_to_csv(input_file, output_file):
    columns = [
        "domain_id",
        "class",
        "architecture",
        "topology",
        "homology",
        "S35_cluster",
        "S60_cluster",
        "S95_cluster",
        "S100_cluster",
        "domain_length"
    ]

    df = pd.read_csv(
        input_file,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=columns
    )

    df.to_csv(output_file, index=False)
    print(f"Saved CSV to {output_file}")

cath_txt_to_csv("datasets/cath-domain-list-v4_3_0.txt", "datasets/cath_annotation.csv")