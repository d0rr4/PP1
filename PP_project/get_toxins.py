import requests
import re
import time
import pandas as pd

base_url = "https://rest.uniprot.org"
url = f"{base_url}/uniprotkb/search"

query_params = {
    "query": (
        "reviewed:true "
        "AND taxonomy_id:33208 "
        "AND keyword:KW-0800 "
        "AND cc_tissue_specificity:venom"   # <-- this was missing
    ),
    "format": "tsv",
    "fields": "accession,id,sequence,lineage,protein_families",
    "size": 500
}

all_pages_raw = []
current_url = url
page_count = 1

next_link_pattern = re.compile(r'<([^>]+)>;\s*rel="next"')

print("Starting UniProt download...")

while current_url:
    print(f"Fetching page {page_count}...")
    try:
        response = requests.get(
            current_url,
            params=query_params if page_count == 1 else None,
            timeout=30
        )
        response.raise_for_status()

        page_text = response.text.strip()

        if page_text:
            if page_count > 1:
                lines = page_text.splitlines()
                if len(lines) > 1:
                    page_text = "\n".join(lines[1:])
            all_pages_raw.append(page_text)
            print(f"   Collected page {page_count}.")
        else:
            print("   Empty response. Stopping.")
            break

        current_url = None
        if "Link" in response.headers:
            match = next_link_pattern.search(response.headers["Link"])
            if match:
                next_url_raw = match.group(1).strip()
                current_url = f"{base_url}{next_url_raw}" if next_url_raw.startswith("/") else next_url_raw
                page_count += 1
                time.sleep(0.5)
                continue

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        break

if all_pages_raw:
    print("\nFormatting dataset...")
    complete_tsv_string = "\n".join(all_pages_raw)

    output_filename = "metadata/toxins.tsv"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(complete_tsv_string)

    print(f"Saved to '{output_filename}'.")

    df = pd.read_csv(output_filename, sep="\t")
    print(f"Total rows: {len(df)}")

    print("\n--- Dataset Profile ---")

    lineage_col = [col for col in df.columns if 'lineage' in col.lower()]
    if lineage_col:
        print(f"Unique full taxonomic lineages: {df[lineage_col[0]].dropna().nunique()}")
    else:
        print("Could not find lineage column.")

    family_col = [col for col in df.columns if 'famil' in col.lower()]
    if family_col:
        print(f"Unique protein families: {df[family_col[0]].dropna().nunique()}")
    else:
        print("Could not find protein families column.")

else:
    print("No data collected.")