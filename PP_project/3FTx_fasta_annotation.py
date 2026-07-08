import pandas as pd
import requests
import os

def process_3ftx_data(input_excel, output_csv, mature_fasta, full_fasta):
    # 1. Load the data
    print("Loading data...")
    df = pd.read_excel(input_excel)

    # Automatically scan for a full sequence column candidate if it exists
    full_seq_candidates = ['Sequence', 'full_seq', 'full_sequence', 'Full Sequence']
    full_seq_col = next((c for c in full_seq_candidates if c in df.columns), None)

    # 2. Define the columns to retain from the original spreadsheet
    cols_to_keep = [
        'id_new', 'identifier', 'Density-based clustering (ε=0.55)', 
        'Density-based clustering (ε=0.59)', 'Kmeans (n=7)', 'evolutionary_order', 
        'major_group', 'sub_group', 'cysteine_group', 'taxon_numerical', 
        'taxon_of_interest', 'family', 'genus', 'species', 'taxon_id', 
        'data_origin', 'membran_prediction', 'mature_seq', 'number_cysteines'
    ]
    
    if full_seq_col:
        cols_to_keep.append(full_seq_col)
    
    # Filter columns (ignoring any missing columns safely)
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[available_cols].copy()

    # Determine 'has_full' status per row
    if full_seq_col:
        df['has_full'] = df[full_seq_col].notna() & (df[full_seq_col].astype(str).str.strip() != '')
    else:
        print("Warning: No explicit full sequence column found. All entries flagged as has_full = False.")
        df['has_full'] = False

    # 3. Extract IDs
    print("Extracting IDs and filtering for UniProt entries...")
    uniprot_ids = set()
    row_info = {} # Maps row index -> (is_uniprot, extracted_id)

    for idx, row in df.iterrows():
        identifier = row.get('identifier')
        id_new = row.get('id_new')
        is_up = False
        
        if pd.notna(identifier) and '|' in str(identifier):
            parts = str(identifier).split('|')
            extracted_id = parts[1] if len(parts) > 1 else str(identifier)
            if len(parts) > 1 and parts[0].lower() in ['sp', 'tr']:
                is_up = True
                uniprot_ids.add(extracted_id)
        else:
            extracted_id = str(id_new) if pd.notna(id_new) else f"seq_{idx}"
        
        row_info[idx] = (is_up, extracted_id)

    # Add the unified 'ID' column and move it to the first position
    df['ID'] = [row_info[idx][1] for idx in df.index]
    cols = ['ID'] + [c for c in df.columns if c != 'ID']
    df = df[cols]

    # 4. Fetch Signal Peptide annotations from UniProt API in batches
    uniprot_ids = list(uniprot_ids)
    sp_annotations = {uid: False for uid in uniprot_ids} 
    
    chunk_size = 100
    print(f"Fetching data from UniProt API for {len(uniprot_ids)} unique IDs...")
    
    for i in range(0, len(uniprot_ids), chunk_size):
        chunk = uniprot_ids[i:i+chunk_size]
        query = " OR ".join([f"accession:{uid}" for uid in chunk])
        
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "fields": "accession,ft_signal",
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                for entry in results:
                    acc = entry.get("primaryAccession")
                    features = entry.get("features", [])
                    if features:
                        sp_annotations[acc] = True
            else:
                print(f"  Warning: Batch {i//chunk_size + 1} failed (Status code: {response.status_code})")
        except Exception as e:
            print(f"  Error fetching batch {i//chunk_size + 1}: {e}")

    # 5. Map UniProt annotations back to the original rows
    print("Mapping annotations back to dataset...")
    signal_peptide_col = []
    
    for idx in df.index:
        is_up, extracted_id = row_info[idx]
        
        if is_up and extracted_id in sp_annotations:
            signal_peptide_col.append(sp_annotations[extracted_id])
        else:
            signal_peptide_col.append(pd.NA)
            
    df['signal_peptide'] = signal_peptide_col

    # Rename 'identifier' to avoid Protspace collision
    if 'identifier' in df.columns:
        df.rename(columns={'identifier': 'original_identifier'}, inplace=True)

    # 6. Generate FASTA Files Before Stripping Sequence Columns
    print("Writing FASTA files...")
    
    with open(mature_fasta, 'w', encoding='utf-8') as f_mat, \
         open(full_fasta, 'w', encoding='utf-8') as f_full:
         
        for _, row in df.iterrows():
            seq_id = row['ID']
            m_seq = str(row['mature_seq']).strip() if pd.notna(row['mature_seq']) else ""
            
            # 1. Mature Sequence Only FASTA
            if m_seq:
                f_mat.write(f">{seq_id}\n{m_seq}\n")
            
            # 2. Full Sequence (with Mature Sequence Fallback) FASTA
            if row['has_full'] and full_seq_col:
                f_seq = str(row[full_seq_col]).strip()
                f_full.write(f">{seq_id}\n{f_seq}\n")
            elif m_seq: # Fallback to mature sequence
                f_full.write(f">{seq_id}\n{m_seq}\n")

    # Drop the raw sequence columns from the dataframe so it contains ONLY annotations
    cols_to_drop = [c for c in ['mature_seq', full_seq_col] if c and c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 7. Save the final annotation files (Both CSV and TSV)
    output_tsv = output_csv.replace('.csv', '.tsv')
    print(f"Saving annotation outputs...")
    df.to_csv(output_csv, index=False)
    df.to_csv(output_tsv, sep='\t', index=False)
    
    print(f"Done! Successfully saved:")
    print(f"  - Annotation CSV: {output_csv}")
    print(f"  - Annotation TSV: {output_tsv}")
    print(f"  - Mature FASTA:   {mature_fasta}")
    print(f"  - Full/FB FASTA:  {full_fasta}")

# --- Execute the script ---
if __name__ == "__main__":
    excel_input = "datasets/3FTx_raw_data.xlsx"
    csv_output = "datasets/3FTx_annotation.csv"
    
    # FASTA target locations
    mature_fasta_out = "datasets/3FTx_mature.fasta"
    full_fasta_out = "datasets/3FTx_full_fallback.fasta"
    
    process_3ftx_data(excel_input, csv_output, mature_fasta_out, full_fasta_out)