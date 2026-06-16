import pandas as pd
import requests
import os

def process_3ftx_data(input_excel, output_csv):
    # 1. Load the data
    print("Loading data...")
    df = pd.read_excel(input_excel)

    # 2. Define the columns to retain from the original spreadsheet
    cols_to_keep = [
        'id_new', 'identifier', 'Density-based clustering (ε=0.55)', 
        'Density-based clustering (ε=0.59)', 'Kmeans (n=7)', 'evolutionary_order', 
        'major_group', 'sub_group', 'cysteine_group', 'taxon_numerical', 
        'taxon_of_interest', 'family', 'genus', 'species', 'taxon_id', 
        'data_origin', 'membran_prediction', 'mature_seq', 'number_cysteines'
    ]
    
    # Filter columns (ignoring any missing columns safely)
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[available_cols].copy()

    # 3. Extract IDs (Synchronized with FASTA generation logic to avoid alignment bugs)
    print("Extracting IDs and filtering for UniProt entries...")
    uniprot_ids = set()
    row_info = {} # Maps row index -> (is_uniprot, extracted_id)

    for idx, row in df.iterrows():
        identifier = row.get('identifier')
        id_new = row.get('id_new')
        is_up = False
        
        # Mirror the exact ID fallback hierarchy used in the FASTA script
        if pd.notna(identifier) and '|' in str(identifier):
            parts = str(identifier).split('|')
            extracted_id = parts[1] if len(parts) > 1 else str(identifier)
            # Explicitly flag as a UniProt entry if it uses standard Swiss-Prot/TrEMBL prefixes
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

    # 4.5. Load and parse SignalP 6.0 prediction outputs
    signalp_preds = {}
    pred_files = [
        "datasets/3FTx_prediction_results_part1.txt", 
        "datasets/3FTx_prediction_results_part2.txt"
    ]
    
    for pred_file in pred_files:
        if os.path.exists(pred_file):
            print(f"Loading SignalP 6.0 predictions from {pred_file}...")
            # skiprows=1 leaves the column names line active but jumps past the file header
            sp_df = pd.read_csv(pred_file, sep='\t', skiprows=1)
            # Remove the comment character '# ' out of the ID column name
            sp_df.columns = sp_df.columns.str.replace('# ', '').str.strip()
            
            for _, row in sp_df.iterrows():
                pid = str(row['ID']).strip()
                pred_val = str(row['Prediction']).strip()
                # Map into binary output (False if OTHER, True if any SP variants like SP, LIPO, TAT)
                signalp_preds[pid] = False if pred_val == 'OTHER' else True
        else:
            print(f"Warning: SignalP output file missing at: {pred_file}")

    # 5. Map annotations back to the original rows
    print("Mapping annotations back to dataset...")
    signal_peptide_col = []
    signalp6_col = []
    
    for idx in df.index:
        is_up, extracted_id = row_info[idx]
        
        # Map UniProt Web API results
        if is_up and extracted_id in sp_annotations:
            signal_peptide_col.append(sp_annotations[extracted_id])
        else:
            signal_peptide_col.append(pd.NA)
            
        # Map SignalP 6.0 Local CLI results
        if extracted_id in signalp_preds:
            signalp6_col.append(signalp_preds[extracted_id])
        else:
            signalp6_col.append(pd.NA)
            
    df['signal_peptide'] = signal_peptide_col
    df['signalp6.0'] = signalp6_col

    # 6. Save the final files (Both CSV and TSV)
    output_tsv = output_csv.replace('.csv', '.tsv')
    
    print(f"Saving outputs...")
    df.to_csv(output_csv, index=False)
    df.to_csv(output_tsv, sep='\t', index=False) # sep='\t' writes as Tab-Separated Values
    
    print(f"Done! Successfully saved:")
    print(f"  - CSV: {output_csv}")
    print(f"  - TSV: {output_tsv}")

# --- Execute the script ---
if __name__ == "__main__":
    process_3ftx_data("datasets/3FTx_raw_data.xlsx", "datasets/3FTx_annotation.csv")