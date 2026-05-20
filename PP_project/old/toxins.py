import os
import csv
import re
import sys
from collections import Counter

def robust_venom_analysis(data_filepath):
    # Fallback to look for any local TSV/CSV if the specified file isn't found
    if not os.path.exists(data_filepath):
        possible_files = [f for f in os.listdir('.') if f.endswith(('.tsv', '.csv', '.txt')) and f != 'toxins.py']
        if possible_files:
            data_filepath = possible_files[0]
        else:
            print(f"❌ Error: Reference data file could not be found.")
            return

    orders = Counter()
    parent_families = Counter()
    subfamilies = Counter()
    total_records = 0

    with open(data_filepath, mode='r', encoding='utf-8') as f:
        # Sniff delimiter dynamically (handles tabs or commas smoothly)
        sample = f.read(4096)
        delimiter = '\t' if '\t' in sample else ','
        f.seek(0)
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            total_records += 1
            
            # 1. Robust Taxonomic Order Extraction
            lineage = row.get('Taxonomic lineage', '')
            if lineage:
                order_match = re.search(r'([^,]+)\s*\(order\)', lineage)
                if order_match:
                    orders[order_match.group(1).strip()] += 1
            
            # 2. Robust Protein Family & Subfamily Extraction
            families_raw = row.get('Protein families', '')
            if not families_raw or families_raw.strip() == '':
                parent_families['Unclassified Family'] += 1
                subfamilies['Unclassified Subfamily'] += 1
            else:
                parts = [p.strip() for p in families_raw.split(',')]
                
                # Safely extract Parent Family without breaking 'superfamily'
                first_part = parts[0]
                if first_part.endswith(" superfamily"):
                    parent = first_part.rsplit(" superfamily", 1)[0].strip()
                elif first_part.endswith(" family"):
                    parent = first_part.rsplit(" family", 1)[0].strip()
                else:
                    parent = first_part
                
                if not parent:
                    parent = 'Unclassified Family'
                parent_families[parent] += 1
                
                # Safely extract Subfamily matching the target keyword
                subfamily = 'Unclassified Subfamily'
                for part in parts:
                    if part.endswith(" subfamily"):
                        subfamily = part.rsplit(" subfamily", 1)[0].strip()
                        break
                subfamilies[subfamily] += 1

    # Format-stable execution output
    print("=" * 50)
    print("📊        DYNAMIC DATASET ANALYSIS SUMMARY         ")
    print("=" * 50)
    print(f"✨ Total Sequences Processed              : {total_records}")
    print(f"🦋 Total Unique Biological Orders Found   : {len(orders)}")
    print(f"🧪 Total Unique Parent Protein Families   : {len(parent_families)}")
    print(f"🧬 Total Unique Protein Subfamilies Found : {len(subfamilies)}")
    print("=" * 50)
    print()
    print("🔢 SEQUENCE COUNTS BY TAXONOMIC ORDER:")
    print("-" * 50)
    for order, count in orders.most_common():
        print(f" 🔹 {order:<32} : {count} sequences")
    print()
    print("🔝 TOP 20 PARENT PROTEIN FAMILIES:")
    print("-" * 50)
    for family, count in parent_families.most_common(20):
        print(f" 🧪 {family:<32} : {count} sequences")
    print("=" * 50)

if __name__ == "__main__":
    # Allows passing path as argument: python toxins.py my_data.tsv
    # Otherwise defaults to looking for "toxins.tsv"
    target_file = sys.argv[1] if len(sys.argv) > 1 else "toxins.tsv"
    robust_venom_analysis(target_file)