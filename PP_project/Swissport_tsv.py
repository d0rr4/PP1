import requests
import re
import time

url = "https://rest.uniprot.org/uniprotkb/search"

# We request page sizes of 500 entries (UniProt's recommended maximum per page)
params = {
    "query": "reviewed:true",
    "format": "tsv",
    "fields": "accession,id,gene_names,organism_name,length,reviewed,protein_name",
    "size": 500
}

output_filename = "datasets/swissprot.tsv"
next_url = None
page_count = 0

print("Beginning paged download of Swiss-Prot entries...")

with open(output_filename, "w", encoding="utf-8") as f:
    while True:
        try:
            # If we have a next_url from the Link header, use it; otherwise start from scratch
            current_url = next_url if next_url else url
            response = requests.get(current_url, params=None if next_url else params)
            response.raise_for_status()
            
            lines = response.text.strip().split("\n")
            
            if len(lines) <= 1 and page_count > 0:
                # No data rows returned, we are done
                break
                
            # If it's the very first page, write the header + data rows. 
            # Otherwise, skip the header (index 0) and write only the data.
            if page_count == 0:
                f.write(response.text)
            else:
                f.write("\n" + "\n".join(lines[1:]))
                
            page_count += 1
            if page_count % 50 == 0:
                print(f"Downloaded {page_count * 500} entries successfully...")

            # Extract the next page cursor link from the HTTP Link response header
            link_header = response.headers.get("Link")
            if link_header and 'rel="next"' in link_header:
                # Extract the exact URL inside the < > characters
                next_url = re.search(r'<(.*)>; rel="next"', link_header).group(1)
            else:
                break # No next link means we reached the end
                
            # Slight sleep to respect UniProt's rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"\nError encountered on page {page_count}: {e}")
            print("Retrying this chunk in 5 seconds...")
            time.sleep(5)
            continue

print(f"\nFinished! Total pages grabbed: {page_count}. Combined data saved to '{output_filename}'")