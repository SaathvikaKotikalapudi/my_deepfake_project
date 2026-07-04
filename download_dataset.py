import os
import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# --- CONFIGURATION ---
CSV_FILE_PATH = "FINAL_DATASET.csv"  
OUTPUT_DIR = "dataset"
MAX_CONCURRENT_DOWNLOADS = 15  
TIMEOUT_SECONDS = 30

async def download_image(session, semaphore, url, destination_path):
    if os.path.exists(destination_path):
        return True  # Skip if already downloaded

    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    with open(destination_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
                    return True
                else:
                    return False
        except Exception:
            return False

async def main():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at '{CSV_FILE_PATH}'.")
        return

    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Use your exact matched columns
    url_col = 'image_url'
    label_col = 'label'
    split_col = 'dataset_split'

    print(f"✅ Columns mapped successfully!")
    print(f"Loaded metadata for {len(df)} images. Building folder structures...")

    # Create directories based on your exact column values
    splits = df[split_col].unique()        
    labels = df[label_col].unique()        
    
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(OUTPUT_DIR, str(split), str(label)), exist_ok=True)

    # Initialize async workers
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, row in df.iterrows():
            url = row[url_col]
            split = str(row[split_col]).strip()
            label = str(row[label_col]).strip()
            
            # Determine file extension from image_id or fallback to .jpg
            img_id = str(row['image_id'])
            ext = os.path.splitext(url.split('?')[0])[-1]
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                ext = '.jpg'
                
            filename = f"{img_id}{ext}"
            dest_path = os.path.join(OUTPUT_DIR, split, label, filename)
            
            tasks.append(download_image(session, semaphore, url, dest_path))
        
        print(f"Starting download pipeline (Max concurrency: {MAX_CONCURRENT_DOWNLOADS})...")
        results = await tqdm_asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r)
        failed_count = len(results) - success_count
        print(f"\n--- Execution Summary ---")
        print(f"Successfully processed: {success_count} images")
        print(f"Failed/Skipped targets: {failed_count} images")

if __name__ == "__main__":
    asyncio.run(main())