import faiss 
import numpy as np
import pandas as pd
import cv2
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

# First, we need to load the image from shopify and convert it into embedding

'''
-> IMage are downloaded with multi threading 32 workers.
'''

# MAX_WORKERS = 32  # Depending on your system — 32 is great for cloud
# SAVE_DIR = "product_images"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def download_image(row):
#     img_id = str(row['id'])
#     img_url = row['image_url']
#     try:
#         response = requests.get(img_url, timeout=10)
#         if response.status_code == 200:
#             file_path = os.path.join(SAVE_DIR, f"{img_id}.jpg")
#             with open(file_path, 'wb') as f:
#                 f.write(response.content)
#             return True
#         else:
#             return False
#     except Exception as e:
#         print(f"Failed to download image {img_id}: {e}")
#         return False

# # Read CSV and launch threads
# def read_csv_and_download(csv_file):
#     df = pd.read_csv(csv_file)
#     rows = df.to_dict(orient='records')

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = [executor.submit(download_image, row) for row in rows]
#         for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading Images"):
#             pass



# read_csv_and_download("images.csv")   # uncomment if you wnt to dowload the images..



