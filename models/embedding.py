import os 
import pandas as pd 
import numpy as np 
from torchvision import transforms
import faiss 
from tqdm import tqdm
import clip 
import torch
from PIL import Image


# loading the CLIP   MODEL (ViT-B/32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model , preprocess= clip.load("ViT-B/16",device=device)


#image folder is product_images 
image_folder = "product_images"
image_files = [ f for f in os.listdir(image_folder) if f.endswith(".jpg")]  # all image path in image_files list

#preparing the storage 
image_ids = []
all_embeddings = []

# extracting the features from imagas and them embed it and store in faiss vector db

for img_files in tqdm(image_files,desc="Embedding Images"):
    image_path = os.path.join(image_folder,img_files)
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
            all_embeddings.append(embedding)
            image_ids.append(img_files.split('.')[0]) # store the image id
    except Exception as e:
        print(f"Error processing {img_files}: {str(e)}")

all_embeddings = np.stack(all_embeddings).astype('float32')


# creating the faiss index 
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

# saving the index and metadata 
faiss.write_index(index,"fashio_index.faiss")
np.save("image_ids.npy",np.array(image_ids))

print(f"faiss ndex sucessfully created {len(image_ids)} images")


