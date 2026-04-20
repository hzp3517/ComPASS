# image2faiss.py
import os
import sys
import pickle
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# basic config
local_model_path = os.path.join(base_dir, "models", "clip-vit-base-patch32")
image_folder = os.path.join(base_dir, "..", "SERdataset", "Images")
index_out = os.path.join(base_dir, "clip_image.index")
paths_out = os.path.join(base_dir, "image_paths.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] cwd={os.getcwd()}")
print(f"[info] base_dir={base_dir}")
print(f"[info] local_model_path={local_model_path}")
print(f"[info] image_folder={image_folder}")
print(f"[info] device={device}")

if not os.path.isdir(local_model_path):
    print(f"[error] local_model_path not found: {local_model_path}")
    sys.exit(1)
if not os.path.isdir(image_folder):
    print(f"[error] image_folder not found: {image_folder}")
    sys.exit(1)

print("[info] loading CLIP model...")
model = CLIPModel.from_pretrained(local_model_path).to(device)
processor = CLIPProcessor.from_pretrained(local_model_path)
model.eval()

supported_ext = (".jpg", ".jpeg", ".png")
image_paths = []
image_embeddings = []

file_count = 0
for root, dirs, files in os.walk(image_folder):
    print(f"[walk] root={root}  # files={len(files)}")
    for fname in files:
        file_count += 1
        if not fname.lower().endswith(supported_ext):
            continue
        path = os.path.join(root, fname)
        print(f"[proc] {path}")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {path}: {e}")
            continue

        print(f"       size={img.size}")

        try:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(**inputs)   # (1, D)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            image_paths.append(path)
            image_embeddings.append(emb.cpu().numpy())
        except Exception as e:
            print(f"[warn] failed to process {path}: {e}")
            continue

print(f"[info] total files encountered (any ext): {file_count}")
print(f"[info] embeddings extracted: {len(image_embeddings)}")

if len(image_embeddings) == 0:
    print("[error] no embeddings were extracted. Please check the printed logs above.")
    sys.exit(1)

image_embeddings = np.vstack(image_embeddings).astype("float32")

d = image_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(image_embeddings)
faiss.write_index(index, index_out)
print(f"[info] faiss index saved to: {index_out}")

with open(paths_out, "wb") as f:
    pickle.dump(image_paths, f)
print(f"[info] image paths saved to: {paths_out}")
