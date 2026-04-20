import os
import pickle
import math
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import random
import sys
import threading
from pathlib import Path
import base64
try:
    from APIclient import onechatAPIclient
except ImportError:
    pass

class EmojiRetriever:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_resources(*args, **kwargs)
        return cls._instance

    def _init_resources(self,
                        local_model_path="model/clip-vit-base-patch32",
                        index_filename="clip_image.index",
                        paths_filename="image_paths.pkl",
                        device_override=None):

        print(f"[Thread {threading.current_thread().name}] Initializing emoji retriever...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.api_model = onechatAPIclient(model="gpt-4.1")
        except NameError:
            self.api_model = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(base_dir, index_filename)
        self.paths_path = os.path.join(base_dir, paths_filename)
        self.device = device_override or device

        original_device = torch.device("cpu")
        if hasattr(torch, "get_default_device"):
            original_device = torch.get_default_device()

        try:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            if hasattr(torch, "set_default_tensor_type"):
                torch.set_default_tensor_type(torch.FloatTensor)

            self.model = CLIPModel.from_pretrained(
                local_model_path,
                low_cpu_mem_usage=False,
                device_map=None,
                torch_dtype=torch.float32
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(local_model_path)
            self.model.eval()
        finally:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device(original_device)

        self.index = faiss.read_index(self.index_path)

        with open(self.paths_path, "rb") as f:
            self.image_paths = pickle.load(f)

        print(f"[Thread {threading.current_thread().name}] Emoji retriever ready.")

    def _check_dim(self, text_emb_np):
        idx_dim = getattr(self.index, "d", None)
        if idx_dim and text_emb_np.shape[1] != idx_dim:
            raise ValueError("Dimension mismatch")

    def encode_texts(self, queries):
        single = isinstance(queries, str)
        if single:
            queries = [queries]

        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        text_np = text_emb.cpu().numpy().astype("float32")
        return text_np[0] if single else text_np

    def retrieve(self, query, top_k=5):
        text_np = self.encode_texts(query).reshape(1, -1).astype("float32")
        self._check_dim(text_np)
        D, I = self.index.search(text_np, 5)
        results = []
        for idx, i in enumerate(I[0]):
            if 0 <= i < len(self.image_paths):
                results.append((self.image_paths[i], float(D[0][idx])))
        return results

    def retrieve_batch(self, queries, top_k=5):
        text_np = self.encode_texts(queries).astype("float32")
        self._check_dim(text_np)
        D, I = self.index.search(text_np, 5)
        batch = []
        for q in range(len(D)):
            res = []
            for j, idx in enumerate(I[q]):
                if idx >= 0:
                    res.append((self.image_paths[idx], float(D[q][j])))
            batch.append(res)
        return batch

    def build_query(self, tag=None, subject=None, alternative_descriptions=None, style=None):
        q = ""
        if tag:
            q = str(tag) + (str(alternative_descriptions) if alternative_descriptions else "")
        elif alternative_descriptions:
            q = str(alternative_descriptions).strip()

        if subject:
            q += " " + str(subject) if q else str(subject)
        if style:
            q += " " + str(style) if q else str(style)
        return q.strip()

    def api_discrip(self, image_url):
        image_path = Path(image_url)
        if not image_path.exists():
            print(f"Warning: missing {image_path}")
            return "Image not found."

        with open(image_path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode()

        system_prompt = "you are a helpful assistance."
        user_prompt = "Describe this picture: features, style, emotion."
        prompt = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]

        if self.api_model:
            return self.api_model.apicall(system_prompt, prompt)
        return "API client unavailable."

    def retrieve_stickers(self, tag=None, subject=None, alternative_descriptions=None, style=None, topk=3):
        q = self.build_query(tag, subject, alternative_descriptions, style)
        if not q:
            print("[warn] Empty query")

        results = self.retrieve(q, top_k=topk)
        final = []
        take = min(topk, len(results))
        for i in range(take):
            desc = self.api_discrip(results[i][0])
            final.append({
                "image_url": results[i][0],
                "discription": desc
            })
        return final

    def show_results(self, results, cols=4):
        n = len(results)
        if n == 0:
            print("[info] No results")
            return
        rows = math.ceil(n / cols)
        plt.figure(figsize=(cols*2.5, rows*2.5))
        for i, (p, s) in enumerate(results):
            try:
                img = Image.open(p).convert("RGB")
            except:
                continue
            plt.subplot(rows, cols, i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{os.path.basename(p)}\n{s:.3f}")
        plt.tight_layout()
        plt.show()

emoji_retriever = EmojiRetriever()

def retrieve_emoji(tag=None, subject=None, alternative_descriptions=None, style=None, topk=3):
    return emoji_retriever.retrieve_stickers(tag, subject, alternative_descriptions, style, topk)

if __name__ == "__main__":
    res = retrieve_emoji(tag="sad", subject="cat", style="cartoon", topk=1)
    print("Top results:")
    print(res)