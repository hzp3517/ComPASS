import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def retriever(emotion_tag, query, index_dir="faiss_indexes", top_k=1):
    model = SentenceTransformer("/xxx/.../all-mpnet-base-v2",device="cpu")
    
    index = faiss.read_index(f"{index_dir}/{emotion_tag}.faiss")
    mapping = pd.read_csv(f"{index_dir}/{emotion_tag}_map.csv")
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)
    
    results = [(mapping.iloc[i]["situation"], float(scores[0][j])) for j, i in enumerate(ids[0])]
    return results

if __name__ == "__main__":
    query = "I just failed my exam again and feel really hopeless."
    emotion_tag = "sad"
    res = retriever(emotion_tag, query)
    for text, score in res:
        print(f"({score:.3f}) {text}")
