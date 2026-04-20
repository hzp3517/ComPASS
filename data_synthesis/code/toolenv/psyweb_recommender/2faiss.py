import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def build_faiss_index(
    data_path: str,
    index_path: str,
    mapping_path: str,
    model_name: str = "xx/all-mpnet-base-v2"
):

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer(model_name)

    texts = [item["user_situation"] for item in data]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[info] faiss index saved to: {index_path}")
    print(f"[info] data mapping saved to: {mapping_path}")
    print(f"[info] indexed {len(data)} items.")


if __name__ == "__main__":
    build_faiss_index(
        data_path="xx/Empathetic_Interaction/toolenv/v1/psyweb_recommender/psy_webDataset.json",
        index_path="xx/Empathetic_Interaction/toolenv/v1/psyweb_recommender/faiss_index_1.bin",
        mapping_path="xx/Empathetic_Interaction/toolenv/v1/psyweb_recommender/data_mapping_1.pkl"
    )