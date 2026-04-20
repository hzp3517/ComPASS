import os
import pickle
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import random

class Retriever:
    def __init__(self,
                 index_path: str = "faiss_index_1.bin",
                 id_map_path: str = "data_mapping_1.pkl",
                 model_name: str = "/xxx/all-mpnet-base-v2"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(base_dir, index_path)
        self.id_map_path = os.path.join(base_dir, id_map_path)

       
        self.index = faiss.read_index(self.index_path)

        with open(self.id_map_path, "rb") as f:
            self.data_map = pickle.load(f)

        self.model = SentenceTransformer(model_name_or_path='/xxx/all-mpnet-base-v2',device="cpu")

        self.all_data = self.data_map

    def _encode(self, texts: List[str]) -> np.ndarray:
        """embedding"""
        return self.model.encode(texts, normalize_embeddings=True)

    def psychology_websites_recommender(self,
               situation: str,
               problem_type: Optional[List[str]] = None,
               domain: Optional[str] = None,
               user_group: Optional[str] = None,
               topk: int = 3) -> List[dict]:

        candidates = self.data_map
        if problem_type:
            candidates = [d for d in candidates if any(t in d.get("tags", []) for t in problem_type)]
        elif domain:
            candidates = [d for d in candidates if d.get("domain") == domain]

        if user_group:
            candidates = [d for d in candidates if d.get("user_group") == user_group]

        
        if not candidates or len(candidates) < topk-1:
            candidates = self.all_data

        candidate_texts = [d["user_situation"] for d in candidates]
        candidate_embs = self._encode(candidate_texts)
        query_emb = self._encode([situation])[0]


        dim = candidate_embs.shape[1]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(candidate_embs)
        D, I = temp_index.search(query_emb.reshape(1, -1), 3)

        results = []
        for rank, idx in enumerate(I[0]):
            item = candidates[idx].copy()
            #print(item)
            results.append(
                {
                "link" : item["link"],
                "summary" : item["summary"]
            })
        #print("results:",results)
        return results


if __name__ == "__main__":
    retriever = Retriever(
        index_path="faiss_index_1.bin",
        id_map_path="data_mapping_1.pkl"
    )

    situation = "My work-life balance is overwhelming me, i drank too much alcohol recently"
    problem_type = []
    domain = None
    user_group = None

    results = retriever.psychology_websites_recommender(situation, problem_type=problem_type, domain=domain, user_group=user_group)
    print("resul:",results)
