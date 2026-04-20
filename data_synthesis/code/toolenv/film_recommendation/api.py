import os
import json
import pandas as pd
import ast
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import random

warnings.filterwarnings('ignore')

# config
MODEL_PATH = "/xxx/.../all-mpnet-base-v2"
MOVIES_PATH = os.path.join(os.path.dirname(__file__), "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(os.path.dirname(__file__), "tmdb_5000_credits.csv")
EMBEDDING_CACHE_PATH = os.path.join(os.path.dirname(__file__), "movie_embeddings.npy")
DATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "processed_movies.json")


def dict_to_string(d, key_val_sep=": ", item_sep=", "):
    return item_sep.join([f"{k}{key_val_sep}{v}" for k, v in d.items()])


class MovieRetriever:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_resources()
        return cls._instance

    def _init_resources(self):
        print("Initializing resources...")

        self.model = SentenceTransformer(model_name_or_path=MODEL_PATH, device="cpu")

        if os.path.exists(DATA_CACHE_PATH) and os.path.exists(EMBEDDING_CACHE_PATH):
            self._load_from_cache()
        else:
            self._process_and_cache_data()

        print("Initialization complete")

    def _parse_json_list_safe(self, x):
        try:
            return [item["name"].lower() for item in ast.literal_eval(x)] if pd.notna(x) else []
        except:
            return []

    def _parse_cast(self, c):
        try:
            return [i["name"].lower() for i in ast.literal_eval(c)] if pd.notna(c) else []
        except:
            return []

    def _parse_crew(self, c):
        try:
            items = ast.literal_eval(c) if pd.notna(c) else []
            return {i["job"].lower(): i["name"].lower() for i in items}
        except:
            return {}

    def _process_and_cache_data(self):
        if not os.path.exists(MOVIES_PATH):
            raise FileNotFoundError(f"Movies file not found at {MOVIES_PATH}")
        if not os.path.exists(CREDITS_PATH):
            raise FileNotFoundError(f"Credits file not found at {CREDITS_PATH}")

        df_movies = pd.read_csv(MOVIES_PATH)
        df_credits = pd.read_csv(CREDITS_PATH)

        df_movies["genres"] = df_movies["genres"].apply(self._parse_json_list_safe)
        df_movies["keywords"] = df_movies["keywords"].apply(self._parse_json_list_safe)
        df_movies["overview"] = df_movies["overview"].fillna("").str.lower()
        df_movies["title"] = df_movies["title"].fillna("").str.lower()
        df_movies["id"] = df_movies["id"].astype(int)

        df_movies["release_year"] = pd.to_datetime(df_movies["release_date"], errors="coerce").dt.year
        df_movies["release_year"] = df_movies["release_year"].where(pd.notna(df_movies["release_year"]), None)

        df_credits["cast"] = df_credits["cast"].apply(self._parse_cast)
        df_credits["crew"] = df_credits["crew"].apply(self._parse_crew)
        df_credits["movie_id"] = df_credits["movie_id"].astype(int)

        self.df = df_movies.merge(
            df_credits[["movie_id", "cast", "crew"]],
            left_on="id",
            right_on="movie_id",
            how="left"
        ).drop(columns=["movie_id"])

        self.df["vote_average"] = self.df["vote_average"].fillna(0.0)
        self.df["vote_count"] = self.df["vote_count"].fillna(0)

        self.df["cast"] = self.df["cast"].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [])
        self.df["crew"] = self.df["crew"].apply(lambda x: x if isinstance(x, dict) and len(x) > 0 else {})

        self.movie_texts = [
            f"{row['title']} {' '.join(row['genres'])} {' '.join(row['keywords'])} {row['overview']}"
            for _, row in self.df.iterrows()
        ]

        self.movie_embeddings = self.model.encode(
            self.movie_texts,
            device="cpu",
            batch_size=32,
            show_progress_bar=False
        )

        self.df.to_json(DATA_CACHE_PATH, orient="records", force_ascii=False)
        np.save(EMBEDDING_CACHE_PATH, self.movie_embeddings)

    def _load_from_cache(self):
        self.df = pd.DataFrame(json.load(open(DATA_CACHE_PATH, "r", encoding="utf-8")))

        for col in ["genres", "keywords", "cast", "crew"]:
            self.df[col] = self.df[col].apply(lambda x: x if isinstance(x, (list, dict)) else [])

        self.movie_embeddings = np.load(EMBEDDING_CACHE_PATH)

        self.movie_texts = [
            f"{row['title']} {' '.join(row['genres'])} {' '.join(row['keywords'])} {row['overview']}"
            for _, row in self.df.iterrows()
        ]

    def retrieve_movies(
        self,
        keyword: Optional[List[str]] = None,
        actors: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        director: Optional[str] = None,
        min_rating: float = 0.0,
        max_rating: float = 10.0,
        min_vote_count: int = 20,
        emotion: Optional[str] = None,
        page: int = 1,
        include_adult: bool = False,
        max_results: int = 5,
        similarity_threshold: float = 0.1
    ) -> List[Dict]:

        query_parts = []
        if keyword:
            query_parts.extend(g.lower() for g in keyword)
        if genres:
            query_parts.extend([g.lower() for g in genres])
        if emotion:
            query_parts.append(emotion.lower())
        if actors:
            query_parts.extend([a.lower() for a in actors])
        if director:
            query_parts.append(director.lower())

        user_query = " ".join(query_parts)

        if not user_query:
            similarities = np.ones(len(self.df))
        else:
            query_embedding = self.model.encode([user_query], device="cpu")
            similarities = cosine_similarity(query_embedding, self.movie_embeddings)[0]

        mask = np.ones(len(self.df), dtype=bool)

        mask &= (self.df["vote_average"] >= min_rating) & (self.df["vote_average"] <= max_rating)
        mask &= (self.df["vote_count"] >= min_vote_count)

        if not include_adult:
            mask &= ~self.df["title"].str.contains("adult", case=False, na=False)

        if genres:
            genres_lower = set([g.lower() for g in genres])
            mask &= self.df["genres"].apply(lambda x: bool(genres_lower.intersection(x)))

        if actors:
            actors_lower = set([a.lower() for a in actors])
            mask &= self.df["cast"].apply(lambda x: bool(actors_lower.intersection(x)))

        if director:
            director_lower = director.lower()
            mask &= self.df["crew"].apply(lambda x: x.get("director") == director_lower)

        mask &= (similarities >= similarity_threshold)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_similarities = similarities[valid_indices]
        sorted_idx = np.lexsort((-self.df.iloc[valid_indices]["vote_average"], -valid_similarities))
        sorted_valid_indices = valid_indices[sorted_idx]

        top_indices = sorted_valid_indices[:5]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]

            results.append({
                "title": row["title"].title(),
                "overview": row["overview"],
                "genres": [g.title() for g in row["genres"]],
                "keywords": [k.title() for k in row["keywords"]],
                "cast": [a.title() for a in row["cast"][:4]],
                "director": row["crew"].get("director", "").title(),
                "rating": float(row["vote_average"]),
                "release_year": int(row["release_year"]) if row["release_year"] is not None else None,
                "vote_count": int(row["vote_count"])
            })

        return results


movie_retriever = MovieRetriever()


def search_movies(**kwargs) -> List[Dict]:
    kwargs["max_results"] = 5
    kwargs["page"] = 1
    return movie_retriever.retrieve_movies(**kwargs)


if __name__ == "__main__":
    import time

    start_time = time.time()

    results = search_movies(
        genres=["drama", "inspirational", "slice of life"],
        emotion="heartwarming",
        keyword=["health check", "back pain", "resilience", "facing health concerns", "gardening-related injury"],
        min_rating=7.0,
        min_vote_count=100
    )

    end_time = time.time()

    print(f"Time cost: {end_time - start_time:.3f}s")
    print(f"\nFound {len(results)} similar movies:")

    for i, movie in enumerate(results, 1):
        print(f"\n{i}. {movie['title']}")
        print(f"   Genres: {', '.join(movie['genres'])}")
        print(f"   Director: {movie['director']}")
        if len(movie['overview']) > 150:
            print(f"   Overview: {movie['overview'][:150]}...")
        else:
            print(f"   Overview: {movie['overview']}")