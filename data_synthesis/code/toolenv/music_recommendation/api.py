from typing import List, Dict, Optional
import sqlite3
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(model_name_or_path='xx/all-mpnet-base-v2', device="cpu")

DB_PATH = os.path.join(os.path.dirname(__file__), "music_database_description.db")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "music_embeddings.npy")
MUSIC_DATA_PATH = os.path.join(os.path.dirname(__file__), "music_data.json")

class MusicQueryTool:
    def __init__(self, db_name: str = DB_PATH):
        self.db_name = db_name
        self.music_data: List[Dict] = []
        self.music_embeddings: np.ndarray = None
        self._load_data_and_embeddings()

    def _load_from_db(self) -> List[Dict]:
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM music")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Database load error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def _generate_embeddings(self, music_data: List[Dict]) -> np.ndarray:
        music_texts = [
            f"{music.get('title', '')} {music.get('artist', '')} {music.get('description', '')} {music.get('keywords', '')}"
            for music in music_data
        ]
        embeddings = model.encode(music_texts, device="cpu")
        return embeddings

    def _save_embeddings_and_data(self):
        np.save(EMBEDDINGS_PATH, self.music_embeddings)
        with open(MUSIC_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.music_data, f, ensure_ascii=False)
        print(f"Embeddings and music data saved to cache")

    def _load_data_and_embeddings(self):
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(MUSIC_DATA_PATH):
            try:
                self.music_embeddings = np.load(EMBEDDINGS_PATH)
                with open(MUSIC_DATA_PATH, 'r', encoding='utf-8') as f:
                    self.music_data = json.load(f)
                print(f"Loaded music data and embeddings from cache (total {len(self.music_data)} songs)")
            except Exception as e:
                print(f"Cache load error: {e}, reloading from database...")
                self.music_data = self._load_from_db()
                self.music_embeddings = self._generate_embeddings(self.music_data)
                self._save_embeddings_and_data()
        else:
            print("No cache found, loading from database and generating embeddings...")
            self.music_data = self._load_from_db()
            if self.music_data:
                self.music_embeddings = self._generate_embeddings(self.music_data)
                self._save_embeddings_and_data()
            else:
                print("No music data found in database")

    def _format_result(self, song_data: Dict, similarity_score: float = None) -> Dict:
        formatted = {
            "id": song_data.get("id", ""),
            "title": song_data.get("title", ""),
            "artist": song_data.get("artist", ""),
            "genre": song_data.get("genre", ""),
            "duration": song_data.get("duration", ""),
            "description": song_data.get("description", ""),
            "url": song_data.get("url", ""),
            "keywords": song_data.get("keywords", "")
        }
        
        return formatted

    def query_by_semantic(self,
                          keywords: Optional[List[str]] = None,
                          artist: Optional[str] = None,
                          genre: Optional[str] = None,
                          limit: int = 5,
                          initial_threshold: float = 0.1,
                          min_threshold: float = 0.05) -> List[Dict]:
        if not keywords or not isinstance(keywords, list) or len(keywords) == 0:
            raise ValueError("keywords must be a non-empty list of strings")

        keyword_embeddings = model.encode(keywords, device="cpu")
        all_similarities = [cosine_similarity([kw_emb], self.music_embeddings)[0] for kw_emb in keyword_embeddings]
        avg_similarities = np.mean(all_similarities, axis=0)
        sorted_indices = np.argsort(avg_similarities)[::-1]

        print(f"Similarity distribution - Max: {avg_similarities.max():.3f}, Min: {avg_similarities.min():.3f}, Mean: {avg_similarities.mean():.3f}")

        filter_strategies = [
            (True, True, "artist and genre filters"),
            (True, False, "only artist filter"),
            (False, True, "only genre filter"),
            (False, False, "no filters")
        ]

        current_threshold = initial_threshold
        
        for use_artist, use_genre, strategy_desc in filter_strategies:
            print(f"\nTrying strategy: {strategy_desc} with threshold {current_threshold:.3f}")
            
            while current_threshold >= min_threshold:
                results = []
                for idx in sorted_indices:
                    song = self.music_data[idx]
                    sim_score = avg_similarities[idx]

                    if sim_score < current_threshold:
                        continue
                    if use_artist and artist and artist.lower() not in song.get("artist", "").lower():
                        continue
                    if use_genre and genre and song.get("genre", "").lower() != genre.lower():
                        continue

                    results.append(self._format_result(song, sim_score))
                    if len(results) >= limit:
                        break

                if results:
                    print(f"Found {len(results)} results with strategy: '{strategy_desc}' and threshold {current_threshold:.3f}")
                    return results

                current_threshold -= 0.02
                print(f"No results with threshold {current_threshold+0.02:.3f}, lowering to {current_threshold:.3f}...")
            
            current_threshold = initial_threshold 
            print(f"No results with strategy: '{strategy_desc}' even at min threshold. Trying next strategy...")

        print(f"All strategies failed. Returning top {limit} most similar songs regardless of filters.")
        top_results = []
        for idx in sorted_indices[:limit]:
            song = self.music_data[idx]
            top_results.append(self._format_result(song, avg_similarities[idx]))
        return top_results

    def format_results_as_string(self, results: List[Dict]) -> str:
        if not results:
            return "No matching songs found."

        result_strings = []
        for i, song in enumerate(results, 1):
            song_info = (
                f"{i}. {song['title']} - {song['artist']}\n"
                f"   Genre: {song['genre']}\n"
                f"   Keywords: {song['keywords'] if song['keywords'] else 'N/A'}\n"
                f"   Description: {song['description']}..." if song['description'] else "   Description: N/A"
            )
            result_strings.append(song_info)
        
        return "\n\n".join(result_strings)

    def get_formatted_results(self,
                              keywords: Optional[List[str]] = None,
                              artist: Optional[str] = None,
                              genre: Optional[str] = None,
                              limit: int = 5,
                              initial_threshold: float = 0,
                              min_threshold: float = 0.1) -> str:
        results = self.query_by_semantic(
            keywords=keywords,
            artist=artist,
            genre=genre,
            limit=limit,
            initial_threshold=initial_threshold,
            min_threshold=min_threshold
        )
        return results

def query_music(artist: Optional[str] = None,
                genre: Optional[str] = None,
                keywords: Optional[List[str]] = None,
                limit: int = 5,
                initial_threshold: float = 0,
                min_threshold: float = 0.1) -> str:
    tool = MusicQueryTool()
    try:
        return tool.get_formatted_results(
            keywords=keywords,
            artist=artist,
            genre=genre,
            limit=limit,
            initial_threshold=initial_threshold,
            min_threshold=min_threshold
        )
    finally:
        pass

if __name__ == "__main__":
    print("=== Test: Get Formatted String Results ===")
    music_results_str = query_music(
        keywords=["relaxing", "piano"],
        limit=5,
        initial_threshold=0.08
    )
    
    print("\n" + "="*80)
    print("Final Formatted Results:")
    print("="*80)
    print(music_results_str)