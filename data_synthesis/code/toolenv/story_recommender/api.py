import os
import json
import threading
from sentence_transformers import SentenceTransformer, util
import sys

# Keep your original API import path unchanged

try:
    from APIclient import onechatAPIclient
except ImportError:
    pass  # Avoid error if the package is missing in test environments


class StoryRetriever:
    _instance = None
    _lock = threading.Lock()  # Thread-safe singleton lock

    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_resources(*args, **kwargs)
        return cls._instance

    def _init_resources(self, model_path=None, stories_path=None):
        """
        One-time initialization: load model, precompute all story embeddings
        Called only once during first instantiation
        """
        print(f"[Thread {threading.current_thread().name}] Initializing story retriever resources...")

        if model_path is None:
            model_path = "/xxx/all-mpnet-base-v2"
        if stories_path is None:
            stories_path = os.path.join(os.path.dirname(__file__), "stories.json")

        # Initialize API client (for compatibility)
        try:
            self.api_model = onechatAPIclient()
        except NameError:
            self.api_model = None

        # Load torch and handle device safety
        import torch

        # Preserve original default device to avoid breaking external LLM frameworks
        original_device = torch.device("cpu")
        if hasattr(torch, "get_default_device"):
            original_device = torch.get_default_device()
            if original_device.type == "meta":
                torch.set_default_device("cpu")

        try:
            # Force CPU and disable meta device usage
            self.model = SentenceTransformer(
                model_name_or_path='/xxx/all-mpnet-base-v2',
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False}
            )
        finally:
            # Restore original device setting
            if hasattr(torch, "set_default_device") and original_device.type == "meta":
                torch.set_default_device(original_device)

        # Load story database
        if not os.path.exists(stories_path):
            raise FileNotFoundError(f"Stories file not found at {stories_path}")

        with open(stories_path, "r", encoding="utf-8") as f:
            self.stories = json.load(f)

        # Precompute all story embeddings once at startup
        story_texts = [story['content'] for story in self.stories]
        self.story_embeddings = self.model.encode(
            story_texts,
            convert_to_tensor=True,
            device="cpu"
        )

        print(f"[Thread {threading.current_thread().name}] Story retriever ready! Loaded {len(self.stories)} stories.")

    def retrieve_story(self, theme=None, type=None, style=None, user_prompt=None):
        """
        Retrieve top-3 most similar stories based on:
        - Metadata filtering (theme, type, style)
        - Semantic similarity (user prompt)
        """
        if not user_prompt:
            raise ValueError("user_prompt should not be empty")

        # Filter by metadata matching score
        scored_stories = []
        for idx, story in enumerate(self.stories):
            match_score = 0
            if theme and story.get("theme") == theme:
                match_score += 1
            if type and story.get("type") == type:
                match_score += 1
            if style and story.get("style") == style:
                match_score += 1

            scored_stories.append((match_score, idx, story))

        # Keep only highest-score stories
        max_score = max(score for score, _, _ in scored_stories)
        matched = [(idx, story) for score, idx, story in scored_stories if score == max_score]

        # Compute semantic similarity (optimized)
        user_emb = self.model.encode(user_prompt, convert_to_tensor=True, device="cpu")

        results_with_sim = []
        for idx, story in matched:
            # Use precomputed embedding for speed
            story_emb = self.story_embeddings[idx]
            sim = util.cos_sim(user_emb, story_emb).item()
            results_with_sim.append((sim, story))

        # Sort by similarity descending
        results_with_sim.sort(key=lambda x: x[0], reverse=True)

        # Return top 3 stories
        results = [story for sim, story in results_with_sim[:3]]
        return results

# Global singleton instance
story_retriever = StoryRetriever()

def search_stories(theme=None, type=None, style=None, user_prompt=None):
    """Unified external search interface"""
    return story_retriever.retrieve_story(theme, type, style, user_prompt)


if __name__ == "__main__":
    print("=== Story Retrieval Test ===")

    retriever = StoryRetriever()

    stories = retriever.retrieve_story(
        user_prompt="At work today, I realized halfway through my shift that my sweatshirt was inside out. A younger coworker pointed it out, people laughed, and I tried to play it cool, but I felt embarrassed and wanted to disappear—another absent-minded moment that's bothering me.",
        theme="Daily Emotions and Mood",
        type="Fable and Parable",
        style="Warm and Comforting"
    )

    print("\nTop matched stories:")
    for i, s in enumerate(stories):
        print(f"\n[{i+1}] Theme: {s.get('theme')} | Style: {s.get('style')}")
        print(f"Content snippet: {s.get('content')[:200]}...")