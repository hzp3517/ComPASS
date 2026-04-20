import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

# load model
model = SentenceTransformer(model_name_or_path='/xxx/.../all-mpnet-base-v2', device="cpu")

# paths
wock_path = os.path.join(os.path.dirname(__file__), "wocka.json")
joke_embeddings_path = os.path.join(os.path.dirname(__file__), "joke_embeddings.npy")
joke_texts_path = os.path.join(os.path.dirname(__file__), "joke_texts.json")


def preprocess_and_cache():
    with open(wock_path, 'r') as f:
        jokes_data = json.load(f)

    joke_texts = [f"{joke['title']} {joke['body']}" for joke in jokes_data]
    joke_embeddings = model.encode(joke_texts, device="cpu")

    np.save(joke_embeddings_path, joke_embeddings)
    with open(joke_texts_path, 'w') as f:
        json.dump(joke_texts, f)

    return jokes_data, joke_texts, joke_embeddings


if os.path.exists(joke_embeddings_path) and os.path.exists(joke_texts_path):
    with open(wock_path, 'r') as f:
        jokes_data = json.load(f)
    with open(joke_texts_path, 'r') as f:
        joke_texts = json.load(f)
    joke_embeddings = np.load(joke_embeddings_path)
else:
    jokes_data, joke_texts, joke_embeddings = preprocess_and_cache()


def search_jokes(description=None, category=None, max_results=5, similarity_threshold=0.3):
    def _search(use_category_filter=True, use_similarity_threshold=True):
        if description:
            description_embedding = model.encode([description], device="cpu")
            similarities = cosine_similarity(description_embedding, joke_embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]

            if use_similarity_threshold:
                filtered_indices = [i for i in sorted_indices if similarities[i] >= similarity_threshold]
            else:
                filtered_indices = sorted_indices
        else:
            filtered_indices = range(len(jokes_data))

        results = []
        for idx in filtered_indices:
            joke = jokes_data[idx]

            if use_category_filter and category:
                if category.lower() not in joke["category"].lower():
                    continue

            result_joke = joke.copy()

            result_joke["title"] = result_joke["title"].replace("\n", " ").replace("\r", " ")
            result_joke["body"] = result_joke["body"].replace("\n", " ").replace("\r", " ")

            results.append(result_joke)

            if len(results) >= max_results:
                break

        return results

    first_results = _search(use_category_filter=True, use_similarity_threshold=False)

    if not first_results and description:
        second_results = _search(use_category_filter=False, use_similarity_threshold=False)
        return second_results[:max_results] if second_results else []

    return first_results[:max_results] if first_results else []


def display_jokes(jokes):
    if not jokes:
        print("No matching jokes found.")
        return

    print(f"Found {len(jokes)} matching jokes:\n")
    for i, joke in enumerate(jokes, 1):
        print(f"Joke {i}:")
        print(f"Title: {joke['title']}")
        print(f"Category: {joke['category']}")
        print(f"Content: {joke['body']}\n")
        print("-" * 50)

def test_search_function():
    print("=== Joke search test ===")
    print("=" * 50 + "\n")

    print("Test 1: keyword 'computer'")
    results = search_jokes(description="computer", category="At Work")
    print(f"Returned {len(results)} jokes")
    print(results)


if __name__ == "__main__":
    test_search_function()