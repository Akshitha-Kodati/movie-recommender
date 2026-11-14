"""Builds model.pkl and similarity.pkl from the TMDB 5000 CSV files.

Run:  python build_model.py
"""
import ast
import os
import pickle
import sys

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def names(text):
    return [d["name"] for d in ast.literal_eval(text)]


def top3_cast(text):
    return [d["name"] for d in ast.literal_eval(text)[:3]]


def director(text):
    return [d["name"] for d in ast.literal_eval(text) if d["job"] == "Director"]


def build(data_dir=None, out_dir=None):
    data_dir = data_dir or os.path.join(BASE_DIR, "data")
    out_dir = out_dir or BASE_DIR
    movies_csv = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_csv = os.path.join(data_dir, "tmdb_5000_credits.csv")
    for f in (movies_csv, credits_csv):
        if not os.path.exists(f):
            sys.exit(f"Missing file: {f}\nPut both TMDB CSV files in the data/ folder.")

    # 1. Load + merge
    movies = pd.read_csv(movies_csv).merge(pd.read_csv(credits_csv), on="title")
    movies = movies.rename(columns={"id": "movie_id"})
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.dropna().drop_duplicates(subset="title").reset_index(drop=True)

    # 2. Parse JSON-like columns
    movies["genres"] = movies["genres"].apply(names)
    movies["keywords"] = movies["keywords"].apply(names)
    movies["cast"] = movies["cast"].apply(top3_cast)
    movies["crew"] = movies["crew"].apply(director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda lst: [i.replace(" ", "") for i in lst])

    # 3. Tags
    movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())
    ps = PorterStemmer()
    movies["tags"] = movies["tags"].apply(lambda t: " ".join(ps.stem(w) for w in t.split()))

    # 4. Vectorize + cosine similarity
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors).astype("float32")

    # 5. Save
    pickle.dump(movies[["movie_id", "title"]], open(os.path.join(out_dir, "model.pkl"), "wb"))
    pickle.dump(similarity, open(os.path.join(out_dir, "similarity.pkl"), "wb"))
    print(f"Saved model.pkl and similarity.pkl ({len(movies)} movies)")


if __name__ == "__main__":
    build()
