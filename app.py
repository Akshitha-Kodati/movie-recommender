import os
import pickle

from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SIM_PATH = os.path.join(BASE_DIR, "similarity.pkl")

# If the pickles don't exist yet, build them from the CSVs.
if not (os.path.exists(MODEL_PATH) and os.path.exists(SIM_PATH)):
    from build_model import build
    build()

app = Flask(__name__)
movies = pickle.load(open(MODEL_PATH, "rb"))
similarity = pickle.load(open(SIM_PATH, "rb"))
TITLES = sorted(movies["title"].tolist())


def recommend(title, n=5):
    matches = movies[movies["title"] == title]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1 : n + 1]
    return [movies.iloc[i].title for i, _ in scores]


@app.route("/")
def home():
    return render_template("index.html", titles=TITLES)


@app.route("/recommend", methods=["POST"])
def recommend_route():
    selected = request.form.get("movie", "")
    recs = recommend(selected)
    return render_template("index.html", titles=TITLES, selected=selected, recommendations=recs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
