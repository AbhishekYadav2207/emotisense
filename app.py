from flask import Flask, render_template, request, jsonify
import re
import math
import sqlite3
import functools
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import numpy as np
import scipy.sparse as sp

# ML Pipeline
ml_pipeline = None
try:
    ml_pipeline = joblib.load('emotion_model.pkl')
    print("Hybrid ML Pipeline Loaded Successfully!")
except Exception as e:
    print(f"Running without ML Pipeline. Lexicon routing active. ({e})")


# Ensure wordnet is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Single read-only DB connection (safe for threads if check_same_thread=False)
db_conn = sqlite3.connect('lexicon.db', check_same_thread=False)
db_conn.row_factory = sqlite3.Row

@functools.lru_cache(maxsize=100000)
def get_word_emotions(word):
    """Fetch emotion values for a given word with a lightning-fast memory cache."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM words WHERE word = ?", (word,))
    return cursor.fetchone()

# ─── Emotion Lexicon ───────────────────────────────────────────────────────────
EMOTION_LEXICON = {
    "happy": {
        "color": "#F59E0B",
        "emoji": "😄",
        "phrases": [
            "so happy", "very happy", "feel good", "feeling good", "on top of the world",
            "over the moon", "on cloud nine", "best day", "love it", "can't wait",
            "made my day", "great day", "having fun", "good time", "good news"
        ],
        "negation_reduces": True
    },
    "sad": {
        "color": "#3B82F6",
        "emoji": "😢",
        "phrases": [
            "so sad", "very sad", "feel sad", "feeling sad", "fall apart",
            "can't stop crying", "miss you", "miss them", "nothing matters",
            "no hope", "bad news", "worst day", "not okay", "not well"
        ],
        "negation_reduces": True
    },
    "angry": {
        "color": "#EF4444",
        "emoji": "😠",
        "phrases": [
            "so angry", "very angry", "makes me angry", "pissed off", "fed up",
            "had enough", "lost it", "can't stand", "drives me crazy", "makes me mad",
            "not fair", "how dare", "absolutely not", "this is wrong", "enough is enough"
        ],
        "negation_reduces": True
    },
    "fear": {
        "color": "#8B5CF6",
        "emoji": "😨",
        "phrases": [
            "so scared", "very scared", "terribly afraid", "heart racing", "can't breathe",
            "worst case", "what if", "going wrong", "something bad", "feel unsafe",
            "going to happen", "not safe", "be careful", "watch out", "be aware"
        ],
        "negation_reduces": True
    },
    "surprise": {
        "color": "#10B981",
        "emoji": "😲",
        "phrases": [
            "can't believe", "who knew", "didn't expect", "out of nowhere",
            "took me by surprise", "never thought", "didn't see that coming",
            "what a surprise", "blown away"
        ],
        "negation_reduces": False
    },
    "disgust": {
        "color": "#84CC16",
        "emoji": "🤢",
        "phrases": [
            "makes me sick", "turns my stomach", "can't stomach", "so gross",
            "how disgusting", "what a mess", "absolutely vile", "totally gross"
        ],
        "negation_reduces": True
    }
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nobody", "nothing",
                  "nowhere", "neither", "don't", "doesn't", "didn't", "won't",
                  "wouldn't", "couldn't", "can't", "isn't", "aren't", "wasn't",
                  "weren't", "haven't", "hadn't", "shouldn't", "doesn't"}

INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "so": 1.4, "really": 1.4,
                "incredibly": 1.8, "absolutely": 1.7, "utterly": 1.7,
                "terribly": 1.6, "awfully": 1.5, "quite": 1.2, "pretty": 1.1,
                "rather": 1.1, "deeply": 1.5, "truly": 1.4, "genuinely": 1.3}

def tokenize(text):
    """Split text into lowercase word tokens, supporting unicode for multiple languages."""
    return re.findall(r"\w+(?:'\w+)*", text.lower(), re.UNICODE)

def is_negated(tokens, index):
    """Check if a token at `index` is preceded by a negation word within 3 positions."""
    start = max(0, index - 3)
    for k in range(start, index):
        if tokens[k] in NEGATION_WORDS:
            return True
    return False

def get_intensifier(tokens, index):
    """Return multiplier if an intensifier appears within 2 positions before index."""
    start = max(0, index - 2)
    for k in range(start, index):
        if tokens[k] in INTENSIFIERS:
            return INTENSIFIERS[tokens[k]]
    return 1.0

def detect_emotions(text):
    if not text.strip():
        return []

    tokens = tokenize(text)
    text_lower = text.lower()
    scores = {emotion: 0.0 for emotion in EMOTION_LEXICON}

    for emotion, data in EMOTION_LEXICON.items():
        # --- Phrase matching ---
        for phrase in data.get("phrases", []):
            if phrase in text_lower:
                scores[emotion] += 2.5

    # --- Database word matching ---
    for i, token in enumerate(tokens):
        # Fetch directly or attempt lemmatized root word matching
        row = get_word_emotions(token)
        if not row:
            lemmatized = lemmatizer.lemmatize(token)
            if lemmatized != token:
                row = get_word_emotions(lemmatized)
                
        if row:
            multiplier = get_intensifier(tokens, i)
            base_score = 1.0 * multiplier
            
            for emotion in EMOTION_LEXICON:
                # If the DB row indicates an association with this emotion
                if row[emotion]:
                    if EMOTION_LEXICON[emotion]["negation_reduces"] and is_negated(tokens, i):
                        scores[emotion] -= base_score * 0.5
                    elif emotion.lower() in ("sad", "sadness") and row.keys() and "sadness" in row.keys() and row["sadness"]:
                        scores[emotion] += base_score
                    else:
                        scores[emotion] += base_score

    # --- Normalize to percentages ---
    total = sum(max(0.0, s) for s in scores.values())
    results = []

    if total == 0:
        return [{
            "emotion": "Neutral",
            "score": 1.0,
            "percentage": 100,
            "color": "#9CA3AF",
            "emoji": "😐",
            "label": "Neutral",
            "dominant": True
        }]

    for emotion, score in scores.items():
        if score > 0:
            pct = round((score / total) * 100, 1)
            results.append({
                "emotion": emotion.capitalize(),
                "score": round(score, 3),
                "percentage": pct,
                "color": EMOTION_LEXICON[emotion]["color"],
                "emoji": EMOTION_LEXICON[emotion]["emoji"],
                "label": emotion.capitalize(),
                "dominant": False
            })

    results.sort(key=lambda x: x["percentage"], reverse=True)

    if results:
        results[0]["dominant"] = True

    return results

def get_analysis_summary(results, text):
    if not results:
        return "No significant emotion detected."

    word_count = len(text.split())
    dominant = results[0]

    summaries = {
        "Happy": f"The text radiates positivity with {dominant['percentage']}% happy sentiment.",
        "Sad": f"A melancholic tone pervades the text at {dominant['percentage']}%.",
        "Angry": f"Strong anger signals detected — {dominant['percentage']}% intensity.",
        "Fear": f"Anxiety and fear dominate this text at {dominant['percentage']}%.",
        "Surprise": f"The text expresses genuine surprise or astonishment ({dominant['percentage']}%).",
        "Disgust": f"Strong aversion and disgust characterize this text ({dominant['percentage']}%).",
        "Neutral": "The text appears emotionally neutral or ambiguous."
    }

    return summaries.get(dominant["emotion"], f"Primary emotion: {dominant['emotion']} ({dominant['percentage']}%)")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    lexicon_results = detect_emotions(text)
    
    if ml_pipeline:
        EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]
        vec = {e: 0.0 for e in EMOTIONS}
        for r in lexicon_results:
            vec[r['label']] = r['score']
            
        lex_features = np.array([[vec[e] for e in EMOTIONS]])
        
        vectorizer = ml_pipeline['vectorizer']
        clf = ml_pipeline['clf']
        
        tfidf_features = vectorizer.transform([text])
        combined_features = sp.hstack((tfidf_features, lex_features), format='csr')
        
        probas = clf.predict_proba(combined_features)[0]
        
        mapping = {
            0: "Sad", 1: "Happy", 2: "Happy", 3: "Angry", 4: "Fear", 5: "Surprise"
        }
        
        prob_dict = {"Happy": 0.0, "Sad": 0.0, "Angry": 0.0, "Fear": 0.0, "Surprise": 0.0}
        for c_idx, p in zip(clf.classes_, probas):
            lbl = mapping.get(c_idx, "Neutral")
            if lbl != "Neutral":
                prob_dict[lbl] += p
                
        results = []
        for lbl, p in prob_dict.items():
            if p > 0.05:  # threshold
                results.append({
                    "emotion": lbl,
                    "score": round(p, 3),
                    "percentage": round(p * 100, 1),
                    "color": EMOTION_LEXICON[lbl.lower()]["color"],
                    "emoji": EMOTION_LEXICON[lbl.lower()]["emoji"],
                    "label": lbl,
                    "dominant": False
                })
        results.sort(key=lambda x: x["percentage"], reverse=True)
        if results:
            results[0]["dominant"] = True
        elif not results and lexicon_results:
            results = lexicon_results
    else:
        results = lexicon_results

    summary = get_analysis_summary(results, text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text.strip()))

    return jsonify({
        "results": results,
        "summary": summary,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "text": text
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)