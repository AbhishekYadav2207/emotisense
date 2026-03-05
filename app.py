from flask import Flask, render_template, request, jsonify
import re
import math

app = Flask(__name__)

# ─── Emoji to Emotion Map ────────────────────────────────────────────────────
# A simple mapping of common emojis to your emotion categories.
# You can expand this later with a more comprehensive dataset (e.g., EmojiNet).
EMOJI_MAP = {
    # Happy / Joy
    "😀": "happy", "😄": "happy", "😂": "happy", "🤣": "happy", "😊": "happy",
    "😁": "happy", "🥰": "happy", "😍": "happy", "🤩": "happy", "😘": "happy",
    "😗": "happy", "😚": "happy", "😙": "happy", "😋": "happy", "😛": "happy",
    "😜": "happy", "🤪": "happy", "😝": "happy", "🤑": "happy", "🤗": "happy",
    "🤭": "happy", "🤫": "happy", "🤔": "happy", "🤐": "happy", "🤨": "happy",
    "😏": "happy",
    # Sadness
    "😢": "sad", "😭": "sad", "😿": "sad", "😔": "sad", "😞": "sad", "😟": "sad",
    # Anger
    "😠": "angry", "😡": "angry", "🤬": "angry", "😤": "angry",
    # Fear
    "😨": "fear", "😰": "fear", "😱": "fear", "😥": "fear", "😓": "fear",
    # Surprise
    "😲": "surprise", "😮": "surprise", "😯": "surprise", "😳": "surprise",
    # Disgust
    "🤢": "disgust", "🤮": "disgust", "🤧": "disgust", "😷": "disgust",
    # Neutral / Other (you can map these as you like)
    "😐": None,      # ignore or treat as neutral
    "😑": None,
    "😶": None
}

# ─── Emotion Lexicon (unchanged) ─────────────────────────────────────────────
EMOTION_LEXICON = {
    "happy": {
        "color": "#F59E0B",
        "emoji": "😄",
        "keywords": [
            "happy", "joy", "joyful", "excited", "wonderful", "fantastic", "great",
            "love", "lovely", "amazing", "awesome", "delighted", "cheerful", "glad",
            "pleased", "thrilled", "ecstatic", "blissful", "elated", "content",
            "satisfied", "grateful", "thankful", "enjoy", "enjoyed", "enjoying",
            "smile", "smiling", "laugh", "laughing", "fun", "celebrate", "celebrating",
            "celebration", "good", "excellent", "superb", "brilliant", "marvelous",
            "terrific", "splendid", "magnificent", "perfect", "beautiful", "lucky",
            "blessed", "optimistic", "hopeful", "positive", "radiant", "gleeful",
            "euphoric", "jubilant", "proud", "sunny", "warm", "bright", "lively",
            "playful", "cheerful", "merry", "jolly", "festive", "vibrant", "wonderful",
            "overjoyed", "exhilarated", "beaming", "thriving", "flourishing"
        ],
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
        "keywords": [
            "sad", "unhappy", "depressed", "depression", "miserable", "heartbroken",
            "grief", "grieve", "grieving", "sorrow", "sorrowful", "melancholy",
            "gloomy", "downcast", "dejected", "despondent", "disheartened", "hopeless",
            "lonely", "alone", "isolated", "abandoned", "rejected", "hurt", "pain",
            "suffering", "suffer", "cry", "crying", "tears", "weeping", "sob",
            "sobbing", "miss", "missing", "lost", "loss", "mourn", "mourning",
            "regret", "disappointed", "disappointment", "unfortunate", "tragic",
            "tragedy", "broken", "devastated", "crushed", "shattered", "empty",
            "numb", "helpless", "worthless", "useless", "failure", "pathetic",
            "terrible", "awful", "horrible", "dreadful", "wretched", "dismal",
            "bleak", "dark", "heavy", "tired", "exhausted", "drained", "down",
            "low", "blue", "gloomy", "forlorn", "desolate", "heartache", "dismay"
        ],
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
        "keywords": [
            "angry", "anger", "furious", "rage", "raging", "mad", "outraged",
            "irate", "livid", "enraged", "infuriated", "irritated", "annoyed",
            "frustrated", "frustration", "aggravated", "hostile", "aggressive",
            "bitter", "resentful", "resentment", "hatred", "hate", "despise",
            "loathe", "disgusted", "disgust", "repulsed", "appalled", "offended",
            "insulted", "provoked", "threatened", "violent", "fierce", "vicious",
            "brutal", "cruel", "nasty", "mean", "rude", "obnoxious", "hideous",
            "abominable", "contempt", "scorn", "wrath", "fury", "temper",
            "explode", "exploding", "boiling", "seething", "steaming", "fuming",
            "snap", "snapping", "yell", "yelling", "scream", "screaming",
            "fight", "fighting", "attack", "attacking", "revenge", "punish",
            "ridiculous", "absurd", "unacceptable", "intolerable", "stupid", "idiot"
        ],
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
        "keywords": [
            "fear", "afraid", "scared", "terrified", "terror", "horror", "horrified",
            "frightened", "fright", "panic", "panicking", "anxious", "anxiety",
            "nervous", "worried", "worry", "dread", "dreading", "apprehensive",
            "uneasy", "tense", "trembling", "shaking", "tremor", "phobia",
            "paranoid", "paranoia", "insecure", "unsafe", "danger", "dangerous",
            "threat", "threatened", "vulnerable", "helpless", "powerless",
            "nightmare", "nightmares", "creepy", "eerie", "ominous", "sinister",
            "suspicious", "alarmed", "alert", "startled", "shock", "shocked",
            "daunting", "overwhelming", "intimidated", "intimidating", "petrified",
            "frozen", "paralyzed", "hesitant", "reluctant", "uncertain", "unsure",
            "doubt", "doubting", "hesitate", "hesitation", "worry", "concern",
            "concerned", "disturbed", "haunted", "haunting", "foreboding", "ominous"
        ],
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
        "keywords": [
            "surprised", "surprise", "astonished", "astonishment", "amazed", "amazement",
            "stunned", "shocked", "unexpected", "unbelievable", "incredible", "wow",
            "whoa", "omg", "no way", "really", "seriously", "unbelievable", "astounded",
            "flabbergasted", "dumbfounded", "bewildered", "baffled", "speechless",
            "remarkable", "extraordinary", "unprecedented", "never expected",
            "caught off guard", "out of nowhere", "suddenly", "suddenly", "bizarre",
            "strange", "odd", "weird", "unusual", "peculiar", "curious", "wonder",
            "wondering", "mystery", "mysterious", "revelation", "discover", "found out"
        ],
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
        "keywords": [
            "disgusting", "disgusted", "disgust", "gross", "revolting", "repulsive",
            "nauseating", "nausea", "sick", "sickening", "vile", "filthy", "dirty",
            "nasty", "repelled", "appalled", "horrified", "abhorrent", "despicable",
            "loathsome", "detestable", "repugnant", "odious", "yuck", "eww", "ugh",
            "awful", "terrible", "dreadful", "abysmal", "atrocious", "foul",
            "putrid", "rotten", "corrupt", "toxic", "poisonous", "contaminated"
        ],
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
    """Split text into lowercase word tokens, keeping apostrophes for contractions."""
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())

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

    # --- Keyword matching (unchanged) ---
    for emotion, data in EMOTION_LEXICON.items():
        keyword_set = set(data["keywords"])
        for i, token in enumerate(tokens):
            if token in keyword_set:
                multiplier = get_intensifier(tokens, i)
                base_score = 1.0 * multiplier
                if data["negation_reduces"] and is_negated(tokens, i):
                    scores[emotion] -= base_score * 0.5
                else:
                    scores[emotion] += base_score

        # --- Phrase matching (unchanged) ---
        for phrase in data["phrases"]:
            if phrase in text_lower:
                scores[emotion] += 2.5

    # --- NEW: Emoji contribution ---
    for char in text:
        if char in EMOJI_MAP:
            target_emotion = EMOJI_MAP[char]
            if target_emotion:  # ignore mapped None (neutral)
                # Add a base score of 1.0 per emoji (you can adjust weight)
                scores[target_emotion] += 1.0

    # --- Normalize to percentages (unchanged) ---
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

    results = detect_emotions(text)
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