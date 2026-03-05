# EmotiSense

EmotiSense is a lightweight, fast, and zero-dependency (aside from Flask) emotion detection web application built with Python. It analyzes text input to determine the underlying emotional tone and visualizes the results seamlessly.

## Features

- **Multi-layered Emotion Detection**: Identifies 6 core emotions (Happy, Sad, Angry, Fear, Surprise, Disgust) plus a Neutral state.
- **Robust Keyword & Phrase Lexicon**: Utilizes a comprehensive, built-in dictionary of keywords and phrases for instant, memory-efficient text analysis.
- **Emoji Support**: Natively parses and scores emojis to capture sentiment in modern text communication (e.g. 😀, 😢, 😡, 🤢).
- **Context Awareness**:
  - **Intensifiers**: Automatically boosts emotion scores when words like "very", "extremely", or "absolutely" are detected near emotional keywords.
  - **Negation Handling**: Accurately reduces emotion scores when keywords are preceded by negations like "not", "never", or "didn't".
- **Real-time Web UI**: A simple, intuitive frontend to input text and view emotion percentage breakdowns and summaries.

## Getting Started

### Prerequisites
- Python 3.x
- Flask

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbhishekYadav2207/emotisense.git
   cd emotisense
   ```

2. Install dependencies:
   ```bash
   pip install flask
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`.

## How It Works
The backend engine tokenizes the user's text input, preserving apostrophes for contractions. It then scans each token against the `EMOTION_LEXICON`. The base score is adjusted dynamically based on nearby negations or intensifiers. Phrase matches and emoji matches provide additional score weighting. Finally, the raw scores are normalized into percentages to declare the dominant emotion and generate an informative analytical summary.

## Branches & Advanced Models
- **`main` branch**: Contains the fast, lightweight, rule-based approach (Keywords, Phrases, Emojis).
- **`ml` branch**: Contains advanced experimental integrations using SQLite databases derived from the NRC Emotion Lexicon, NLTK Lemmatization, and Scikit-Learn Logistic Regression & TF-IDF classifiers for multi-lingual and zero-shot ML detection.

## Contributing
Contributions are always welcome! Feel free to open issues or submit pull requests for additional keywords, phrases, emoji mappings, or UI improvements!
