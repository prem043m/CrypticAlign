# Load users.csv
# Combine profile text
# Clean text
# Return processed text

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)