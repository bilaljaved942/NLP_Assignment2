"""
abstractive_summarization.py

Generates abstractive summaries from extractive summaries using
rule-based POS analysis (no pretrained models).
"""

import json
import os
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Compatibility fix for new NLTK tagger naming
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger_eng")
    except:
        nltk.download("averaged_perceptron_tagger")


# -----------------------------
# CONFIG
# -----------------------------
EXTRACTIVE_PATH = "../summaries/extractive_summaries_v2.json"
OUTPUT_PATH = "../summaries/abstractive_summaries.json"
STOP_WORDS = set(stopwords.words("english"))

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    """Basic cleanup."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ,.\'-]', '', text)
    return text.strip()

def extract_keywords(text):
    """Return nouns, verbs, adjectives, and adverbs."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOP_WORDS and t.isalpha()]
    tagged = pos_tag(tokens)

    keywords = [w for w, tag in tagged if tag.startswith(("NN", "VB", "JJ", "RB"))]
    return keywords

def merge_sentences(sentences):
    """Merge multiple extractive sentences into one fluent abstractive one."""
    all_text = " ".join(sentences)
    all_text = clean_text(all_text)
    keywords = extract_keywords(all_text)

    # Count frequency
    freq = Counter(keywords)
    top_words = [w for w, _ in freq.most_common(10)]

    # Select key sentences that contain top keywords
    key_sentences = []
    for s in sentences:
        if any(kw in s for kw in top_words):
            key_sentences.append(clean_text(s))

    # Merge overlapping or similar phrases
    merged = " ".join(key_sentences)
    merged = re.sub(r'\b(the|a|an)\b\s+\1', r'\1', merged, flags=re.IGNORECASE)
    merged = re.sub(r'\s+', ' ', merged)

    # Optional trimming: keep most relevant part
    merged_tokens = merged.split()
    if len(merged_tokens) > 50:
        merged = " ".join(merged_tokens[:50]) + "..."

    return merged.strip()

# -----------------------------
# MAIN
# -----------------------------
with open(EXTRACTIVE_PATH, "r", encoding="utf-8") as f:
    extractive_data = json.load(f)

abstractive_summaries = []

for doc in extractive_data:
    case_id = doc.get("case_id", "unknown_case")
    extractive_sents = doc.get("extractive_summary", [])

    if not extractive_sents:
        continue

    abstractive = merge_sentences(extractive_sents)

    abstractive_summaries.append({
        "case_id": case_id,
        "extractive_summary": extractive_sents,
        "abstractive_summary": abstractive
    })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(abstractive_summaries, f, indent=2, ensure_ascii=False)

print(f"✅ Abstractive summaries saved to: {OUTPUT_PATH}")
