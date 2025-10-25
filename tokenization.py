"""
tokenize_and_normalize.py

Performs:
1. Tokenization of sentences
2. Text normalization (lowercasing, punctuation cleanup)
3. Vocabulary building
"""

import json
import os
import re
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------------------
# CONFIG
# -------------------------------
INPUT_FILE = "../processed_sentences/merged_sentences_nested.json"
OUTPUT_FILE = "../processed_sentences/tokenized_cases.json"
VOCAB_FILE = "../processed_sentences/vocab.json"

# download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

STOPWORDS = set(stopwords.words("english"))

# -------------------------------
# CLEANING + TOKENIZATION
# -------------------------------
def normalize_text(text: str) -> str:
    """
    Normalize text: remove special characters, fix spacing, lowercase.
    """
    text = text.lower()
    text = re.sub(r"[\n\t]", " ", text)              # remove newlines/tabs
    text = re.sub(r"[^a-z0-9\s]", " ", text)         # keep alphanumeric + space
    text = re.sub(r"\s+", " ", text).strip()         # collapse multiple spaces
    return text


def tokenize_sentence(sentence: str, remove_stopwords=True):
    text = normalize_text(sentence)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# -------------------------------
# MAIN PROCESS
# -------------------------------
def process_documents(input_file, output_file, vocab_file):
    with open(input_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_tokens = []
    tokenized_docs = []

    for doc in tqdm(docs, desc="Tokenizing cases"):
        case_id = doc.get("case_id")
        sentences = doc.get("sentences", {})
        tokenized_sentences = {}

        for sid, sent_text in sentences.items():
            tokens = tokenize_sentence(sent_text)
            tokenized_sentences[sid] = tokens
            all_tokens.extend(tokens)

        tokenized_docs.append({
            "case_id": case_id,
            "tokenized_sentences": tokenized_sentences
        })

    # save tokenized data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokenized_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Tokenized data saved to {output_file}")

    # -------------------------------
    # Build vocabulary
    # -------------------------------
    freq = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    # apply frequency cutoff (optional)
    MIN_FREQ = 3
    for word, count in freq.items():
        if count >= MIN_FREQ:
            vocab[word] = idx
            idx += 1

    # save vocab
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ Vocabulary saved to {vocab_file} ({len(vocab)} tokens kept)")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    process_documents(INPUT_FILE, OUTPUT_FILE, VOCAB_FILE)
