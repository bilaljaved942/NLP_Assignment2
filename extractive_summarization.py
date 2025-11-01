"""
extractive_summarization_fixed_v2.py

Improved extractive summarizer that merges OCR fragments (e.g., '5.', 'Mr.')
before computing sentence embeddings, and returns top-K sentences in original order.
"""

import numpy as np
import json
import os
from numpy.linalg import norm
from tqdm import tqdm
import re

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_DATA_DIR = "../training_data"
PROCESSED_DIR = "../processed_sentences"
OUTPUT_DIR = "../summaries"
INPUT_FILE = "merged_sentences_nested.json"

TOP_K = 3  # number of sentences to include in summary
EPS = 1e-10
MIN_WORDS_FOR_SENT = 3  # merge sentences with fewer words than this
NUMBERING_RE = re.compile(r'^[\[\(]?\d+[\]\)\.:]?$')  # matches '5', '5.', '(5)', '5:'
ABBREV_RE = re.compile(r'^[A-Za-z]{1,4}\.$')  # "Mr.", "Dr.", "J.", etc.

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL & META INFO
# -----------------------------
with open(f"{TRAIN_DATA_DIR}/meta.json", "r") as f:
    meta = json.load(f)

vocab_size = meta["vocab_size"]

print(f"Loaded meta info: vocab_size={vocab_size}")

# Load trained embeddings (W1 from your CBOW)
W1 = np.load(f"{TRAIN_DATA_DIR}/W1_shared.npy")
print(f"Loaded embeddings with shape: {W1.shape}")

# -----------------------------
# HELPERS
# -----------------------------
def cosine_similarity(vec_a, vec_b):
    denom = (norm(vec_a) * norm(vec_b)) + EPS
    return np.dot(vec_a, vec_b) / denom

def sentence_embedding(sentence, vocab_size, W1):
    """Map words into embedding indices consistently using hash and average them."""
    words = sentence.lower().split()
    vectors = []
    for w in words:
        # map each word deterministically into embedding index space
        idx = hash(w) % vocab_size
        vectors.append(W1[idx])
    if len(vectors) == 0:
        return np.zeros(W1.shape[1], dtype=np.float32)
    return np.mean(vectors, axis=0)

def is_fragment(sent):
    """
    Return True if sentence is likely a fragment:
    - very short (fewer than MIN_WORDS_FOR_SENT words)
    - or just a numbering like '5.' or a small abbreviation like 'Mr.'
    """
    stripped = sent.strip()
    if not stripped:
        return True
    tokens = stripped.split()
    if len(tokens) < MIN_WORDS_FOR_SENT:
        # treat purely numeric/numbering tokens as fragments
        if NUMBERING_RE.match(stripped):
            return True
        # treat single-token abbreviations (Mr., Dr., J.) as fragments
        if len(tokens) == 1 and ABBREV_RE.match(tokens[0]):
            return True
        # also treat things like "No." or "Vol." (two-letter abbrev) as fragments
        if len(tokens) == 1 and tokens[0].endswith('.'):
            # if token length is small, consider fragment
            if len(tokens[0]) <= 4:
                return True
        # otherwise if it's short (< MIN_WORDS_FOR_SENT), mark fragment
        return True
    return False

def merge_fragments(sentences):
    """
    Merge short fragments into previous sentence if possible, otherwise into next.
    Returns a new list of sentences (strings) with fragments merged.
    """
    if not sentences:
        return []

    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i].strip()
        if is_fragment(s):
            # try merge with previous if exists
            if merged:
                merged[-1] = merged[-1].rstrip() + " " + s
            else:
                # no previous; try to merge with next
                if i + 1 < len(sentences):
                    # merge into next by prepending
                    sentences[i+1] = s + " " + sentences[i+1]
                else:
                    # single fragment document -> keep as-is
                    merged.append(s)
            i += 1
        else:
            merged.append(s)
            i += 1
    # remove possible empty strings and strip
    merged = [x.strip() for x in merged if x and x.strip()]
    return merged

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
with open(os.path.join(PROCESSED_DIR, INPUT_FILE), "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents.")

# -----------------------------
# SUMMARIZATION LOOP
# -----------------------------
summaries = []

for doc in tqdm(documents, desc="Summarizing documents"):
    case_id = doc.get("case_id", "unknown_case")
    sentences_dict = doc.get("sentences", {})

    # Convert dict of numbered sentences to list (sorted order)
    ordered_keys = sorted(sentences_dict.keys(), key=lambda x: int(x))
    sentences = [sentences_dict[k] for k in ordered_keys]

    if not sentences:
        continue

    # Merge fragments produced by OCR (fix '5.' 'Mr.' etc.)
    merged_sentences = merge_fragments(sentences)

    if not merged_sentences:
        continue

    # Compute sentence embeddings
    sentence_vectors = [sentence_embedding(sent, vocab_size, W1) for sent in merged_sentences]

    # Document embedding
    doc_vec = np.mean(sentence_vectors, axis=0)

    # Scores and ranking
    scores = [cosine_similarity(v, doc_vec) for v in sentence_vectors]
    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    # Keep top indices but return them in original order for readability
    top_indices_sorted = sorted(ranked_indices)

    summary_sents = [merged_sentences[i] for i in top_indices_sorted]

    summaries.append({
        "case_id": case_id,
        "extractive_summary": summary_sents
    })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_path = os.path.join(OUTPUT_DIR, "extractive_summaries_v2.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print(f"\n✅ Extractive summaries saved to: {output_path}")
