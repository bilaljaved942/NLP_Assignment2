"""
generate_training_data.py
Create CBOW or Skip-gram training pairs from tokenized text.
"""

import json
import random
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
TOKENIZED_FILE = "../processed_sentences/tokenized_cases.json"
VOCAB_FILE = "../processed_sentences/vocab.json"
OUTPUT_DIR = "../training_data"
MODEL_TYPE = "cbow"   # "cbow" or "skipgram"
WINDOW_SIZE = 2
MIN_SENT_LEN = 3

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD
# -------------------------------
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)
word2id = vocab
vocab_size = len(vocab)

with open(TOKENIZED_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Loaded {len(docs)} cases, vocab size={vocab_size}")

# -------------------------------
# HELPER
# -------------------------------
def encode_token(token):
    return word2id.get(token, word2id["<UNK>"])

# -------------------------------
# BUILD TRAINING PAIRS
# -------------------------------
pairs = []

for doc in tqdm(docs, desc="Generating pairs"):
    for sid, tokens in doc["tokenized_sentences"].items():
        if len(tokens) < MIN_SENT_LEN:
            continue
        token_ids = [encode_token(t) for t in tokens]

        for i, target in enumerate(token_ids):
            # context window
            start = max(0, i - WINDOW_SIZE)
            end = min(len(token_ids), i + WINDOW_SIZE + 1)
            context = [token_ids[j] for j in range(start, end) if j != i]

            if MODEL_TYPE == "cbow":
                pairs.append({"context": context, "target": target})
            elif MODEL_TYPE == "skipgram":
                for ctx in context:
                    pairs.append({"center": target, "target": ctx})

print(f"Generated {len(pairs)} training pairs for {MODEL_TYPE.upper()} model")

# -------------------------------
# SPLIT (train/valid/test)
# -------------------------------
random.shuffle(pairs)
n = len(pairs)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_pairs = pairs[:train_split]
valid_pairs = pairs[train_split:val_split]
test_pairs = pairs[val_split:]

with open(f"{OUTPUT_DIR}/train_pairs.json", "w", encoding="utf-8") as f:
    json.dump(train_pairs, f, indent=2)
with open(f"{OUTPUT_DIR}/valid_pairs.json", "w", encoding="utf-8") as f:
    json.dump(valid_pairs, f, indent=2)
with open(f"{OUTPUT_DIR}/test_pairs.json", "w", encoding="utf-8") as f:
    json.dump(test_pairs, f, indent=2)

meta = {
    "model_type": MODEL_TYPE,
    "window_size": WINDOW_SIZE,
    "vocab_size": vocab_size,
    "train_size": len(train_pairs),
    "valid_size": len(valid_pairs),
    "test_size": len(test_pairs)
}
with open(f"{OUTPUT_DIR}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Data saved in '{OUTPUT_DIR}' with {len(train_pairs)} training samples")
