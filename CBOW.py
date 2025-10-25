"""
train_nlm_numpy.py
Train a simple CBOW/Skip-gram neural language model from scratch using NumPy.
"""

import json
import numpy as np
import random
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "../training_data"
EMBEDDING_DIM = 100
LEARNING_RATE = 0.05
EPOCHS = 10
MODEL_TYPE = "cbow"  # or "skipgram"

# -----------------------------
# LOAD DATA
# -----------------------------
with open(f"{DATA_DIR}/meta.json", "r") as f:
    meta = json.load(f)
vocab_size = meta["vocab_size"]

with open(f"{DATA_DIR}/train_pairs.json", "r") as f:
    train_pairs = json.load(f)

print(f"Training {MODEL_TYPE.upper()} model on {len(train_pairs)} samples (vocab={vocab_size})")

# -----------------------------
# INITIALIZE WEIGHTS
# -----------------------------
W1 = np.random.randn(vocab_size, EMBEDDING_DIM) * 0.01  # input -> hidden
W2 = np.random.randn(EMBEDDING_DIM, vocab_size) * 0.01  # hidden -> output

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# -----------------------------
# TRAINING LOOP (CBOW)
# -----------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    random.shuffle(train_pairs)

    for sample in tqdm(train_pairs, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        if MODEL_TYPE == "cbow":
            context_ids = sample["context"]
            target_id = sample["target"]

            # forward pass
            h = np.mean(W1[context_ids], axis=0)  # average context vectors
            u = np.dot(W2.T, h)
            y_pred = softmax(u)

            # loss (negative log likelihood)
            loss = -np.log(y_pred[target_id] + 1e-9)
            total_loss += loss

            # gradient
            y_true = np.zeros(vocab_size)
            y_true[target_id] = 1.0
            e = y_pred - y_true

            # backward
            dW2 = np.outer(h, e)
            dW1 = np.zeros_like(W1)
            for c in context_ids:
                dW1[c] += np.dot(W2, e) / len(context_ids)

            # update weights
            W1 -= LEARNING_RATE * dW1
            W2 -= LEARNING_RATE * dW2

    avg_loss = total_loss / len(train_pairs)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
np.save(f"{DATA_DIR}/W1.npy", W1)
np.save(f"{DATA_DIR}/W2.npy", W2)
print(f"✅ Model saved: W1.npy (embeddings), W2.npy")
