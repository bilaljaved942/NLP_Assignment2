"""
train_nlm_multiproc.py

Multiprocessing CBOW training using shared memory (Hogwild-style).
Keeps full softmax and your original gradient updates.

Notes:
- Requires Python 3.8+ (multiprocessing.shared_memory).
- This is still NumPy-only (no PyTorch/CuPy), but uses multiple processes.
- Use NUM_WORKERS carefully (each worker will compute full softmax for its samples).
"""

import json
import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import math
import time

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "../training_data"
EMBEDDING_DIM = 100
LEARNING_RATE = 0.05
EPOCHS = 10
MODEL_TYPE = "cbow"  # keep same
NUM_WORKERS = max(1, 4)  # tune this
CHUNK_SIZE = 5000  # how many samples per worker task; tune to balance overhead
RANDOM_SEED = 42

# -----------------------------
# UTILS: softmax
# -----------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# -----------------------------
# LOAD DATA (main process)
# -----------------------------
with open(f"{DATA_DIR}/meta.json", "r") as f:
    meta = json.load(f)
vocab_size = meta["vocab_size"]

with open(f"{DATA_DIR}/train_pairs.json", "r") as f:
    train_pairs = json.load(f)

n_samples = len(train_pairs)
print(f"Loaded {n_samples:,} training pairs. Vocab size={vocab_size}. Workers={NUM_WORKERS}")

# -----------------------------
# Create shared memory for weights
# -----------------------------
rng = np.random.default_rng(RANDOM_SEED)
W1_init = (rng.normal(0, 0.01, (vocab_size, EMBEDDING_DIM))).astype(np.float32)
W2_init = (rng.normal(0, 0.01, (EMBEDDING_DIM, vocab_size))).astype(np.float32)

# Create shared memory blocks
shm_W1 = shared_memory.SharedMemory(create=True, size=W1_init.nbytes)
shm_W2 = shared_memory.SharedMemory(create=True, size=W2_init.nbytes)

# Create numpy views on those shared blocks in main process and copy init values
W1_shared = np.ndarray(W1_init.shape, dtype=np.float32, buffer=shm_W1.buf)
W2_shared = np.ndarray(W2_init.shape, dtype=np.float32, buffer=shm_W2.buf)
W1_shared[:] = W1_init[:]
W2_shared[:] = W2_init[:]

# For safety: store shapes and dtype for workers to reconstruct views
SHARED_META = {
    "W1_shape": W1_init.shape,
    "W2_shape": W2_init.shape,
    "dtype": "float32",
    "shm_W1_name": shm_W1.name,
    "shm_W2_name": shm_W2.name
}

# -----------------------------
# Worker function
# -----------------------------
# We will pass chunks of indices into the dataset. Each worker reattaches to shared memory,
# processes its chunk sequentially, and performs in-place updates to the shared arrays.


def worker_init(shared_meta):
    """Initializer for each worker process to attach to shared memory."""
    global W1, W2, W1_shape, W2_shape, dtype
    W1_shape = tuple(shared_meta["W1_shape"])
    W2_shape = tuple(shared_meta["W2_shape"])
    dtype = np.float32

    existing_shm_W1 = shared_memory.SharedMemory(name=shared_meta["shm_W1_name"])
    existing_shm_W2 = shared_memory.SharedMemory(name=shared_meta["shm_W2_name"])
    # create numpy views (these refer to the shared memory)
    W1 = np.ndarray(W1_shape, dtype=dtype, buffer=existing_shm_W1.buf)
    W2 = np.ndarray(W2_shape, dtype=dtype, buffer=existing_shm_W2.buf)

    # Workers will hold reference to shared memory objects by assigning to globals
    # Keep the SharedMemory objects alive by attaching them to the global module
    global _worker_shm_W1, _worker_shm_W2
    _worker_shm_W1 = existing_shm_W1
    _worker_shm_W2 = existing_shm_W2

    # small RNG per worker
    global worker_rng
    worker_rng = np.random.default_rng(os.getpid() ^ RANDOM_SEED)


def process_chunk(args):
    """
    args = (start_idx, end_idx, epoch, shared_meta)
    Processes train_pairs[start_idx:end_idx] and updates W1, W2 in shared memory.
    Returns (processed_count, cumulative_loss) for logging.
    """
    start_idx, end_idx, epoch = args
    global W1, W2, worker_rng

    processed = 0
    total_loss = 0.0

    # Local alias for speed
    W1_local = W1
    W2_local = W2

    # For each sample in the chunk
    for idx in range(start_idx, end_idx):
        sample = train_pairs[idx]
        if MODEL_TYPE != "cbow":
            continue  # this script keeps CBOW logic to remain consistent

        context_ids = sample["context"]
        target_id = sample["target"]

        # forward
        # compute hidden vector h (mean of context embeddings)
        h = np.mean(W1_local[context_ids], axis=0)  # shape (emb_dim,)

        # compute scores for all vocab: u = W2^T dot h  => shape (vocab_size,)
        # W2_local shape is (emb_dim, vocab_size), so dot product:
        u = np.dot(W2_local.T, h)  # shape (vocab_size,)
        # softmax
        maxu = np.max(u)
        e_x = np.exp(u - maxu)
        y_pred = e_x / np.sum(e_x)

        # loss
        loss = -math.log(y_pred[target_id] + 1e-12)
        total_loss += loss

        # gradient (y_pred - y_true)
        y_pred[target_id] -= 1.0  # now y_pred is (y_pred - y_true)

        # dW2 = outer(h, e)  => shape (emb_dim, vocab_size)
        # update W2 in-place (Hogwild)
        # W2_local -= LEARNING_RATE * dW2
        # equivalent: for j in vocab: W2[:, j] -= lr * e[j] * h
        # vector update:
        W2_local -= LEARNING_RATE * np.outer(h, y_pred)

        # dW1: contribution only for context rows
        # d_hidden = W2 * e  (sum over vocab) => shape (emb_dim,)
        d_hidden = np.dot(W2_local, y_pred)  # shape (emb_dim,)
        # propagate to context word rows (average)
        for c in context_ids:
            W1_local[c] -= LEARNING_RATE * (d_hidden / len(context_ids))

        processed += 1

    return processed, total_loss


# -----------------------------
# Training loop (main process coordinates chunks)
# -----------------------------
def chunks_indices(n_samples, chunk_size):
    for i in range(0, n_samples, chunk_size):
        yield (i, min(i + chunk_size, n_samples))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start_time = time.time()

    # Precompute chunk slices to map to tasks per epoch
    base_chunks = list(chunks_indices(n_samples, CHUNK_SIZE))

    # Create worker pool with initializer to attach shared memory arrays
    pool = mp.Pool(processes=NUM_WORKERS, initializer=worker_init, initargs=(SHARED_META,))

    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
            random.shuffle(train_pairs)  # shuffle global list in main process

            # Recompute chunk boundaries given shuffle: we will treat train_pairs as shuffled in place
            # But since we will only pass indices, the indices refer to the shuffled order already
            tasks = []
            for (s, e) in base_chunks:
                tasks.append((s, e, epoch))

            # Map tasks to worker processes (imap_unordered gives streaming results)
            processed_total = 0
            loss_total = 0.0

            for processed, loss in tqdm(pool.imap_unordered(process_chunk, tasks), total=len(tasks)):
                processed_total += processed
                loss_total += loss

            avg_loss = loss_total / processed_total if processed_total > 0 else float("nan")
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} done | Processed {processed_total:,} samples | Avg loss {avg_loss:.4f} | Time {epoch_time:.1f}s")

    finally:
        # close pool and cleanup shared memory
        pool.close()
        pool.join()
        # read back shared arrays into main process's numpy arrays
        W1_final = np.ndarray(SHARED_META["W1_shape"], dtype=np.float32, buffer=shm_W1.buf).copy()
        W2_final = np.ndarray(SHARED_META["W2_shape"], dtype=np.float32, buffer=shm_W2.buf).copy()

        # Save final matrices
        np.save(f"{DATA_DIR}/W1_shared.npy", W1_final)
        np.save(f"{DATA_DIR}/W2_shared.npy", W2_final)

        # Unlink shared memory
        shm_W1.close()
        shm_W1.unlink()
        shm_W2.close()
        shm_W2.unlink()

        elapsed = time.time() - start_time
        print(f"\nTraining complete. Total time: {elapsed:.1f}s")
        print(f"Saved W1_shared.npy and W2_shared.npy")
