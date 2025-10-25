"""
sentence_segmentation.py

Usage:
- Adjust INPUT_JSON or INPUT_DIR depending on whether you have a single combined JSON
  or individual per-document JSON files.
- This script will create `processed_sentences/` with one JSON per case and a combined file.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import re
import nltk
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
# Either set INPUT_JSON to the combined file (preferred) or INPUT_DIR to a folder with per-case JSONs.
INPUT_JSON = "../processed_data_ocr_parallel/merged_all.json"   # update path if needed
INPUT_DIR = None  # e.g., "processed_data_ocr_safe"  (set to None to use INPUT_JSON)
OUTPUT_DIR = "../processed_sentences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure punkt tokenizer is available
nltk.download("punkt")

# -----------------------
# Helpers
# -----------------------
def normalize_whitespace(text: str) -> str:
    # collapse multiple spaces/newlines to single space, but keep sentence breaks intact for tokenization
    # here we replace multiple spaces and tabs with single space
    return re.sub(r"[ \t]+", " ", text).strip()

def split_into_sentences_with_offsets(text: str) -> List[Dict]:
    """
    Use nltk.sent_tokenize to split text into sentences and compute start/end char offsets.
    Returns list of dicts: [{"text":sentence, "start":int, "end":int}, ...]
    """
    sentences = []
    if not text:
        return sentences

    text = normalize_whitespace(text)
    # get sentence boundaries using nltk
    spans = list(nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(text))
    for i, (start, end) in enumerate(spans):
        sent_text = text[start:end].strip()
        if sent_text == "":
            continue
        sentences.append({
            "sentence_id": i + 1,
            "text": sent_text,
            "start_char": start,
            "end_char": end
        })
    return sentences

# -----------------------
# Load input documents
# -----------------------
def load_documents_from_combined(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs

def load_documents_from_dir(dir_path: str) -> List[Dict]:
    docs = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            p = os.path.join(root, fn)
            with open(p, "r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                    docs.append(doc)
                except Exception as e:
                    print(f"[WARN] Could not load {p}: {e}")
    return docs

# -----------------------
# Main processing
# -----------------------
def process_documents(docs: List[Dict], output_dir: str):
    all_sentence_records = []
    for doc in tqdm(docs, desc="Segmenting documents"):
        case_id = doc.get("case_id") or Path(doc.get("pdf_source","")).stem
        raw_text = doc.get("ocr_text", "")
        sentences = split_into_sentences_with_offsets(raw_text)

        # Build record to save
        out_record = {
            "case_id": case_id,
            "pdf_source": doc.get("pdf_source", ""),
            "num_sentences": len(sentences),
            "sentences": sentences
        }

        # Save per-document JSON (use safe filename)
        safe_dir = os.path.join(output_dir, os.path.dirname(doc.get("pdf_source","")))
        os.makedirs(safe_dir, exist_ok=True)
        out_path = os.path.join(safe_dir, f"{case_id}.sentences.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_record, f, indent=2, ensure_ascii=False)

        # also append to combined list (flattened)
        for s in sentences:
            all_sentence_records.append({
                "case_id": case_id,
                "pdf_source": doc.get("pdf_source",""),
                "sentence_id": s["sentence_id"],
                "text": s["text"],
                "start_char": s["start_char"],
                "end_char": s["end_char"]
            })

    # Save combined sentences file
    combined_path = os.path.join(output_dir, "all_sentences.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_sentence_records, f, indent=2, ensure_ascii=False)

    print(f"Saved per-document sentence JSONs under: {output_dir}")
    print(f"Saved combined sentences file: {combined_path}")


if __name__ == "__main__":
    # Load docs
    if INPUT_DIR:
        docs = load_documents_from_dir(INPUT_DIR)
    else:
        docs = load_documents_from_combined(INPUT_JSON)

    print(f"Loaded {len(docs)} documents. Starting sentence segmentation...")
    process_documents(docs, OUTPUT_DIR)
