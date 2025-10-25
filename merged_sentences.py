"""
merge_sentences_nested.py

Merges per-document sentence JSONs into one combined file
with nested sentence structure.

Input:
  processed_sentences/
      supreme court judgements/*.sentences.json
      shariah court judgements/*.sentences.json

Output:
  merged_sentences_nested.json
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "../processed_sentences"       # path to root folder with subdirs
OUTPUT_FILE = "../merged_sentences_nested.json"

# -----------------------------
# MERGE FUNCTION
# -----------------------------
def merge_sentence_jsons(input_dir: str, output_path: str):
    merged_docs = []

    # Walk through subdirectories and collect sentence JSON files
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.endswith(".sentences.json"):
                continue

            file_path = os.path.join(root, fn)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Could not read {file_path}: {e}")
                continue

            case_id = data.get("case_id", Path(fn).stem)
            pdf_source = data.get("pdf_source", "")
            sentences = data.get("sentences", [])

            # Convert list of sentences into nested dict {id: text}
            nested_sentences = {
                str(s["sentence_id"]): s["text"] for s in sentences if s.get("text")
            }

            merged_docs.append({
                "case_id": case_id,
                "pdf_source": pdf_source,
                "num_sentences": len(nested_sentences),
                "sentences": nested_sentences
            })

    # Save combined file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(merged_docs)} documents into {output_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    merge_sentence_jsons(INPUT_DIR, OUTPUT_FILE)
