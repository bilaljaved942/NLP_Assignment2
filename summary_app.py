"""
summary_app.py

Streamlit app (Bonus 10 Marks)
-----------------------------------
Compares Extractive and Abstractive Summaries using BLEU & ROUGE metrics,
and displays OCR text + highlighted summaries.
"""

import streamlit as st
import json
import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# -----------------------------
# SETUP
# -----------------------------
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="Summary Comparison App", layout="wide")

TRAIN_DATA_DIR = "../training_data"
PROCESSED_DIR = "../processed_sentences"
SUMMARY_DIR = "../summaries"

OCR_FILE = os.path.join(PROCESSED_DIR, "merged_sentences_nested.json")
EXTRACTIVE_FILE = os.path.join(SUMMARY_DIR, "extractive_summaries_v2.json")
ABSTRACTIVE_FILE = os.path.join(SUMMARY_DIR, "abstractive_summaries.json")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ocr_docs = load_json(OCR_FILE)
extractive_summaries = load_json(EXTRACTIVE_FILE)
abstractive_summaries = load_json(ABSTRACTIVE_FILE)

# Convert to dict for easy access
extractive_map = {doc["case_id"]: doc for doc in extractive_summaries}
abstractive_map = {doc["case_id"]: doc for doc in abstractive_summaries}
ocr_map = {doc["case_id"]: doc for doc in ocr_docs}

# -----------------------------
# APP UI
# -----------------------------
st.title("📚 NLP Assignment 2 — Summary Comparison App")
st.markdown("### Streamlit App for Extractive vs Abstractive Summaries")

case_ids = list(abstractive_map.keys())
selected_case = st.selectbox("Select a Case:", case_ids)

if selected_case:
    st.subheader(f"📄 Case ID: {selected_case}")

    # OCR Text
    ocr_data = ocr_map.get(selected_case, {}).get("sentences", {})
    ocr_sentences = [v for k, v in sorted(ocr_data.items(), key=lambda x: int(x[0]))]
    ocr_text = " ".join(ocr_sentences)

    # Extractive + Abstractive summaries
    extractive = extractive_map.get(selected_case, {}).get("extractive_summary", [])
    abstractive = abstractive_map.get(selected_case, {}).get("abstractive_summary", "")

    # -----------------------------
    # Metrics: ROUGE & BLEU
    # -----------------------------
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(" ".join(extractive), abstractive)
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([" ".join(extractive).split()], abstractive.split(), smoothing_function=smoothie)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧾 Original OCR Text")
        st.write(ocr_text)

    with col2:
        st.markdown("### 🧩 Extractive Summary (Highlighted)")
        highlighted_text = ocr_text
        for sent in extractive:
            highlighted_text = highlighted_text.replace(sent, f"**:blue[{sent}]**")
        st.markdown(highlighted_text)

        st.markdown("### 🪶 Abstractive Summary")
        st.markdown(f"**:green[{abstractive}]**")

    # -----------------------------
    # METRICS DISPLAY
    # -----------------------------
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")
    st.write({
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(rouge["rouge1"].fmeasure, 3),
        "ROUGE-2": round(rouge["rouge2"].fmeasure, 3),
        "ROUGE-L": round(rouge["rougeL"].fmeasure, 3),
    })

    st.markdown("---")
    st.info("✅ App compares extractive vs abstractive summaries and highlights extracted sentences in blue.")

