import os
import json
import gc
from tqdm import tqdm
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count

BASE_DIR = "/home/bilal/Documents/NLP_Assignment2/Legal Dataset"
OUTPUT_DIR = "processed_data_ocr_parallel"
DPI = 200
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_ocr_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n?\s*Page\s*\d+\s*\n?', ' ', text, flags=re.IGNORECASE)
    return text.strip()


def ocr_single_pdf(pdf_path: str):
    """Perform OCR for a single PDF (run inside a subprocess)."""
    case_id = Path(pdf_path).stem
    rel_path = os.path.relpath(pdf_path, BASE_DIR)
    record = None

    try:
        info = pdfinfo_from_path(pdf_path)
        num_pages = int(info.get("Pages", 0))
    except Exception:
        num_pages = 0

    if not num_pages:
        print(f"[WARN] Cannot read {pdf_path}")
        return None

    page_texts = []
    for p in range(1, num_pages + 1):
        try:
            images = convert_from_path(
                pdf_path,
                dpi=DPI,
                first_page=p,
                last_page=p,
                fmt="jpeg",
                thread_count=1
            )
            img = images[0]
            text = pytesseract.image_to_string(img, lang="eng")
            page_texts.append(text)
        except Exception as e:
            print(f"[WARN] Page {p} failed for {pdf_path}: {e}")
        finally:
            try:
                img.close()
            except Exception:
                pass
            del images
            gc.collect()

    full_text = clean_ocr_text("\n".join(page_texts))
    record = {
        "case_id": case_id,
        "pdf_source": rel_path,
        "ocr_text": full_text
    }

    # Save individual JSON
    out_dir = os.path.join(OUTPUT_DIR, os.path.dirname(rel_path))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return record


def process_all_pdfs_parallel(base_dir=BASE_DIR, output_dir=OUTPUT_DIR):
    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(base_dir)
        for f in files if f.lower().endswith(".pdf")
    ]

    print(f"📄 Found {len(pdf_files)} PDFs. Using multiprocessing...")

    # Limit workers to avoid RAM explosion
    max_workers = max(1, cpu_count() // 2)
    print(f"🧠 Using {max_workers} parallel workers\n")

    all_records = []
    with Pool(processes=max_workers) as pool:
        for result in tqdm(pool.imap_unordered(ocr_single_pdf, pdf_files), total=len(pdf_files)):
            if result:
                all_records.append(result)
            gc.collect()

    # Save combined
    combined_path = os.path.join(output_dir, "all_cases.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! {len(all_records)} PDFs processed.")
    print(f"Combined dataset saved at: {combined_path}")


if __name__ == "__main__":
    process_all_pdfs_parallel()
