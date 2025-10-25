import json
import os
import argparse


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # If top-level is list -> return list
    if isinstance(obj, list):
        return obj
    # If wrapper contains "Cases"
    if isinstance(obj, dict):
        if "Cases" in obj and isinstance(obj["Cases"], list):
            return obj["Cases"]
        # try other common wrappers (optional)
        for k in ("data", "results", "cases"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # dict-of-id -> list
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            return vals
    # fallback
    return [obj]


def record_key(rec):
    if not isinstance(rec, dict):
        return json.dumps(rec, sort_keys=True)
    for k in ("case_id", "Case_No", "Case_Title", "pdf_source", "id"):
        v = rec.get(k)
        if v:
            # normalize to string
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            return str(v).strip().lower()
    # fallback to full content
    return json.dumps(rec, sort_keys=True)


def merge(all_cases_path, data_path, out_path):
    if not os.path.exists(all_cases_path):
        raise FileNotFoundError(all_cases_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    all_cases = load_json_file(all_cases_path)
    data_cases = load_json_file(data_path)

    merged = []
    seen = set()

    # add all_cases first
    for r in all_cases:
        k = record_key(r)
        if k not in seen:
            seen.add(k)
            merged.append(r)

    # then append remaining from data.json
    added = 0
    for r in data_cases:
        k = record_key(r)
        if k not in seen:
            seen.add(k)
            merged.append(r)
            added += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged: {len(all_cases)} (all_cases) + {added} (from data.json) -> {len(merged)} total")
    print(f"Saved merged file to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all_cases", default="processed_data_ocr_parallel/all_cases.json")
    p.add_argument("--data", default="processed_data_ocr_parallel/data.json")
    p.add_argument("--out", default="processed_data_ocr_parallel/merged_all.json")
    args = p.parse_args()
    merge(args.all_cases, args.data, args.out)