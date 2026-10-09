from pathlib import Path
import pickle
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
WANDB_ROOT = ROOT / "adhav" / "uncertainty" / "wandb"
RUNS = {
    "English": {"generation": "offline-run-20260930_183309-54vqc2cs", "uncertainty": "offline-run-20261006_174058-91oo4kq8"},
    "Tamil": {"generation": "offline-run-20261006_175821-mrp8kz02", "uncertainty": "offline-run-20261007_162120-8rooknac"},
    "Tanglish": {"generation": "offline-run-20261007_171341-w63zuznq", "uncertainty": "offline-run-20261008_231922-cb1522ci"},
}

def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)

def find_generation_file(run_id):
    run_dir = WANDB_ROOT / run_id
    matches = list(run_dir.rglob("validation_generations.pkl")) if run_dir.exists() else []
    if not matches:
        raise FileNotFoundError(f"Could not find validation_generations.pkl in {run_dir}")
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]

def find_uncertainty_file(run_id):
    run_dir = WANDB_ROOT / run_id
    candidates = []
    if run_dir.exists():
        for p in run_dir.rglob("*.pkl"):
            try:
                obj = load_pickle(p)
            except Exception:
                continue
            if isinstance(obj, dict) and "validation_is_false" in obj and "semantic_ids" in obj:
                candidates.append((p, obj))
    if not candidates:
        raise FileNotFoundError(f"Could not find uncertainty_measures pickle in {run_dir}")
    candidates.sort(key=lambda pair: (("uncertainty" not in pair[0].name.lower()), len(pair[0].parts)))
    return candidates[0]

def to_list(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def normalize_records(obj, language):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if not isinstance(obj, dict):
        raise TypeError(f"{language}: unsupported generation object type {type(obj)}")
    for k in ("validation_generations", "generations", "data", "records", "examples", "items"):
        if k in obj and isinstance(obj[k], (list, tuple)):
            return list(obj[k])
        if k in obj and isinstance(obj[k], dict):
            v = obj[k]
            if v and all(isinstance(x, dict) for x in v.values()):
                return list(v.values())
    if obj and all(isinstance(v, dict) for v in obj.values()):
        return list(obj.values())
    # This project pickle appears dictionary-shaped. Try parallel columns as a fallback.
    fields = {k: to_list(v) for k, v in obj.items()
              if isinstance(v, (list, tuple)) or hasattr(v, "tolist")}
    preferred = [len(fields[k]) for k in ("question", "questions", "responses", "reference", "references")
                 if k in fields]
    if preferred:
        n = min(preferred)
        return [{k: values[i] for k, values in fields.items() if len(values) > i} for i in range(n)]
    raise TypeError(f"{language}: can't infer generation record layout. Top-level keys: {list(obj.keys())[:40]}")

def extract_accuracy(item):
    if isinstance(item, dict):
        for k in ("accuracy", "is_correct", "correct", "exact_match", "em"):
            if k in item:
                return item[k]
    if isinstance(item, (tuple, list)) and len(item) >= 4:
        return item[3]
    return ""

def extract_text(item):
    if isinstance(item, dict):
        for k in ("response", "text", "answer", "generated_text"):
            if k in item:
                return str(item[k])
    if isinstance(item, (tuple, list)) and item:
        return str(item[0])
    return str(item) if item is not None else ""

def safe_str(v):
    if v is None:
        return ""
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)

def main():
    rows = []
    for language, cfg in RUNS.items():
        gen_path = find_generation_file(cfg["generation"])
        unc_path, unc = find_uncertainty_file(cfg["uncertainty"])
        records = normalize_records(load_pickle(gen_path), language)
        labels = to_list(unc["validation_is_false"])
        n = min(len(records), len(labels))
        if len(records) != len(labels):
            print(f"WARNING {language}: generations={len(records)}, labels={len(labels)}; using first {n}")
        counts = {"label_false": 0, "label_true": 0, "acc_false": 0, "acc_true": 0, "acc_missing": 0, "mismatch": 0}
        for i in range(n):
            rec = records[i] if isinstance(records[i], dict) else {}
            label_false = bool(labels[i])
            likely = rec.get("most_likely_answer", {})
            acc = extract_accuracy(likely)
            answer = extract_text(likely)
            if isinstance(acc, bool):
                acc_bool = acc
            elif isinstance(acc, (int, float)) and acc in (0, 1):
                acc_bool = bool(acc)
            elif isinstance(acc, str) and acc.strip().casefold() in ("true", "false", "1", "0"):
                acc_bool = acc.strip().casefold() in ("true", "1")
            else:
                acc_bool = None
            if label_false: counts["label_false"] += 1
            else: counts["label_true"] += 1
            if acc_bool is None:
                counts["acc_missing"] += 1
            elif acc_bool: counts["acc_true"] += 1
            else: counts["acc_false"] += 1
            # Expected relationship: validation_is_false=True means answer is incorrect.
            mismatch = (acc_bool is not None and label_false == acc_bool)
            if mismatch:
                counts["mismatch"] += 1
            rows.append({
                "language": language,
                "index": i,
                "question": safe_str(rec.get("question", rec.get("prompt", ""))),
                "reference_answer": safe_str(rec.get("reference", rec.get("reference_answer", rec.get("ground_truth", "")))),
                "most_likely_answer": answer,
                "validation_is_false": int(label_false),
                "most_likely_answer_accuracy_raw": safe_str(acc),
                "accuracy_interpreted_as_correct": "" if acc_bool is None else int(acc_bool),
                "label_accuracy_mismatch_candidate": int(mismatch),
                "generation_run": cfg["generation"],
                "uncertainty_run": cfg["uncertainty"],
            })
        print(f"\n{language}: N={n}")
        print(f"  label says false/hallucinated: {counts['label_false']}; true/correct: {counts['label_true']}")
        print(f"  saved most-likely accuracy: correct={counts['acc_true']}; incorrect={counts['acc_false']}; missing/unrecognized={counts['acc_missing']}")
        print(f"  possible label/accuracy mismatches: {counts['mismatch']}")
        print("  NOTE: this is an audit only; no labels are changed.")
    out = ROOT / "saved_accuracy_label_audit.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")
    print("Upload saved_accuracy_label_audit.csv here. Treat mismatch candidates as flags for review, not automatic relabeling.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("No baseline files were modified.")
        sys.exit(1)
