from pathlib import Path
import pickle
import csv
import sys
import json

ROOT = Path(__file__).resolve().parent
WANDB_ROOT = ROOT / "adhav" / "uncertainty" / "wandb"
RUNS = {
    "English": "offline-run-20260930_183309-54vqc2cs",
    "Tamil": "offline-run-20261006_175821-mrp8kz02",
    "Tanglish": "offline-run-20261007_171341-w63zuznq",
}

def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)

def find_generation_file(run_id):
    run_dir = WANDB_ROOT / run_id
    found = list(run_dir.rglob("validation_generations.pkl")) if run_dir.exists() else []
    if not found:
        raise FileNotFoundError(f"Cannot find validation_generations.pkl under {run_dir}")
    found.sort(key=lambda p: len(p.parts))
    return found[0]

def to_list(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v) if isinstance(v, (list, tuple)) else [v]

def records_from_obj(obj, lang):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        for k in ("validation_generations", "generations", "data", "records", "examples", "items"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                return list(obj[k])
            if k in obj and isinstance(obj[k], dict) and obj[k] and all(isinstance(v, dict) for v in obj[k].values()):
                return list(obj[k].values())
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
        fields = {k: to_list(v) for k, v in obj.items() if isinstance(v, (list, tuple)) or hasattr(v, "tolist")}
        likely = [len(fields[k]) for k in ("question", "questions", "responses", "reference", "references") if k in fields]
        if likely:
            n = min(likely)
            return [{k: v[i] for k, v in fields.items() if len(v) > i} for i in range(n)]
    raise TypeError(f"{lang}: unsupported pickle layout: {type(obj)}")

def get_acc(item):
    if isinstance(item, dict):
        for key in ("accuracy", "is_correct", "correct", "exact_match", "em"):
            if key in item:
                return item[key]
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return item[3]
    return None

def to_bool_or_blank(x):
    if x is None:
        return ""
    if hasattr(x, "item"):
        try: x = x.item()
        except Exception: pass
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(bool(x))
    if isinstance(x, str) and x.strip().lower() in ("true", "false", "1", "0"):
        return int(x.strip().lower() in ("true", "1"))
    return ""

def get_text(item):
    if isinstance(item, dict):
        for k in ("response", "text", "answer", "generated_text"):
            if k in item: return str(item[k])
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item) if item is not None else ""

def as_text(v):
    if v is None: return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)

def main():
    rows = []
    for lang, run_id in RUNS.items():
        path = find_generation_file(run_id)
        records = records_from_obj(load_pickle(path), lang)
        n_correct = n_incorrect = n_missing = 0
        all_correct_counts = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict): rec = {}
            responses = rec.get("responses", rec.get("generated_responses", rec.get("samples", [])))
            responses = list(responses) if isinstance(responses, (list, tuple)) else []
            likely = rec.get("most_likely_answer", {})
            likely_acc = to_bool_or_blank(get_acc(likely))
            response_accs = [to_bool_or_blank(get_acc(r)) for r in responses]
            correct_n = sum(a == 1 for a in response_accs)
            incorrect_n = sum(a == 0 for a in response_accs)
            missing_n = len(response_accs) - correct_n - incorrect_n
            all_correct_counts.append(correct_n)
            n_correct += correct_n
            n_incorrect += incorrect_n
            n_missing += missing_n
            rows.append({
                "language": lang,
                "index": i,
                "question": as_text(rec.get("question", rec.get("prompt", ""))),
                "reference_answer": as_text(rec.get("reference", rec.get("reference_answer", rec.get("ground_truth", "")))),
                "most_likely_answer": get_text(likely),
                "most_likely_answer_accuracy": likely_acc,
                "response_count": len(responses),
                "responses_correct_count": correct_n,
                "responses_incorrect_count": incorrect_n,
                "responses_missing_accuracy_count": missing_n,
                **{f"response_{j+1}": get_text(responses[j]) if j < len(responses) else "" for j in range(10)},
                **{f"response_{j+1}_accuracy": response_accs[j] if j < len(response_accs) else "" for j in range(10)},
            })
        print(f"\n{lang}: questions={len(records)}")
        print(f"  Across sampled responses: correct={n_correct}, incorrect={n_incorrect}, missing/unrecognized accuracy={n_missing}")
        if all_correct_counts:
            print(f"  Correct responses per question (among saved samples): min={min(all_correct_counts)}, max={max(all_correct_counts)}, mean={sum(all_correct_counts)/len(all_correct_counts):.2f}")
        print(f"  Source: {path}")
    out = ROOT / "per_response_accuracy_audit.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")
    print("This report only summarizes saved accuracy flags. It does not independently verify answer correctness or change labels.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("No baseline files were modified.")
        sys.exit(1)
