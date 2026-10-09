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
    if not run_dir.exists():
        raise FileNotFoundError(f"Generation run directory not found: {run_dir}")
    matches = list(run_dir.rglob("validation_generations.pkl"))
    if not matches:
        matches = [p for p in run_dir.rglob("*.pkl") if "generation" in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No generation pickle found under {run_dir}")
    matches.sort(key=lambda p: (p.name != "validation_generations.pkl", len(p.parts)))
    return matches[0]

def find_uncertainty_file(run_id):
    run_dir = WANDB_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Uncertainty run directory not found: {run_dir}")
    candidates = []
    for p in run_dir.rglob("*.pkl"):
        try:
            obj = load_pickle(p)
        except Exception:
            continue
        if isinstance(obj, dict) and "validation_is_false" in obj and "semantic_ids" in obj:
            candidates.append((p, obj))
    if not candidates:
        names = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*.pkl")]
        raise FileNotFoundError(f"No pickle containing validation_is_false and semantic_ids found under {run_dir}. Pickle candidates: {names[:30]}")
    candidates.sort(key=lambda pair: (("uncertainty" not in pair[0].name.lower()), len(pair[0].parts)))
    return candidates[0]

def to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]

def answer_text(item):
    if isinstance(item, (tuple, list)) and item:
        return item[0] if isinstance(item[0], str) else str(item[0])
    if isinstance(item, dict):
        for k in ("response", "text", "answer", "generated_text"):
            if k in item:
                return str(item[k])
    return str(item)

def get_record_value(record, keys, default=""):
    if isinstance(record, dict):
        for key in keys:
            if key in record and record[key] is not None:
                value = record[key]
                return value if isinstance(value, str) else str(value)
    return default

def normalize_generations(obj, language, path):
    """Support record lists, dictionaries of records, and column-oriented dictionaries."""
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if not isinstance(obj, dict):
        raise TypeError(f"{language}: unexpected generation object type {type(obj)} in {path}")

    # Common wrapper around the actual records.
    for key in ("validation_generations", "generations", "data", "records", "examples", "items"):
        if key in obj and isinstance(obj[key], (list, tuple)):
            return list(obj[key])
        if key in obj and isinstance(obj[key], dict):
            nested = obj[key]
            if nested and all(isinstance(v, dict) for v in nested.values()):
                return list(nested.values())

    # Dictionary indexed by example ID -> record.
    if obj and all(isinstance(v, dict) for v in obj.values()):
        return list(obj.values())

    # Column-oriented structure: each field contains one value per example.
    list_fields = {k: to_list(v) for k, v in obj.items()
                   if isinstance(v, (list, tuple)) or hasattr(v, "tolist")}
    likely_lengths = [len(v) for k, v in list_fields.items()
                      if k in ("question", "questions", "responses", "reference", "references",
                               "most_likely_answer", "contexts", "context", "prompt", "prompts")]
    if likely_lengths:
        n = min(likely_lengths)
        if n > 0:
            records = []
            for i in range(n):
                rec = {}
                for key, values in list_fields.items():
                    if len(values) == n or len(values) >= n:
                        rec[key] = values[i]
                records.append(rec)
            return records

    key_info = {str(k): type(v).__name__ for k, v in list(obj.items())[:25]}
    raise TypeError(
        f"{language}: generation pickle is a dict but could not identify its record layout: {path}\n"
        f"Top-level keys/types: {key_info}\n"
        "No files were modified."
    )

def main():
    all_rows = []
    for language, info in RUNS.items():
        gen_path = find_generation_file(info["generation"])
        unc_path, unc = find_uncertainty_file(info["uncertainty"])
        generations = normalize_generations(load_pickle(gen_path), language, gen_path)
        labels = to_list(unc["validation_is_false"])
        semantic_ids = to_list(unc["semantic_ids"])
        measures = unc.get("uncertainty_measures", {})
        sem_entropy = measures.get("semantic_entropy", []) if isinstance(measures, dict) else []
        sem_entropy = to_list(sem_entropy) if sem_entropy is not None else []

        n = min(len(generations), len(labels), len(semantic_ids))
        if not (len(generations) == len(labels) == len(semantic_ids)):
            print(f"WARNING {language}: generations={len(generations)}, labels={len(labels)}, semantic_ids={len(semantic_ids)}; exporting first {n}.")
        print(f"\n{language}: N={n}, false/hallucinated={sum(bool(x) for x in labels[:n])}, true/correct={n-sum(bool(x) for x in labels[:n])}")
        print(f"  generation file: {gen_path}")
        print(f"  uncertainty file: {unc_path}")

        for i in range(n):
            rec = generations[i]
            question = get_record_value(rec, ("question", "prompt", "input", "query", "questions"))
            context = get_record_value(rec, ("context", "passage", "contexts"))
            reference = get_record_value(rec, ("reference", "reference_answer", "ground_truth", "target", "references"))
            likely = rec.get("most_likely_answer", {}) if isinstance(rec, dict) else {}
            if isinstance(likely, dict):
                likely_answer = get_record_value(likely, ("response", "text", "answer", "generated_text"))
            else:
                likely_answer = answer_text(likely)
            responses = rec.get("responses", []) if isinstance(rec, dict) else []
            # Some formats store one response collection under alternate field names.
            if not responses and isinstance(rec, dict):
                responses = rec.get("generated_responses", rec.get("samples", []))
            response_texts = [answer_text(x) for x in responses] if isinstance(responses, (tuple, list)) else []
            row = {
                "language": language, "index": i, "question": question, "context": context,
                "reference_answer": reference, "most_likely_answer": likely_answer,
                "is_false_label": int(bool(labels[i])),
                "label_interpretation": "FALSE / hallucinated" if bool(labels[i]) else "TRUE / correct",
                "baseline_semantic_entropy": sem_entropy[i] if i < len(sem_entropy) else "",
                "baseline_semantic_ids": json.dumps(semantic_ids[i], ensure_ascii=False),
                "num_responses": len(response_texts),
            }
            for j in range(10):
                row[f"response_{j+1}"] = response_texts[j] if j < len(response_texts) else ""
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No audit rows were produced.")
    output = ROOT / "validation_label_response_audit.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 65 + "\nLABEL SUMMARY\n" + "=" * 65)
    for language in RUNS:
        subset = [r for r in all_rows if r["language"] == language]
        false_n = sum(r["is_false_label"] for r in subset)
        print(f"{language:10s}: N={len(subset):3d} | false/hallucinated={false_n:3d} | true/correct={len(subset)-false_n:3d}")
    print(f"\nSaved audit file: {output}")
    print("Inspect question, reference_answer, most_likely_answer, response_1..response_10, and label_interpretation.")
    print("The script does not use response-level accuracy as a feature and does not modify baseline files.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
