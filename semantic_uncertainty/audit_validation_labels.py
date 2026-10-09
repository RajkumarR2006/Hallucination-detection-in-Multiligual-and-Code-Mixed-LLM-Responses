from pathlib import Path
import pickle
import csv
import json
import re
import sys

ROOT = Path(__file__).resolve().parent
WANDB_ROOT = ROOT / "adhav" / "uncertainty" / "wandb"

RUNS = {
    "English": {
        "generation": "offline-run-20260930_183309-54vqc2cs",
        "uncertainty": "offline-run-20261006_174058-91oo4kq8",
    },
    "Tamil": {
        "generation": "offline-run-20261006_175821-mrp8kz02",
        "uncertainty": "offline-run-20261007_162120-8rooknac",
    },
    "Tanglish": {
        "generation": "offline-run-20261007_171341-w63zuznq",
        "uncertainty": "offline-run-20261008_231922-cb1522ci",
    },
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
    # Prefer the exact expected artifact filename.
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
        # Print likely pickle names to make any unexpected structure diagnosable.
        names = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*.pkl")]
        raise FileNotFoundError(
            f"No pickle containing both validation_is_false and semantic_ids found under {run_dir}.\n"
            f"Pickle candidates: {names[:30]}"
        )
    candidates.sort(key=lambda pair: (("uncertainty" not in pair[0].name.lower()), len(pair[0].parts)))
    return candidates[0]

def to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x) if isinstance(x, (tuple, list)) else [x]

def answer_text(item):
    # Expected response tuple: (response_text, token_log_likelihoods, embedding, accuracy)
    if isinstance(item, (tuple, list)) and item:
        value = item[0]
        return value if isinstance(value, str) else str(value)
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
                if isinstance(value, str):
                    return value
                return str(value)
    return default

def main():
    all_rows = []
    summary = []
    for language, info in RUNS.items():
        gen_path = find_generation_file(info["generation"])
        unc_path, unc = find_uncertainty_file(info["uncertainty"])
        generations = load_pickle(gen_path)
        labels = to_list(unc["validation_is_false"])
        semantic_ids = to_list(unc["semantic_ids"])
        measures = unc.get("uncertainty_measures", {})
        sem_entropy = measures.get("semantic_entropy", []) if isinstance(measures, dict) else []
        sem_entropy = to_list(sem_entropy) if sem_entropy is not None else []

        if isinstance(generations, dict):
            # Support a dict wrapping the list of records.
            for k in ("validation_generations", "generations", "data", "records"):
                if k in generations and isinstance(generations[k], (list, tuple)):
                    generations = generations[k]
                    break
        if not isinstance(generations, (list, tuple)):
            raise TypeError(f"{language}: unexpected generation object type {type(generations)} in {gen_path}")

        n = min(len(generations), len(labels), len(semantic_ids))
        if not (len(generations) == len(labels) == len(semantic_ids)):
            print(
                f"WARNING {language}: length mismatch generations={len(generations)}, "
                f"labels={len(labels)}, semantic_ids={len(semantic_ids)}; exporting first {n}."
            )
        if sem_entropy and len(sem_entropy) != n:
            print(f"WARNING {language}: semantic entropy length={len(sem_entropy)}, aligned rows={n}.")

        positives = sum(bool(x) for x in labels[:n])
        print(f"\n{language}: N={n}, false/hallucinated={positives}, true/correct={n-positives}")
        print(f"  generations: {gen_path}")
        print(f"  uncertainty: {unc_path}")

        for i in range(n):
            rec = generations[i]
            question = get_record_value(rec, ("question", "prompt", "input", "query"))
            context = get_record_value(rec, ("context", "passage"))
            reference = get_record_value(rec, ("reference", "reference_answer", "ground_truth", "target"))
            likely = rec.get("most_likely_answer", {}) if isinstance(rec, dict) else {}
            if isinstance(likely, dict):
                likely_answer = get_record_value(likely, ("response", "text", "answer"))
            else:
                likely_answer = answer_text(likely)
            responses = rec.get("responses", []) if isinstance(rec, dict) else []
            response_texts = [answer_text(x) for x in responses]
            row = {
                "language": language,
                "index": i,
                "question": question,
                "context": context,
                "reference_answer": reference,
                "most_likely_answer": likely_answer,
                "is_false_label": int(bool(labels[i])),
                "label_interpretation": "FALSE / hallucinated" if bool(labels[i]) else "TRUE / correct",
                "baseline_semantic_entropy": sem_entropy[i] if i < len(sem_entropy) else "",
                "baseline_semantic_ids": json.dumps(semantic_ids[i], ensure_ascii=False),
                "num_responses": len(response_texts),
            }
            for j in range(10):
                row[f"response_{j+1}"] = response_texts[j] if j < len(response_texts) else ""
            all_rows.append(row)
            summary.append((language, i, bool(labels[i]), question, reference, likely_answer, len(response_texts)))

    output = ROOT / "validation_label_response_audit.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # A compact label-count report, and show sample rows for quick sanity checks.
    print("\n" + "=" * 72)
    print("LABEL SUMMARY")
    print("=" * 72)
    for language in RUNS:
        subset = [r for r in all_rows if r["language"] == language]
        false_n = sum(r["is_false_label"] for r in subset)
        print(f"{language:10s}: N={len(subset):3d} | false/hallucinated={false_n:3d} | true/correct={len(subset)-false_n:3d}")
    print(f"\nSaved audit file: {output}")
    print("Open the CSV and inspect label_interpretation, reference_answer, most_likely_answer, and response_1..response_10.")
    print("Do NOT use response-level accuracy fields as model features; this audit intentionally excludes them.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("No baseline files were modified.")
        sys.exit(1)
