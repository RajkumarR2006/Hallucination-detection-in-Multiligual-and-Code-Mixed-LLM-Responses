"""
Combined uncertainty experiment: baseline semantic entropy + embedding entropy.

Run from the semantic_uncertainty directory:
    python .\evaluate_combined_uncertainty.py

This script:
- Loads the same saved generations and correctness labels.
- Recomputes embedding-cluster entropy at a few exploratory thresholds.
- Reads baseline semantic entropy from uncertainty_measures.pkl.
- Tests standardized linear combinations of the two scores.
- Writes a separate CSV; does not overwrite baseline artifacts.

Caution: this is an exploratory comparison on the existing validation sets.
Selecting thresholds/weights on these same sets is not an unbiased final test.
"""
import os
import pickle
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

BASE = r"adhav\uncertainty\wandb"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

THRESHOLDS = [0.90, 0.93, 0.95, 0.97]
WEIGHTS = [0.25, 0.50, 0.75]  # weight assigned to embedding entropy

CONFIG = {
    "English": {
        "generation_run": "offline-run-20260930_183309-54vqc2cs",
        "uncertainty_run": "offline-run-20261006_174058-91oo4kq8",
    },
    "Tamil": {
        "generation_run": "offline-run-20261006_175821-mrp8kz02",
        "uncertainty_run": "offline-run-20261007_162120-8rooknac",
    },
    "Tanglish": {
        "generation_run": "offline-run-20261007_171341-w63zuznq",
        "uncertainty_run": "offline-run-20261008_231922-cb1522ci",
    },
}

def load_pickle(run, filename):
    path = os.path.join(BASE, run, "files", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def answer_text(response):
    if isinstance(response, (tuple, list)):
        return str(response[0]).strip() if response else ""
    return str(response).strip()

def get_baseline_semantic_entropy(payload):
    """
    Handle common pickle layouts without silently substituting another metric.
    Expected: payload['uncertainty_measures']['semantic_entropy'].
    """
    measures = payload.get("uncertainty_measures")
    if not isinstance(measures, dict):
        raise KeyError("Expected uncertainty_measures to be a dictionary.")

    candidates = [
        "semantic_entropy",
        "semantic_entropy_measures",
    ]
    for key in candidates:
        if key in measures:
            value = measures[key]
            if isinstance(value, dict):
                # Some implementations nest the per-example entropy.
                for nested_key in ("semantic_entropy", "entropy"):
                    if nested_key in value:
                        value = value[nested_key]
                        break
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr

    available = list(measures.keys())
    raise KeyError(
        "Could not find per-example baseline semantic entropy. "
        f"Available uncertainty keys: {available}"
    )

def entropy_from_ids(ids):
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))

def cluster_answers(embeddings, threshold):
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
    return model.fit_predict(embeddings)

def metrics(y, scores):
    if len(y) != len(scores) or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y, scores)),
        float(average_precision_score(y, scores)),
    )

def zscore(values):
    values = np.asarray(values, dtype=float)
    sd = np.std(values)
    if sd < 1e-12:
        return np.zeros_like(values)
    return (values - np.mean(values)) / sd

def main():
    print("Loading multilingual embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)
    output_rows = []

    for language, cfg in CONFIG.items():
        print("\n" + "=" * 78)
        print(language)

        generations = load_pickle(
            cfg["generation_run"], "validation_generations.pkl"
        )
        uncertainty = load_pickle(
            cfg["uncertainty_run"], "uncertainty_measures.pkl"
        )

        y = np.asarray(
            uncertainty["validation_is_false"], dtype=int
        ).reshape(-1)
        baseline_sem = get_baseline_semantic_entropy(uncertainty)

        samples = list(generations.items())
        n = min(len(samples), len(y), len(baseline_sem))
        if len({len(samples), len(y), len(baseline_sem)}) != 1:
            print(
                "WARNING: length mismatch; using aligned prefix "
                f"generations={len(samples)}, labels={len(y)}, "
                f"baseline_entropy={len(baseline_sem)}, n={n}"
            )

        samples = samples[:n]
        y = y[:n]
        baseline_sem = baseline_sem[:n]

        encoded = []
        for sample_id, example in samples:
            answers = [
                answer_text(r) for r in example.get("responses", [])
            ]
            if len(answers) < 2:
                answers = answers or [""]
                emb = embedder.encode(
                    answers, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False
                )
            else:
                emb = embedder.encode(
                    answers, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False
                )
            encoded.append((sample_id, answers, emb))

        baseline_auc, baseline_ap = metrics(y, baseline_sem)
        output_rows.append({
            "language": language,
            "method": "baseline_semantic_entropy",
            "threshold": "",
            "embedding_weight": "",
            "n": n,
            "false_count": int(np.sum(y == 1)),
            "true_count": int(np.sum(y == 0)),
            "mean_embedding_entropy": "",
            "auroc": baseline_auc,
            "auprc": baseline_ap,
        })
        print(
            f"Baseline semantic entropy: AUROC={baseline_auc:.4f}, "
            f"AUPRC={baseline_ap:.4f}"
        )

        z_base = zscore(baseline_sem)

        for threshold in THRESHOLDS:
            emb_entropy = []
            for _, answers, embeddings in encoded:
                if len(answers) < 2:
                    emb_entropy.append(0.0)
                else:
                    ids = cluster_answers(embeddings, threshold)
                    emb_entropy.append(entropy_from_ids(ids))

            emb_entropy = np.asarray(emb_entropy, dtype=float)
            emb_auc, emb_ap = metrics(y, emb_entropy)
            output_rows.append({
                "language": language,
                "method": "embedding_entropy_only",
                "threshold": threshold,
                "embedding_weight": 1.0,
                "n": n,
                "false_count": int(np.sum(y == 1)),
                "true_count": int(np.sum(y == 0)),
                "mean_embedding_entropy": float(np.mean(emb_entropy)),
                "auroc": emb_auc,
                "auprc": emb_ap,
            })

            z_emb = zscore(emb_entropy)
            for weight in WEIGHTS:
                combined = (1.0 - weight) * z_base + weight * z_emb
                auc, ap = metrics(y, combined)
                row = {
                    "language": language,
                    "method": "combined_zscore_entropy",
                    "threshold": threshold,
                    "embedding_weight": weight,
                    "n": n,
                    "false_count": int(np.sum(y == 1)),
                    "true_count": int(np.sum(y == 0)),
                    "mean_embedding_entropy": float(np.mean(emb_entropy)),
                    "auroc": auc,
                    "auprc": ap,
                }
                output_rows.append(row)
                print(
                    f"threshold={threshold:.2f}, emb_weight={weight:.2f} | "
                    f"AUROC={auc:.4f} | AUPRC={ap:.4f}"
                )

    out = "combined_uncertainty_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved: {os.path.abspath(out)}")
    print("No baseline files were modified.")
    print(
        "Exploratory only: thresholds and weights are evaluated on the same "
        "validation sets, so do not treat the best row as an unbiased estimate."
    )

if __name__ == "__main__":
    main()
