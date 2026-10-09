"""
Exploratory embedding-assisted semantic grouping (EASG) evaluation.

This script does NOT overwrite baseline artifacts. It reuses the saved answer
generations and labels, applies cosine-threshold clustering to each question,
and reports AUROC/AUPRC for a small threshold grid.

Run from the semantic_uncertainty directory:
    python .\evaluate_easg.py

Important: threshold results are exploratory because the thresholds are tested
on the same validation examples. Do not present the best threshold as an
unbiased final test result.
"""
import os
import pickle
import csv
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import AgglomerativeClustering

BASE = r"adhav\uncertainty\wandb"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.98]

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

def get_answer(response):
    if isinstance(response, (tuple, list)):
        return str(response[0]).strip() if response else ""
    return str(response).strip()

def get_labels(data):
    labels = data.get("validation_is_false")
    if labels is None:
        raise KeyError("validation_is_false missing from uncertainty_measures.pkl")
    labels = np.asarray(labels).astype(int).reshape(-1)
    return labels

def entropy_from_cluster_ids(cluster_ids):
    _, counts = np.unique(cluster_ids, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))

def cluster_answers(embeddings, threshold):
    # Average-linkage clustering; cosine distance = 1 - cosine similarity.
    # A distance threshold of 1-threshold corresponds to the selected cosine cut.
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
    except TypeError:
        # Compatibility with older scikit-learn releases.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
    return clustering.fit_predict(embeddings)

def safe_metrics(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, scores)),
        float(average_precision_score(y_true, scores)),
    )

def main():
    print("Loading multilingual embedding model...")
    model = SentenceTransformer(MODEL_NAME)
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
        labels = get_labels(uncertainty)
        samples = list(generations.items())

        n = min(len(samples), len(labels))
        if len(samples) != len(labels):
            print(f"WARNING: generations={len(samples)}, labels={len(labels)}; using {n}")
        samples = samples[:n]
        labels = labels[:n]

        # Build embeddings once, then reuse for every threshold.
        encoded_samples = []
        for sample_id, example in samples:
            responses = example.get("responses", [])
            answers = [get_answer(r) for r in responses]
            if not answers:
                answers = [""]
            emb = model.encode(
                answers,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoded_samples.append((sample_id, answers, emb))

        for threshold in THRESHOLDS:
            entropies = []
            cluster_counts = []
            usable_labels = []

            for i, (_, answers, embeddings) in enumerate(encoded_samples):
                if len(answers) < 2:
                    continue
                cluster_ids = cluster_answers(embeddings, threshold)
                entropies.append(entropy_from_cluster_ids(cluster_ids))
                cluster_counts.append(len(np.unique(cluster_ids)))
                usable_labels.append(labels[i])

            y = np.asarray(usable_labels, dtype=int)
            entropy_scores = np.asarray(entropies, dtype=float)
            auroc, auprc = safe_metrics(y, entropy_scores)

            row = {
                "language": language,
                "threshold": threshold,
                "n": len(y),
                "false_count": int(np.sum(y == 1)),
                "true_count": int(np.sum(y == 0)),
                "mean_cluster_count": float(np.mean(cluster_counts)),
                "mean_embedding_cluster_entropy": float(np.mean(entropy_scores)),
                "auroc": auroc,
                "auprc": auprc,
            }
            output_rows.append(row)
            print(
                f"threshold={threshold:.2f} | "
                f"mean_clusters={row['mean_cluster_count']:.2f} | "
                f"mean_entropy={row['mean_embedding_cluster_entropy']:.4f} | "
                f"AUROC={auroc:.4f} | AUPRC={auprc:.4f}"
            )

    out = "easg_threshold_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved results to: {os.path.abspath(out)}")
    print("Baseline files were not modified.")
    print("Treat this threshold sweep as exploratory, not as an unbiased final test.")

if __name__ == "__main__":
    main()
