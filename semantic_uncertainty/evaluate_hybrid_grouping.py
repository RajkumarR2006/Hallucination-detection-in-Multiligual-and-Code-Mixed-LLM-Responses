"""
Hybrid semantic-clustering experiment for English, Tamil, and Tanglish.

Run from the semantic_uncertainty directory:
    python .\evaluate_hybrid_grouping.py

Uses saved validation_generations.pkl and uncertainty_measures.pkl only.
Does not regenerate responses or overwrite baseline files.

Variants:
- baseline: saved DeBERTa semantic entropy
- conservative: split baseline clusters when multilingual similarity is low,
  except exact normalized duplicates
- consensus: merge if baseline cluster IDs agree OR similarity is high
- blended: z-scored baseline entropy + each alternative entropy

Threshold/weight selection on these validation sets is exploratory, not an
unbiased estimate of generalization. The response-level accuracy field is
deliberately not used as a predictor because it leaks correctness labels.
"""
import os
import re
import pickle
import csv
import unicodedata
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = os.path.join("adhav", "uncertainty", "wandb")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.98]
BLEND_WEIGHTS = [0.25, 0.50, 0.75]

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
        # W&B sometimes stores a duplicate under files/files.
        nested = os.path.join(BASE, run, "files", "files", filename)
        if os.path.isfile(nested):
            path = nested
        else:
            raise FileNotFoundError(f"Missing saved artifact: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_answer(text):
    """Conservative normalization; preserves letters/digits and technical tokens."""
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.strip()


def response_text(response):
    if isinstance(response, (tuple, list)):
        return str(response[0]).strip() if response else ""
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(response).strip()


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def relabel(labels):
    mapping = {}
    return np.asarray([mapping.setdefault(x, len(mapping)) for x in labels], dtype=int)


def cluster_hybrid(base_ids, normalized, similarities, threshold, mode):
    """
    Greedy deterministic grouping using pairwise decisions.
    Exact normalized duplicates are always linked.
    conservative: baseline-linked pairs stay linked only when sim >= threshold.
    consensus: baseline-linked pairs OR high-similarity pairs are linked.
    Connected components can create transitive merges; outputs are exploratory.
    """
    n = len(base_ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            exact = bool(normalized[i]) and normalized[i] == normalized[j]
            same_base = base_ids[i] == base_ids[j]
            similar = similarities[i, j] >= threshold
            if exact:
                link = True
            elif mode == "conservative":
                link = same_base and similar
            elif mode == "consensus":
                link = same_base or similar
            else:
                raise ValueError(mode)
            if link:
                union(i, j)

    return relabel([find(i) for i in range(n)])


def metrics(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def main():
    print("Loading multilingual sentence embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    rows = []

    for language, cfg in CONFIG.items():
        print("\n" + "=" * 72)
        print(language)

        generations = load_pickle(cfg["generation_run"], "validation_generations.pkl")
        payload = load_pickle(cfg["uncertainty_run"], "uncertainty_measures.pkl")
        labels = np.asarray(payload["validation_is_false"], dtype=int).reshape(-1)
        base_ids_all = payload["semantic_ids"]
        measures = payload["uncertainty_measures"]
        base_entropy = np.asarray(measures["semantic_entropy"], dtype=float).reshape(-1)

        samples = list(generations.items())
        n = min(len(samples), len(labels), len(base_ids_all), len(base_entropy))
        if not (len(samples) == len(labels) == len(base_ids_all) == len(base_entropy)):
            print("WARNING: lengths differ; using aligned prefix of", n)
            print("generations/labels/ids/entropy:",
                  len(samples), len(labels), len(base_ids_all), len(base_entropy))

        samples = samples[:n]
        labels = labels[:n]
        base_ids_all = base_ids_all[:n]
        base_entropy = base_entropy[:n]

        baseline_auc, baseline_ap = metrics(labels, base_entropy)
        print(f"N={n}; false={int(labels.sum())}; true={int((labels == 0).sum())}")
        print(f"Baseline semantic entropy: AUROC={baseline_auc:.4f}, AUPRC={baseline_ap:.4f}")
        rows.append({
            "language": language, "method": "baseline_semantic_entropy",
            "threshold": "", "blend_weight": "", "n": n,
            "false_count": int(labels.sum()), "true_count": int((labels == 0).sum()),
            "mean_entropy": float(np.mean(base_entropy)),
            "auroc": baseline_auc, "auprc": baseline_ap,
        })

        encoded = []
        for i, (sample_id, example) in enumerate(samples):
            responses = example.get("responses", [])
            answers = [response_text(r) for r in responses]
            if not answers:
                answers = [""]
            embeddings = model.encode(
                answers, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False
            )
            similarities = np.matmul(embeddings, embeddings.T)
            encoded.append((sample_id, answers, embeddings, similarities))

        # Verify that the per-question response count aligns with stored IDs.
        for i, ((sample_id, answers, _, _), ids) in enumerate(zip(encoded, base_ids_all)):
            if len(answers) != len(ids):
                raise ValueError(
                    f"{language} record {sample_id}: {len(answers)} responses but "
                    f"{len(ids)} baseline semantic IDs. Stop rather than misalign."
                )

        z_base = zscore(base_entropy)

        for mode in ("conservative", "consensus"):
            for threshold in THRESHOLDS:
                alt_entropy = []
                for i, (_, answers, _, similarities) in enumerate(encoded):
                    normalized = [normalize_answer(a) for a in answers]
                    ids = cluster_hybrid(
                        base_ids_all[i], normalized, similarities, threshold, mode
                    )
                    alt_entropy.append(entropy(ids))

                alt_entropy = np.asarray(alt_entropy)
                auc, ap = metrics(labels, alt_entropy)
                method = f"{mode}_hybrid_entropy"
                rows.append({
                    "language": language, "method": method,
                    "threshold": threshold, "blend_weight": "",
                    "n": n, "false_count": int(labels.sum()),
                    "true_count": int((labels == 0).sum()),
                    "mean_entropy": float(np.mean(alt_entropy)),
                    "auroc": auc, "auprc": ap,
                })
                print(f"{method:25s} threshold={threshold:.2f} "
                      f"AUROC={auc:.4f} AUPRC={ap:.4f}")

                z_alt = zscore(alt_entropy)
                for weight in BLEND_WEIGHTS:
                    combined = (1 - weight) * z_base + weight * z_alt
                    c_auc, c_ap = metrics(labels, combined)
                    rows.append({
                        "language": language,
                        "method": f"baseline_plus_{mode}_hybrid",
                        "threshold": threshold, "blend_weight": weight,
                        "n": n, "false_count": int(labels.sum()),
                        "true_count": int((labels == 0).sum()),
                        "mean_entropy": float(np.mean(alt_entropy)),
                        "auroc": c_auc, "auprc": c_ap,
                    })
                    print(f"  blend weight={weight:.2f} "
                          f"AUROC={c_auc:.4f} AUPRC={c_ap:.4f}")

    output = "hybrid_grouping_results.csv"
    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output)
    print("Baseline files were not modified.")
    print("Reminder: this is an exploratory validation comparison; do not report")
    print("the best grid-selected result as an independent test estimate.")


if __name__ == "__main__":
    main()
