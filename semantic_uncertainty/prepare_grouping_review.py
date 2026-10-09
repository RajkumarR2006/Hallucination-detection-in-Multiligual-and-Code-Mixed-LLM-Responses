
import os
import pickle
import csv
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r"adhav\uncertainty\wandb"
OUTPUT = "grouping_review.csv"

CONFIG = {
    "Tamil": (
        "offline-run-20261006_175821-mrp8kz02",
        "offline-run-20261007_162120-8rooknac",
    ),
    "Tanglish": (
        "offline-run-20261007_171341-w63zuznq",
        "offline-run-20261008_231922-cb1522ci",
    ),
}

def load_pickle(run, filename):
    path = os.path.join(BASE, run, "files", filename)
    with open(path, "rb") as f:
        return pickle.load(f)

def answer_text(response):
    if isinstance(response, (tuple, list)):
        return str(response[0]).strip() if response else ""
    return str(response).strip()

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

rows = []
seen = set()

for language, (gen_run, unc_run) in CONFIG.items():
    generations = load_pickle(gen_run, "validation_generations.pkl")
    uncertainty = load_pickle(unc_run, "uncertainty_measures.pkl")

    for idx, (sample_id, example) in enumerate(generations.items()):
        if idx >= len(uncertainty["semantic_ids"]):
            break

        answers = [
            answer_text(r) for r in example.get("responses", [])
        ]
        clusters = uncertainty["semantic_ids"][idx]

        if len(answers) != len(clusters) or len(answers) < 2:
            continue

        embeddings = model.encode(
            answers, normalize_embeddings=True
        )
        similarities = embeddings @ embeddings.T

        for i, j in combinations(range(len(answers)), 2):
            a, b = answers[i], answers[j]

            if not a or not b or a == b:
                continue

            same_cluster = clusters[i] == clusters[j]
            score = float(similarities[i, j])

            # Keep only strong candidates for manual review.
            if same_cluster and score > 0.35:
                continue
            if not same_cluster and score < 0.85:
                continue

            # Deduplicate repeated answer pairs within a question.
            pair = tuple(sorted((a.casefold(), b.casefold())))
            key = (language, sample_id, pair)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "language": language,
                "sample_id": sample_id,
                "question": example.get("question", ""),
                "answer_a": a,
                "answer_b": b,
                "baseline_cluster_a": clusters[i],
                "baseline_cluster_b": clusters[j],
                "cosine_similarity": round(score, 4),
                "same_meaning_manual": "",
                "notes": "",
            })

rows.sort(
    key=lambda r: (
        r["language"],
        r["cosine_similarity"],
    ),
    reverse=True,
)

with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

print(f"Saved {len(rows)} candidate pairs to {OUTPUT}")
print("Review same_meaning_manual:")
print("  1 = same meaning")
print("  0 = different meaning")
print("  leave blank = uncertain")
