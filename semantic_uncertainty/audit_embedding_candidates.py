
import os
import pickle
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r"adhav\uncertainty\wandb"

CONFIG = {
    "TAMIL": (
        "offline-run-20261006_175821-mrp8kz02",
        "offline-run-20261007_162120-8rooknac",
    ),
    "TANGLISH": (
        "offline-run-20261007_171341-w63zuznq",
        "offline-run-20261008_231922-cb1522ci",
    ),
}

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_pickle(run, filename):
    path = os.path.join(BASE, run, "files", filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_answer(response):
    if isinstance(response, (tuple, list)):
        return str(response[0]).strip() if response else ""
    return str(response).strip()


model = SentenceTransformer(MODEL_NAME)

for language, (generation_run, uncertainty_run) in CONFIG.items():
    generations = load_pickle(
        generation_run, "validation_generations.pkl"
    )
    uncertainty = load_pickle(
        uncertainty_run, "uncertainty_measures.pkl"
    )
    semantic_ids = uncertainty["semantic_ids"]

    print("\n" + "=" * 80)
    print(language)
    print("=" * 80)

    candidates = []

    for index, (sample_id, example) in enumerate(generations.items()):
        if index >= len(semantic_ids):
            break

        answers = [
            get_answer(r)
            for r in example.get("responses", [])
        ]
        clusters = semantic_ids[index]

        if len(answers) != len(clusters):
            continue

        if len(answers) < 2:
            continue

        embeddings = model.encode(
            answers, normalize_embeddings=True
        )
        similarities = embeddings @ embeddings.T

        for i, j in combinations(range(len(answers)), 2):
            if not answers[i] or not answers[j]:
                continue
            if answers[i] == answers[j]:
                continue

            same_cluster = clusters[i] == clusters[j]
            score = float(similarities[i, j])

            candidates.append({
                "score": score,
                "same_cluster": same_cluster,
                "question": example.get("question", ""),
                "sample_id": sample_id,
                "a": answers[i],
                "b": answers[j],
                "cluster_a": clusters[i],
                "cluster_b": clusters[j],
            })

    # Potential false splits: similar answers separated by baseline.
    cross = sorted(
        (x for x in candidates if not x["same_cluster"]),
        key=lambda x: x["score"],
        reverse=True,
    )

    # Potential false merges: dissimilar answers grouped together.
    within = sorted(
        (x for x in candidates if x["same_cluster"]),
        key=lambda x: x["score"],
    )

    for title, results in [
        ("HIGH SIMILARITY, DIFFERENT BASELINE CLUSTERS", cross[:8]),
        ("LOW SIMILARITY, SAME BASELINE CLUSTER", within[:8]),
    ]:
        print("\n---", title, "---")

        for item in results:
            print(
                f"\nScore={item['score']:.4f} | "
                f"ID={item['sample_id']} | "
                f"Clusters={item['cluster_a']},"
                f"{item['cluster_b']}"
            )
            print("Question:", item["question"])
            print("A:", item["a"])
            print("B:", item["b"])

    print("\nCandidate pairs examined:", len(candidates))
