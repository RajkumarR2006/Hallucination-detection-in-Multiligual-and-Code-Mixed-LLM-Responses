import json
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Tamil": BASE_DIR / "data" / "tamil_qa" / "tamil_qa.json",
    "Tanglish": BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa.json"
}

# --------------------------------------------------
# Analyze Function
# --------------------------------------------------

def analyze_dataset(name, filepath):

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_map = defaultdict(list)

    for item in data:
        question_map[item["question"]].append(item)

    duplicates = {
        q: items
        for q, items in question_map.items()
        if len(items) > 1
    }

    print(f"Duplicate Groups : {len(duplicates)}")

    for idx, (question, items) in enumerate(duplicates.items(), start=1):

        print("\n" + "-" * 80)
        print(f"Duplicate #{idx}")
        print("-" * 80)

        print("Question:")
        print(question)

        print()

        for sample in items:

            print(f"ID : {sample['id']}")
            print(f"Answer : {sample['reference_answer']}")
            print()

# --------------------------------------------------

for dataset, path in DATASETS.items():
    analyze_dataset(dataset, path)