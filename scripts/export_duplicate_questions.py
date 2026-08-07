"""
Export Duplicate Questions for Manual Retranslation

Author : Member 1
Purpose:
    Extract duplicate Tamil and Tanglish questions together with
    their original English question and reference answer.

Output:
    reports/duplicate_translation_review.csv
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

english_file = BASE_DIR / "data" / "english_qa" / "english_qa.json"
tamil_file = BASE_DIR / "data" / "tamil_qa" / "tamil_qa.json"
tanglish_file = BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa.json"

report_dir = BASE_DIR / "reports"
report_dir.mkdir(exist_ok=True)

output_file = report_dir / "duplicate_translation_review.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


english = load_json(english_file)
tamil = load_json(tamil_file)
tanglish = load_json(tanglish_file)


rows = []

# --------------------------------------------------------
# Tamil
# --------------------------------------------------------

question_map = defaultdict(list)

for i, item in enumerate(tamil):
    question_map[item["question"]].append(i)

for question, indexes in question_map.items():

    if len(indexes) <= 1:
        continue

    for idx in indexes:

        rows.append({
            "Language": "Tamil",
            "Dataset_Index": idx,
            "ID": tamil[idx]["id"],
            "English_Question": english[idx]["question"],
            "Current_Question": tamil[idx]["question"],
            "Reference_Answer": tamil[idx]["reference_answer"]
        })

# --------------------------------------------------------
# Tanglish
# --------------------------------------------------------

question_map = defaultdict(list)

for i, item in enumerate(tanglish):
    question_map[item["question"]].append(i)

for question, indexes in question_map.items():

    if len(indexes) <= 1:
        continue

    for idx in indexes:

        rows.append({
            "Language": "Tanglish",
            "Dataset_Index": idx,
            "ID": tanglish[idx]["id"],
            "English_Question": tanglish[idx]["question_en"],
            "Current_Question": tanglish[idx]["question"],
            "Reference_Answer": tanglish[idx]["reference_answer"]
        })

# --------------------------------------------------------

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Language",
            "Dataset_Index",
            "ID",
            "English_Question",
            "Current_Question",
            "Reference_Answer"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)

print("=" * 60)
print("Duplicate Export Completed")
print(f"Saved : {output_file}")
print(f"Total Rows : {len(rows)}")
print("=" * 60)