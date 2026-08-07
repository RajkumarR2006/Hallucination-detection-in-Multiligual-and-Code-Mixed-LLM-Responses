"""
Dataset Validation Utility
---------------------------------------------------------
Project:
Hallucination Detection in Multilingual and Code-Mixed LLM Responses

Author:
Member 1 - Dataset Engineering

Purpose:
    Validate English, Tamil and Tanglish QA datasets.

Checks Performed:
    ✓ JSON validity
    ✓ Required fields
    ✓ Record count
    ✓ Duplicate IDs
    ✓ Duplicate Questions
    ✓ Empty Questions
    ✓ Empty Answers
    ✓ Tamil Unicode validation
    ✓ Tanglish script validation
    ✓ Generates validation report

Usage:
    python scripts/validate_dataset.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Tamil": BASE_DIR / "data" / "tamil_qa" / "tamil_qa.json",
    "Tanglish": BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa.json",
    "English": BASE_DIR / "data" / "english_qa" / "english_qa.json",
}

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "dataset_validation_report.txt"

# ---------------------------------------------------------
# Language Patterns
# ---------------------------------------------------------

TAMIL_PATTERN = re.compile(r'[\u0B80-\u0BFF]')

# ---------------------------------------------------------
# Validation Function
# ---------------------------------------------------------

def validate_dataset(dataset_name, filepath):

    output = []

    output.append("=" * 80)
    output.append(f"DATASET : {dataset_name}")
    output.append("=" * 80)

    if not filepath.exists():
        output.append("❌ Dataset not found.\n")
        return output

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        output.append("❌ Invalid JSON")
        output.append(str(e))
        output.append("")
        return output

    ids = set()

    question_map = defaultdict(list)

    duplicate_ids = 0
    empty_questions = 0
    empty_answers = 0
    missing_fields = 0
    invalid_language = 0

    REQUIRED_FIELDS = [
        "id",
        "question",
        "reference_answer"
    ]

    # -----------------------------------------------------

    for item in data:

        if not all(field in item for field in REQUIRED_FIELDS):
            missing_fields += 1
            continue

        if item["id"] in ids:
            duplicate_ids += 1
        else:
            ids.add(item["id"])

        question_map[item["question"]].append(item["id"])

        if item["question"].strip() == "":
            empty_questions += 1

        if item["reference_answer"].strip() == "":
            empty_answers += 1

        if dataset_name == "Tamil":

            if not TAMIL_PATTERN.search(item["question"]):
                invalid_language += 1

        elif dataset_name == "Tanglish":

            if TAMIL_PATTERN.search(item["question"]):
                invalid_language += 1

    duplicates = {
        question: ids
        for question, ids in question_map.items()
        if len(ids) > 1
    }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    output.append(f"Total Records          : {len(data)}")
    output.append(f"Duplicate IDs          : {duplicate_ids}")
    output.append(f"Duplicate Questions    : {len(duplicates)}")
    output.append(f"Empty Questions        : {empty_questions}")
    output.append(f"Empty Answers          : {empty_answers}")
    output.append(f"Missing Fields         : {missing_fields}")

    if dataset_name == "Tamil":
        output.append(f"Invalid Tamil Entries  : {invalid_language}")

    elif dataset_name == "Tanglish":
        output.append(f"Tamil Script Found     : {invalid_language}")

    # ---------------------------------------------------------
    # Duplicate Details
    # ---------------------------------------------------------

    if len(duplicates):

        output.append("")
        output.append("Duplicate Question Details")
        output.append("-" * 80)

        for index, (question, ids) in enumerate(duplicates.items(), start=1):

            output.append(f"\nDuplicate #{index}")

            output.append(f"Question :")

            output.append(question)

            output.append("IDs :")

            for i in ids:
                output.append(f"   • {i}")

            output.append("-" * 80)

    else:

        output.append("")
        output.append("No Duplicate Questions Found.")

    output.append("")

    return output

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    final_report = []

    print("\n")
    print("=" * 80)
    print("MULTILINGUAL DATASET VALIDATION REPORT")
    print("=" * 80)

    for dataset_name, filepath in DATASETS.items():

        report = validate_dataset(dataset_name, filepath)

        for line in report:
            print(line)

        final_report.extend(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:

        for line in final_report:
            f.write(line + "\n")

    print("=" * 80)
    print("Validation Completed Successfully")
    print(f"Report Saved : {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()