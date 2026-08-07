import json
import os
import re

# ----------------------------
# Configuration
# ----------------------------
DATASETS = {
    "Tamil": "data/tamil_qa/tamil_qa.json",
    "Tanglish": "data/tanglish_qa/tanglish_qa.json",
    "English": "data/english_qa/english_qa.json"
}

# Tamil Unicode Range
TAMIL_PATTERN = re.compile(r'[\u0B80-\u0BFF]')


def validate_dataset(name, filepath):

    print("=" * 60)
    print(f"Checking {name}")
    print("=" * 60)

    if not os.path.exists(filepath):
        print("❌ File not found")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("❌ Invalid JSON")
        print(e)
        return

    ids = set()
    questions = set()

    duplicate_ids = 0
    duplicate_questions = 0
    empty_questions = 0
    empty_answers = 0
    missing_fields = 0
    tamil_errors = 0
    tanglish_errors = 0

    for item in data:

        # -----------------
        # Required fields
        # -----------------
        required = ["id", "question", "reference_answer"]

        if not all(field in item for field in required):
            missing_fields += 1
            continue

        # -----------------
        # Duplicate IDs
        # -----------------
        if item["id"] in ids:
            duplicate_ids += 1
        else:
            ids.add(item["id"])

        # -----------------
        # Duplicate Questions
        # -----------------
        if item["question"] in questions:
            duplicate_questions += 1
        else:
            questions.add(item["question"])

        # -----------------
        # Empty Question
        # -----------------
        if item["question"].strip() == "":
            empty_questions += 1

        # -----------------
        # Empty Answer
        # -----------------
        if item["reference_answer"].strip() == "":
            empty_answers += 1

        # -----------------
        # Language Check
        # -----------------
        if name == "Tamil":
            if not TAMIL_PATTERN.search(item["question"]):
                tamil_errors += 1

        if name == "Tanglish":
            if TAMIL_PATTERN.search(item["question"]):
                tanglish_errors += 1

    print(f"Total Records          : {len(data)}")
    print(f"Duplicate IDs          : {duplicate_ids}")
    print(f"Duplicate Questions    : {duplicate_questions}")
    print(f"Empty Questions        : {empty_questions}")
    print(f"Empty Answers          : {empty_answers}")
    print(f"Missing Fields         : {missing_fields}")

    if name == "Tamil":
        print(f"Invalid Tamil Entries  : {tamil_errors}")

    if name == "Tanglish":
        print(f"Tamil Script Found     : {tanglish_errors}")

    print()


# ----------------------------
# Run Validation
# ----------------------------
for name, path in DATASETS.items():
    validate_dataset(name, path)