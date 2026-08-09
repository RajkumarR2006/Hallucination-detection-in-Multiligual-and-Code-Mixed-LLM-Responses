import json
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SQUAD_FILE = BASE_DIR / "data" / "squad_v2" / "dev-v2.0.json"
ENGLISH_FILE = BASE_DIR / "data" / "english_qa" / "english_qa.json"

# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

print("=" * 70)
print("BUILDING ENGLISH SQuAD-DERIVED SUBSET")
print("=" * 70)

with open(SQUAD_FILE, "r", encoding="utf-8") as f:
    squad = json.load(f)

with open(ENGLISH_FILE, "r", encoding="utf-8") as f:
    current_english = json.load(f)

print(f"SQuAD source loaded      : {SQUAD_FILE}")
print(f"Current English records  : {len(current_english)}")

# ---------------------------------------------------------
# Build lookup from SQuAD
# ---------------------------------------------------------

squad_lookup = {}

for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:

            # We only use answerable SQuAD questions
            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            squad_lookup[qa["id"]] = {
                "question": qa["question"],
                "reference_answer": answers[0]["text"]
            }

print(f"SQuAD answerable questions indexed : {len(squad_lookup)}")

# ---------------------------------------------------------
# Recover the 300-item subset
# ---------------------------------------------------------

rebuilt = []

missing_ids = []
question_mismatches = []
answer_mismatches = []

for item in current_english:

    item_id = item["id"]

    if item_id not in squad_lookup:
        missing_ids.append(item_id)
        continue

    source = squad_lookup[item_id]

    # Verify question
    if item["question"].strip() != source["question"].strip():
        question_mismatches.append({
            "id": item_id,
            "current": item["question"],
            "source": source["question"]
        })

    # Verify reference answer
    if item["reference_answer"].strip() != source["reference_answer"].strip():
        answer_mismatches.append({
            "id": item_id,
            "current": item["reference_answer"],
            "source": source["reference_answer"]
        })

    rebuilt.append({
        "id": item_id,
        "question": source["question"],
        "reference_answer": source["reference_answer"]
    })

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SOURCE ALIGNMENT CHECK")
print("=" * 70)

print(f"Requested records       : {len(current_english)}")
print(f"Recovered from SQuAD    : {len(rebuilt)}")
print(f"Missing SQuAD IDs       : {len(missing_ids)}")
print(f"Question mismatches     : {len(question_mismatches)}")
print(f"Answer mismatches       : {len(answer_mismatches)}")

if missing_ids:
    print("\nMissing IDs:")
    for item_id in missing_ids[:20]:
        print(f"  - {item_id}")

if question_mismatches:
    print("\nQuestion mismatches:")
    for item in question_mismatches[:10]:
        print(f"  - {item['id']}")

if answer_mismatches:
    print("\nAnswer mismatches:")
    for item in answer_mismatches[:10]:
        print(f"  - {item['id']}")

# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

if (
    len(rebuilt) != len(current_english)
    or missing_ids
    or question_mismatches
    or answer_mismatches
):
    print("\n❌ English subset was NOT replaced.")
    print("Fix the reported alignment problems first.")
    raise SystemExit(1)

# ---------------------------------------------------------
# Write rebuilt dataset
# ---------------------------------------------------------

with open(ENGLISH_FILE, "w", encoding="utf-8") as f:
    json.dump(
        rebuilt,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)
print(f"English records written : {len(rebuilt)}")
print(f"Output                  : {ENGLISH_FILE}")
print("Source                  : SQuAD v2 dev-v2.0.json")
print("=" * 70)

