import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

ENGLISH_FILE = BASE_DIR / "data" / "english_qa" / "english_qa.json"
TAMIL_FILE = BASE_DIR / "data" / "tamil_qa" / "tamil_qa_aligned.json"
TANGLISH_FILE = BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa_aligned.json"

REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "cross_language_alignment_report.txt"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("=" * 70)
    print("CROSS-LANGUAGE ALIGNMENT CHECK")
    print("=" * 70)

    english = load_json(ENGLISH_FILE)
    tamil = load_json(TAMIL_FILE)
    tanglish = load_json(TANGLISH_FILE)

    print(f"\nEnglish records  : {len(english)}")
    print(f"Tamil records    : {len(tamil)}")
    print(f"Tanglish records : {len(tanglish)}")

    passed = True
    problems = []

    # ---------------------------------------------------------
    # 1. Record count
    # ---------------------------------------------------------

    if not (
        len(english) == 300
        and len(tamil) == 300
        and len(tanglish) == 300
    ):
        passed = False
        problems.append("Record count mismatch")

    # ---------------------------------------------------------
    # 2. Build ID maps
    # ---------------------------------------------------------

    english_ids = [record.get("id") for record in english]
    tamil_ids = [record.get("id") for record in tamil]
    tanglish_ids = [record.get("id") for record in tanglish]

    # Duplicate IDs
    if len(english_ids) != len(set(english_ids)):
        passed = False
        problems.append("Duplicate English IDs")

    if len(tamil_ids) != len(set(tamil_ids)):
        passed = False
        problems.append("Duplicate Tamil IDs")

    if len(tanglish_ids) != len(set(tanglish_ids)):
        passed = False
        problems.append("Duplicate Tanglish IDs")

    # ---------------------------------------------------------
    # 3. Exact ID alignment
    # ---------------------------------------------------------

    english_id_set = set(english_ids)
    tamil_id_set = set(tamil_ids)
    tanglish_id_set = set(tanglish_ids)

    missing_tamil = english_id_set - tamil_id_set
    extra_tamil = tamil_id_set - english_id_set

    missing_tanglish = english_id_set - tanglish_id_set
    extra_tanglish = tanglish_id_set - english_id_set

    if missing_tamil:
        passed = False
        problems.append(
            f"Missing Tamil IDs: {len(missing_tamil)}"
        )

    if extra_tamil:
        passed = False
        problems.append(
            f"Extra Tamil IDs: {len(extra_tamil)}"
        )

    if missing_tanglish:
        passed = False
        problems.append(
            f"Missing Tanglish IDs: {len(missing_tanglish)}"
        )

    if extra_tanglish:
        passed = False
        problems.append(
            f"Extra Tanglish IDs: {len(extra_tanglish)}"
        )

    # ---------------------------------------------------------
    # 4. Record-order alignment
    # ---------------------------------------------------------

    if english_ids != tamil_ids:
        passed = False
        problems.append("Tamil record order does not match English")

    if english_ids != tanglish_ids:
        passed = False
        problems.append("Tanglish record order does not match English")

    # ---------------------------------------------------------
    # 5. Tanglish English-source question alignment
    # ---------------------------------------------------------

    tanglish_question_mismatches = []

    for index, english_record in enumerate(english):

        english_question = english_record.get("question", "")
        tanglish_record = tanglish[index]

        question_en = tanglish_record.get("question_en", "")

        if english_question != question_en:
            tanglish_question_mismatches.append(
                {
                    "index": index,
                    "id": english_record.get("id"),
                    "english": english_question,
                    "tanglish_question_en": question_en,
                }
            )

    if tanglish_question_mismatches:
        passed = False
        problems.append(
            "Tanglish question_en mismatches: "
            f"{len(tanglish_question_mismatches)}"
        )

    # ---------------------------------------------------------
    # 6. Required fields
    # ---------------------------------------------------------

    for index, record in enumerate(tamil):

        if not record.get("id"):
            passed = False
            problems.append(
                f"Tamil record {index} missing ID"
            )

        if not record.get("question"):
            passed = False
            problems.append(
                f"Tamil record {index} missing question"
            )

        if not record.get("reference_answer"):
            passed = False
            problems.append(
                f"Tamil record {index} missing reference answer"
            )

    for index, record in enumerate(tanglish):

        if not record.get("id"):
            passed = False
            problems.append(
                f"Tanglish record {index} missing ID"
            )

        if not record.get("question_en"):
            passed = False
            problems.append(
                f"Tanglish record {index} missing question_en"
            )

        if not record.get("question"):
            passed = False
            problems.append(
                f"Tanglish record {index} missing question"
            )

        if not record.get("reference_answer"):
            passed = False
            problems.append(
                f"Tanglish record {index} missing reference answer"
            )

    # ---------------------------------------------------------
    # 7. Generate report
    # ---------------------------------------------------------

    REPORT_DIR.mkdir(exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        file.write("Cross-Language Alignment Report\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"English records  : {len(english)}\n")
        file.write(f"Tamil records    : {len(tamil)}\n")
        file.write(f"Tanglish records : {len(tanglish)}\n\n")

        file.write(
            f"English/Tamil IDs aligned     : "
            f"{len(english_id_set & tamil_id_set)}\n"
        )

        file.write(
            f"English/Tanglish IDs aligned  : "
            f"{len(english_id_set & tanglish_id_set)}\n"
        )

        file.write(
            f"Tanglish question_en matches  : "
            f"{len(english) - len(tanglish_question_mismatches)}\n"
        )

        file.write("\n")

        if passed:
            file.write("STATUS: PASS\n")
            file.write("\n")
            file.write(
                "All 300 English, Tamil, and Tanglish records "
                "share the same canonical IDs.\n"
            )
            file.write(
                "Tanglish question_en values exactly match the "
                "English source questions.\n"
            )
            file.write(
                "Required fields are present and non-empty.\n"
            )
        else:
            file.write("STATUS: FAIL\n\n")

            for problem in problems:
                file.write(f"- {problem}\n")

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print("\nID alignment")
    print("-" * 40)
    print(
        "English ↔ Tamil      :",
        len(english_id_set & tamil_id_set),
        "/",
        len(english)
    )
    print(
        "English ↔ Tanglish   :",
        len(english_id_set & tanglish_id_set),
        "/",
        len(english)
    )

    print("\nTanglish source-question alignment")
    print("-" * 40)
    print(
        "Exact question_en matches :",
        len(english) - len(tanglish_question_mismatches),
        "/",
        len(english)
    )

    print("\n" + "=" * 70)

    if passed:
        print("ALIGNMENT STATUS : PASS")
        print("=" * 70)
        print(f"\nReport: {REPORT_FILE}")
    else:
        print("ALIGNMENT STATUS : FAIL")
        print("=" * 70)

        for problem in problems:
            print(" -", problem)

        print(f"\nReport: {REPORT_FILE}")


if __name__ == "__main__":
    main()