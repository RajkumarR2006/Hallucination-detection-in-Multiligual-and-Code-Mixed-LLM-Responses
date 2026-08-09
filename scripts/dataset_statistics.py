import json
import re
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "English": BASE_DIR / "data" / "english_qa" / "english_qa.json",
    "Tamil": BASE_DIR / "data" / "tamil_qa" / "tamil_qa.json",
    "Tanglish": BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa.json",
}

REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "dataset_statistics_report.txt"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def has_tamil(text):
    return bool(re.search(r"[\u0B80-\u0BFF]", text))


def has_latin(text):
    return bool(re.search(r"[A-Za-z]", text))


def analyze(name, records):
    ids = [record.get("id") for record in records]

    questions = [
        str(record.get("question", "")).strip()
        for record in records
    ]

    answers = [
        str(record.get("reference_answer", "")).strip()
        for record in records
    ]

    duplicate_ids = len(ids) - len(set(ids))

    empty_questions = sum(not q for q in questions)
    empty_answers = sum(not a for a in answers)

    question_lengths = [
        len(q.split())
        for q in questions
        if q
    ]

    answer_lengths = [
        len(a.split())
        for a in answers
        if a
    ]

    result = {
        "records": len(records),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "empty_questions": empty_questions,
        "empty_answers": empty_answers,
        "avg_question_words": mean(question_lengths)
        if question_lengths else 0,
        "avg_answer_words": mean(answer_lengths)
        if answer_lengths else 0,
    }

    if name == "Tamil":
        result["questions_with_tamil_script"] = sum(
            has_tamil(q) for q in questions
        )

    if name == "Tanglish":
        result["questions_with_latin_script"] = sum(
            has_latin(q) for q in questions
        )

        result["questions_with_tamil_script"] = sum(
            has_tamil(q) for q in questions
        )

        result["questions_with_question_en"] = sum(
            bool(str(record.get("question_en", "")).strip())
            for record in records
        )

    return result


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("DATASET STATISTICS AND QUALITY SUMMARY")
    print("=" * 70)

    results = {}

    for name, path in DATASETS.items():

        records = load_dataset(path)

        print(f"\n{name}")
        print("-" * 40)

        result = analyze(name, records)
        results[name] = result

        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key:35}: {value:.2f}")
            else:
                print(f"{key:35}: {value}")

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        file.write(
            "Dataset Statistics and Quality Summary\n"
        )
        file.write("=" * 60 + "\n\n")

        for name, result in results.items():

            file.write(f"{name}\n")
            file.write("-" * 40 + "\n")

            for key, value in result.items():

                if isinstance(value, float):
                    file.write(
                        f"{key}: {value:.2f}\n"
                    )
                else:
                    file.write(
                        f"{key}: {value}\n"
                    )

            file.write("\n")

    print("\n" + "=" * 70)
    print("REPORT CREATED")
    print("=" * 70)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()