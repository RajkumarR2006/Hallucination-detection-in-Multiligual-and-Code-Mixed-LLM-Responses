import json

INPUT_FILE = "data/squad_v2/dev-v2.0.json"
OUTPUT_FILE = "data/english_qa/english_qa.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    squad = json.load(f)

english = []

for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if qa["is_impossible"]:
                continue

            answers = qa["answers"]
            if not answers:
                continue

            english.append({
                "id": qa["id"],
                "question": qa["question"],
                "reference_answer": answers[0]["text"]
            })

            if len(english) == 300:
                break
        if len(english) == 300:
            break
    if len(english) == 300:
        break

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(english, f, indent=4, ensure_ascii=False)

print(f"Saved {len(english)} questions.")