import json
import os
from datasets import load_dataset

def prepare_tanglish_benchmark(output_path="data/tanglish_qa/tanglish_qa.json"):
    print("Preparing 300 Code-Mixed (Tanglish) evaluation slots from SQuAD...")
    
    # Load SQuAD v1.1
    dataset = load_dataset("rajpurkar/squad", split="validation")
    
    tanglish_data = []
    
    for index, item in enumerate(dataset):
        # We take the English question and reference answer
        question_en = item["question"]
        answer_en = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else ""
        
        tanglish_data.append({
            "id": f"tanglish_{index}",
            "question_en": question_en,
            "question": question_en, # We will replace/map these with code-mixed prompts
            "reference_answer": answer_en
        })
        
        if len(tanglish_data) >= 300:
            break
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tanglish_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully created {len(tanglish_data)} Tanglish target slots in {output_path}")

if __name__ == "__main__":
    prepare_tanglish_benchmark()