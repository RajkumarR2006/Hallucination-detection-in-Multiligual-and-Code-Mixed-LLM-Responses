import json
import pandas as pd
from datasets import load_dataset

def prepare_tamil_dataset(output_path="data/tamil_qa/tamil_qa.json"):
    print("Downloading Tamil QA dataset from Hugging Face...")
    dataset = load_dataset("AswiN037/tamil-question-answering-dataset", split="train")
    df = dataset.to_pandas()
    
    formatted_data = []
    
    for index, row in df.iterrows():
        formatted_data.append({
            "id": f"tamil_qa_{index}",
            "question": row["question"],
            "reference_answer": row["answer_text"]
        })
        
        # Updated to 300 questions for robust evaluation
        if len(formatted_data) >= 300:
            break
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(formatted_data)} Tamil QA pairs to {output_path}")

if __name__ == "__main__":
    prepare_tamil_dataset()