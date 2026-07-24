import json
import os

def load_dataset_by_name(dataset_name):
    """
    Loads English, Tamil, or Tanglish QA datasets in a unified structure.
    """
    if dataset_name == "tamil_qa":
        path = "data/tamil_qa/tamil_qa.json"
    elif dataset_name == "tanglish_qa":
        path = "data/tanglish_qa/tanglish_qa.json"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded '{dataset_name}' with {len(data)} samples.")
    return data

if __name__ == "__main__":
    # Load both custom datasets
    tamil_data = load_dataset_by_name("tamil_qa")
    tanglish_data = load_dataset_by_name("tanglish_qa")
    
    print("\n" + "="*40)
    print("SAMPLE TAMIL QUESTION:")
    print("Q:", tamil_data[0]['question'])
    print("A:", tamil_data[0]['reference_answer'])
    
    print("\n" + "="*40)
    print("SAMPLE TANGLISH QUESTION:")
    print("Q:", tanglish_data[0]['question'])
    print("A:", tanglish_data[0]['reference_answer'])
    print("="*40)