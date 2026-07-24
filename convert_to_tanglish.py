import json
import time
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def create_true_tanglish():
    print("Loading datasets...")
    with open("data/tanglish_qa/tanglish_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize the translator (English to Tamil)
    translator = GoogleTranslator(source='en', target='ta')

    print(f"Converting {len(data)} English questions to true Tanglish...")

    for i, item in enumerate(data):
        eng_question = item["question_en"]
        
        try:
            # 1. Translate English to Pure Tamil Script
            tamil_script = translator.translate(eng_question)
            
            # 2. Transliterate Tamil Script to Roman/English Alphabet (Tanglish)
            tanglish_text = transliterate(tamil_script, sanscript.TAMIL, sanscript.ITRANS)
            
            # Update the JSON item
            item["question"] = tanglish_text.lower()
            
            if i % 25 == 0:
                print(f"[{i}/{len(data)}] converted...")
                
            time.sleep(0.2) # Small sleep to prevent API rate limiting
            
        except Exception as e:
            print(f"Failed at index {i}: {e}")
            item["question"] = eng_question # Fallback to English if translation fails

    # Save the updated Tanglish questions back to the JSON file
    with open("data/tanglish_qa/tanglish_qa.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print("Successfully converted all questions to Code-Mixed Tanglish!")
    
    # Print a sample to verify the pipeline worked
    print("\nSAMPLE VERIFICATION:")
    print("English:", data[0]['question_en'])
    print("Tanglish:", data[0]['question'])

if __name__ == "__main__":
    create_true_tanglish()