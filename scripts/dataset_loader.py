import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATHS = {
    "english": BASE_DIR / "data" / "english_qa" / "english_qa.json",
    "tamil": BASE_DIR / "data" / "tamil_qa" / "tamil_qa.json",
    "tanglish": BASE_DIR / "data" / "tanglish_qa" / "tanglish_qa.json",
}


def load_dataset(language):
    """
    Load a QA dataset by language.

    Supported languages:
        english
        tamil
        tanglish

    Returns:
        list[dict]: QA records
    """

    language = language.lower()

    if language not in DATASET_PATHS:
        supported = ", ".join(DATASET_PATHS.keys())
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported languages: {supported}"
        )

    dataset_path = DATASET_PATHS[language]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Dataset must contain a JSON list: {dataset_path}"
        )

    return data


def load_english():
    return load_dataset("english")


def load_tamil():
    return load_dataset("tamil")


def load_tanglish():
    return load_dataset("tanglish")


def get_dataset(language):
    """
    Alias for load_dataset().
    """
    return load_dataset(language)


def dataset_size(language):
    return len(load_dataset(language))


if __name__ == "__main__":

    print("=" * 60)
    print("DATASET LOADER TEST")
    print("=" * 60)

    for language in DATASET_PATHS:

        data = load_dataset(language)

        print(f"{language.capitalize():10} : {len(data)} records")

    print("\nLoader test completed successfully.")