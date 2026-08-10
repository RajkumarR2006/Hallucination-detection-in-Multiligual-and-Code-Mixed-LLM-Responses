"""Data Loading Utilities."""

import os
import json
import hashlib
from pathlib import Path

import datasets


def _load_custom_dataset(dataset_name, seed):
    """Load one of the project multilingual QA datasets."""

    base_dir = Path(__file__).resolve().parents[3]

    dataset_paths = {
        "english": base_dir / "data" / "english_qa" / "english_qa.json",
        "tamil": base_dir / "data" / "tamil_qa" / "tamil_qa_aligned.json",
        "tanglish": base_dir / "data" / "tanglish_qa" / "tanglish_qa_aligned.json",
    }

    dataset_path = dataset_paths[dataset_name]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Custom dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            f"Dataset must contain a JSON list: {dataset_path}"
        )

    formatted_records = []

    for record in records:

        if not record.get("id"):
            raise ValueError(
                f"Missing ID in {dataset_name} dataset"
            )

        if not record.get("question"):
            raise ValueError(
                f"Missing question in {dataset_name} dataset "
                f"for ID {record['id']}"
            )

        if not record.get("reference_answer"):
            raise ValueError(
                f"Missing reference answer in {dataset_name} dataset "
                f"for ID {record['id']}"
            )

        formatted_record = {
            "id": record["id"],
            "question": record["question"],
            "context": None,
            "answers": {
                "text": [record["reference_answer"]],
                "answer_start": [0],
            },
        }

        # Preserve the English source question for Tanglish.
        if dataset_name == "tanglish":
            formatted_record["question_en"] = record["question_en"]

        formatted_records.append(formatted_record)

    # Deterministic 80/20 split.
    dataset = datasets.Dataset.from_list(formatted_records)

    dataset = dataset.train_test_split(
        test_size=0.2,
        seed=seed,
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["test"]

    return train_dataset, validation_dataset


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""

    user = os.environ.get("USER", os.environ.get("USERNAME", "user"))

    train_dataset, validation_dataset = None, None

    # ---------------------------------------------------------
    # Original Semantic Entropy datasets
    # ---------------------------------------------------------

    if dataset_name == "squad":

        dataset = datasets.load_dataset("squad_v2")

        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == "svamp":

        dataset = datasets.load_dataset("ChilleD/SVAMP")

        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            "question": x["Question"],
            "context": x["Body"],
            "type": x["Type"],
            "equation": x["Equation"],
            "id": x["ID"],
            "answers": {
                "text": [str(x["Answer"])]
            },
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "nq":

        dataset = datasets.load_dataset("nq_open")

        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

        md5hash = lambda s: str(
            int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        )

        reformat = lambda x: {
            "question": x["question"] + "?",
            "answers": {
                "text": x["answer"]
            },
            "context": "",
            "id": md5hash(str(x["question"])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":

        dataset = datasets.load_dataset(
            "TimoImhof/TriviaQA-in-SQuAD-format"
        )["unmodified"]

        dataset = dataset.train_test_split(
            test_size=0.2,
            seed=seed,
        )

        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif dataset_name == "bioasq":

        scratch_dir = os.getenv("SCRATCH_DIR", ".")
        path = (
            f"{scratch_dir}/{user}/semantic_uncertainty/"
            "data/bioasq/training11b.json"
        )

        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]

        dataset_dict = {
            "question": [],
            "answers": [],
            "id": [],
        }

        for question in questions:

            if "exact_answer" not in question:
                continue

            dataset_dict["question"].append(
                question["body"]
            )

            if "exact_answer" in question:

                if isinstance(
                    question["exact_answer"],
                    list,
                ):

                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question["exact_answer"]
                    ]

                else:

                    exact_answers = [
                        question["exact_answer"]
                    ]

                dataset_dict["answers"].append(
                    {
                        "text": exact_answers,
                        "answer_start": [
                            0
                        ] * len(question["exact_answer"]),
                    }
                )

            else:

                dataset_dict["answers"].append(
                    {
                        "text": question["ideal_answer"],
                        "answer_start": [0],
                    }
                )

            dataset_dict["id"].append(
                question["id"]
            )

            dataset_dict["context"] = [
                None
            ] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(
            dataset_dict
        )

        dataset = dataset.train_test_split(
            test_size=0.8,
            seed=seed,
        )

        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    # ---------------------------------------------------------
    # Our multilingual benchmark
    # ---------------------------------------------------------

    elif dataset_name in {
        "english",
        "tamil",
        "tanglish",
    }:

        train_dataset, validation_dataset = (
            _load_custom_dataset(
                dataset_name,
                seed,
            )
        )

    else:

        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    return train_dataset, validation_dataset