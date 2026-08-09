# Multilingual and Code-Mixed QA Dataset Card

## 1. Overview

This project uses three QA datasets for multilingual and code-mixed
hallucination-detection experiments:

- English
- Tamil
- Tanglish

Each dataset currently contains 300 QA records.

The English dataset is derived from the SQuAD v2 development set.
The Tamil dataset represents Tamil-language QA data.
The Tanglish dataset represents Tamil-English code-mixed QA data.

---

## 2. English Dataset

### Source

The English benchmark subset is derived from:

SQuAD v2 development dataset (`dev-v2.0.json`).

A 300-record subset was recovered from the source dataset using the
original SQuAD question IDs.

### Validation

The English subset was validated against the SQuAD source.

Results:

- Records requested: 300
- Records recovered: 300
- Missing source IDs: 0
- Question mismatches: 0
- Reference-answer mismatches: 0

Therefore, the current English subset is verified against its SQuAD
source.

---

## 3. Tamil Dataset

The Tamil dataset contains 300 QA records.

Each record follows the common QA structure:

- `id`
- `question`
- `reference_answer`

Tamil questions were checked for:

- valid JSON structure
- required fields
- empty questions
- empty reference answers
- duplicate IDs
- Tamil Unicode/script usage

Current validation results:

- Records: 300
- Unique IDs: 300
- Duplicate IDs: 0
- Empty questions: 0
- Empty answers: 0
- Questions containing Tamil script: 300

---

## 4. Tanglish Dataset

The Tanglish dataset contains 300 QA records representing
Tamil-English code-mixed questions.

Records contain:

- `id`
- `question_en`
- `question`
- `reference_answer`

Validation covered:

- JSON formatting
- required fields
- empty questions
- empty reference answers
- duplicate IDs
- English-source question field
- Roman-script usage

Current validation results:

- Records: 300
- Unique IDs: 300
- Duplicate IDs: 0
- Empty questions: 0
- Empty answers: 0
- Records containing `question_en`: 300
- Questions containing Latin script: 300

---

## 5. Dataset Schema

The common QA fields used by the project are:

```text
id
question
reference_answer