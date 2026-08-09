from scripts.dataset_loader import (
    load_dataset,
    load_english,
    load_tamil,
    load_tanglish,
)


def test_english_loader():
    data = load_english()

    assert isinstance(data, list)
    assert len(data) == 300

    for record in data:
        assert "id" in record
        assert "question" in record
        assert "reference_answer" in record


def test_tamil_loader():
    data = load_tamil()

    assert isinstance(data, list)
    assert len(data) == 300

    for record in data:
        assert "id" in record
        assert "question" in record
        assert "reference_answer" in record


def test_tanglish_loader():
    data = load_tanglish()

    assert isinstance(data, list)
    assert len(data) == 300

    for record in data:
        assert "id" in record
        assert "question" in record
        assert "reference_answer" in record
        assert "question_en" in record


def test_dataset_selector():
    assert len(load_dataset("english")) == 300
    assert len(load_dataset("tamil")) == 300
    assert len(load_dataset("tanglish")) == 300


def test_invalid_language():
    try:
        load_dataset("invalid_language")
        assert False
    except ValueError:
        assert True