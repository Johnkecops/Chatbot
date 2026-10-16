"""Smoke tests: pytest tests/  (run from the repo root)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import triage  # noqa: E402


def test_feature_columns_match_ann_csv():
    cols = list(pd.read_csv(ROOT / "ANN_chatbot_files 1" / "data_tutorial.csv").columns[:26])
    assert cols == list(triage.SYMPTOM_DISPLAY) + list(triage.HISTORY_DISPLAY)


def _ticked(*names):
    return {k: int(k in names) for k in list(triage.SYMPTOM_DISPLAY) + list(triage.HISTORY_DISPLAY)}


def test_ann_red_flags():
    assert triage.ann_red_flags(_ticked("Chest_discomfort", "Faint"))
    assert triage.ann_red_flags(_ticked("Chest_discomfort", "Shortness_of_breath"))
    assert triage.ann_red_flags(_ticked("Trouble_in_speaking_seeingwalking"))
    assert not triage.ann_red_flags(_ticked("Chest_discomfort"))
    assert not triage.ann_red_flags(_ticked("Fever", "Diarrhea"))


@pytest.mark.parametrize("text,flagged", [
    ("I have chest pain and shortness of breath", True),
    ("my speech is slurred", True),
    ("I can't breathe", True),
    ("Who treats skin infections", False),
    ("My chest hurts after lifting", False),
])
def test_text_red_flags(text, flagged):
    assert bool(triage.text_red_flags(text)) == flagged


@pytest.fixture(scope="module")
def svm():
    return triage.build_svm(ROOT / "svm_data.csv")


def test_svm_cv_is_stratified_and_sane(svm):
    _, mean, _, folds = svm
    assert folds >= 3 and 0.5 < mean <= 1.0


def test_svm_rejects_off_topic_and_answers_in_domain(svm):
    pipe = svm[0]
    assert triage.svm_answer(pipe, "what is the weather today") == triage.NO_MATCH
    assert triage.svm_answer(pipe, "bitcoin price forecast") == triage.NO_MATCH
    assert "Orthopedic" in triage.svm_answer(pipe, "I broke my leg yesterday")
