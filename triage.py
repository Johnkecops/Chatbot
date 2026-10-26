#!/usr/bin/env python3
"""
Module: triage
Purpose: Streamlit-free logic for the chatbot: red-flag rules and the SVM Q&A model
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - csv_path: Q&A training data (columns: question, Response)
References:
    - Matthew R et al. (2022) ICORIS, doi:10.1109/ICORIS56080.2022.10031400
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Feature display names (26 ANN inputs) ─────────────────────────────────────
SYMPTOM_DISPLAY = {
    "Chest_discomfort"                    : "Chest Discomfort",
    "Faint"                               : "Faint / Dizziness",
    "Discomfort_in_shoulder_neckback_and_jaw": "Shoulder / Neck / Back / Jaw Discomfort",
    "Shortness_of_breath"                 : "Shortness of Breath",
    "Fractured_bone"                      : "Fractured Bone",
    "Persistent_headache"                 : "Persistent Headache",
    "Cuts"                                : "Cuts / Wounds",
    "Foreign_material_stuck"              : "Foreign Material Stuck",
    "Skin_infection"                      : "Skin Infection",
    "Problem_with_vision"                 : "Vision Problems",
    "Fever"                               : "Fever",
    "Diarrhea"                            : "Diarrhea",
    "Cold_sweat"                          : "Cold Sweat",
    "Back_pain"                           : "Back Pain",
    "Weakness"                            : "General Weakness",
    "Trouble_in_speaking_seeingwalking"   : "Speech / Vision / Walking Trouble",
    "Vomit"                               : "Vomiting",
    "Stomachache"                         : "Stomachache",
}

HISTORY_DISPLAY = {
    "Hypertension"                        : "Hypertension (pre-existing)",
    "Diabetes"                            : "Diabetes (pre-existing)",
    "Heart_disease"                       : "Heart Disease (pre-existing)",
    "Liver_disease"                       : "Liver Disease (pre-existing)",
    "Stroke"                              : "Stroke History",
    "Tumor/cancer"                        : "Tumor / Cancer History",
    "Respiration_disease"                 : "Respiratory Disease (pre-existing)",
    "Gastric"                             : "Gastric Disease (pre-existing)",
}

# Clinical rules below approved by the project lead (Dr. A. A. Parikesit, 2026-10-16).
# Not reviewed by a licensed physician: obtain that before any real clinical use.
CARDIAC = ("Chest discomfort with breathlessness or fainting can be a heart emergency. "
           "Go to the nearest emergency department now.")
STROKE = ("Sudden trouble speaking, seeing or walking can be a stroke. "
          "Go to the nearest emergency department now.")

# (symptoms that must all be ticked, at least one of these, message); empty any_of = ignored
ANN_RED_FLAGS = [
    ({"Chest_discomfort"}, {"Faint", "Shortness_of_breath"}, CARDIAC),
    ({"Trouble_in_speaking_seeingwalking"}, set(), STROKE),
]

# \A anchors the lookaheads: unanchored, search() retries them at every offset (O(n^2) on long input)
TEXT_RED_FLAGS = [
    (re.compile(r"\A(?=.*\bchest\b)(?=.*\b(breath|breathing|faint|fainting)\b)", re.I | re.S), CARDIAC),
    (re.compile(r"\A(?=.*\b(trouble|difficulty|unable|cannot|can't|hardly)\b)"
                r"(?=.*\b(speak|speaking|talking|walk|walking|seeing)\b)|slurred", re.I | re.S), STROKE),
    (re.compile(r"can'?t breathe|cannot breathe", re.I), CARDIAC),
]

# ponytail: set from 12 unseen in-domain vs 10 off-topic probes (2026-10-16): in-domain
# margins -0.62..0.52, off-topic -0.91 (no known word) and one at -0.55. At -0.6 this is
# mostly an out-of-vocabulary filter, not a semantic one. Recalibrate on a labelled
# out-of-scope set.
REJECT_MARGIN = -0.6
NO_MATCH = ("I cannot match that to a known question. Please rephrase, "
            "or consult a licensed physician.")


def ann_red_flags(values: dict) -> list[str]:
    return [msg for need, any_of, msg in ANN_RED_FLAGS
            if all(values[k] for k in need) and (not any_of or any(values[k] for k in any_of))]


def text_red_flags(text: str) -> list[str]:
    return list(dict.fromkeys(msg for rx, msg in TEXT_RED_FLAGS if rx.search(text)))


def build_svm(csv_path):
    """Return (fitted pipeline, CV mean accuracy, CV sd, n folds)."""
    df = pd.read_csv(csv_path)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("svm", LinearSVC(C=1.0, max_iter=2000)),
    ])
    k = min(5, int(df["Response"].value_counts().min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, df["question"], df["Response"], cv=cv)
    pipe.fit(df["question"], df["Response"])
    return pipe, float(scores.mean()), float(scores.std()), k


def svm_answer(pipe, text: str) -> str:
    """Best answer, or NO_MATCH when the top one-vs-rest margin is below REJECT_MARGIN."""
    if pipe.decision_function([text]).max() < REJECT_MARGIN:
        return NO_MATCH
    return pipe.predict([text])[0]
