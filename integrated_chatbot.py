#!/usr/bin/env python3
"""
Integrated Medical Specialist Chatbot
Combines ANN (symptom-based) and SVM (free-text Q&A) chatbots
Author: Dr. Arli Aditya Parikesit
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
ANN_DIR = ROOT / "ANN_chatbot_files 1"
SVM_CSV = ROOT / "svm_data.csv"

# ── Specialist display names ───────────────────────────────────────────────────
SPECIALIST_NAMES = {
    "SpJP" : "Cardiologist (SpJP)",
    "SpPD" : "Internist (SpPD)",
    "SpS"  : "Neurologist (SpS)",
    "SpB"  : "General Surgeon (SpB)",
    "THT"  : "ENT Specialist (THT)",
    "SpKK" : "Dermatologist (SpKK)",
    "SpM"  : "Ophthalmologist (SpM)",
    "Paru" : "Pulmonologist (Paru)",
    "SpOT" : "Orthopedic Surgeon (SpOT)",
}

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
    "Hypertension"                        : "Hypertension (pre-existing)",
    "Diabetes"                            : "Diabetes (pre-existing)",
    "Heart_disease"                       : "Heart Disease (pre-existing)",
    "Liver_disease"                       : "Liver Disease (pre-existing)",
    "Stroke"                              : "Stroke History",
    "Tumor/cancer"                        : "Tumor / Cancer History",
    "Respiration_disease"                 : "Respiratory Disease (pre-existing)",
    "Gastric"                             : "Gastric Disease (pre-existing)",
}

SYMPTOMS_COLS   = list(SYMPTOM_DISPLAY.keys())[:18]   # acute symptoms
HISTORY_COLS    = list(SYMPTOM_DISPLAY.keys())[18:]   # medical history


# ── Model loaders (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ANN model…")
def load_ann():
    import tensorflow as tf
    model   = tf.keras.models.load_model(str(ANN_DIR / "chatbot_model1.h5"))
    df      = pd.read_csv(str(ANN_DIR / "data_tutorial.csv"))
    features    = list(df.iloc[:, :26].columns)
    specialists = list(df.iloc[:, -9:].columns)
    return model, features, specialists


@st.cache_resource(show_spinner="Training SVM model on sample data…")
def load_svm():
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(str(SVM_CSV))
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("svm",   LinearSVC(C=1.0, max_iter=2000)),
    ])
    pipeline.fit(df["question"], df["Response"])
    return pipeline


# ── ANN prediction ─────────────────────────────────────────────────────────────
def predict_ann(model, features, specialists, values: dict) -> tuple[str, float]:
    row = [values.get(f, 0) for f in features]
    df  = pd.DataFrame([row], columns=features)
    pred = model.predict(df, verbose=0)
    idx  = int(np.argmax(pred[0]))
    return specialists[idx], float(pred[0][idx])


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Specialist Chatbot",
    page_icon="🏥",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Specialist Recommendation Chatbot")
st.caption(
    "This tool assists in identifying the appropriate medical specialist based on "
    "your symptoms or questions. It is for **informational purposes only** and does "
    "not replace professional medical advice."
)
st.divider()

# ── Model selector ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select Chatbot Model",
    ["🧠 ANN — Symptom Checker", "📊 SVM — Free-Text Q&A"],
    help=(
        "**ANN** uses a trained neural network. Tick your symptoms and "
        "get a specialist recommendation.\n\n"
        "**SVM** uses a support vector machine trained on Q&A pairs. "
        "Type a question in plain language."
    ),
)

st.sidebar.divider()
st.sidebar.markdown(
    "**About the models**\n\n"
    "- **ANN**: Multi-layer perceptron trained on 26 binary symptom/history "
    "features, predicts 1 of 9 specialist categories.\n\n"
    "- **SVM**: LinearSVC with TF-IDF features trained on curated medical Q&A "
    "pairs covering common symptoms and specialist descriptions."
)

# ══════════════════════════════════════════════════════════════════════════════
# ANN TAB
# ══════════════════════════════════════════════════════════════════════════════
if model_choice.startswith("🧠"):
    st.subheader("🧠 ANN Symptom Checker")
    st.markdown(
        "Tick all symptoms and pre-existing conditions that apply, then press "
        "**Predict Specialist**."
    )

    try:
        ann_model, ann_features, ann_specialists = load_ann()
    except Exception as e:
        st.error(
            f"Could not load ANN model. Make sure `chatbot_model1.h5` and "
            f"`data_tutorial.csv` are inside `ANN_chatbot_files 1/`.\n\nError: {e}"
        )
        st.stop()

    # ── Symptom checkboxes ─────────────────────────────────────────────────────
    st.markdown("#### Current Symptoms")
    sym_cols = st.columns(3)
    sym_values: dict[str, int] = {}

    for i, col in enumerate(SYMPTOMS_COLS):
        c = sym_cols[i % 3]
        sym_values[col] = int(
            c.checkbox(SYMPTOM_DISPLAY[col], key=f"sym_{col}")
        )

    st.markdown("#### Medical History / Pre-existing Conditions")
    hist_cols = st.columns(4)
    for i, col in enumerate(HISTORY_COLS):
        c = hist_cols[i % 4]
        sym_values[col] = int(
            c.checkbox(SYMPTOM_DISPLAY[col], key=f"hist_{col}")
        )

    st.markdown("")
    predict_btn = st.button("🔍 Predict Specialist", type="primary", use_container_width=True)

    if predict_btn:
        n_checked = sum(sym_values.values())
        if n_checked == 0:
            st.warning("Please tick at least one symptom before predicting.")
        else:
            with st.spinner("Running ANN prediction…"):
                spec_code, confidence = predict_ann(
                    ann_model, ann_features, ann_specialists, sym_values
                )
            spec_name = SPECIALIST_NAMES.get(spec_code, spec_code)
            checked_labels = [
                SYMPTOM_DISPLAY[k] for k, v in sym_values.items() if v == 1
            ]

            st.success("Prediction complete")
            result_col, detail_col = st.columns([1, 2])

            with result_col:
                st.metric("Recommended Specialist", spec_name)
                st.metric("Model Confidence", f"{confidence:.1%}")

            with detail_col:
                st.markdown("**Symptoms / History entered:**")
                for lbl in checked_labels:
                    st.markdown(f"- {lbl}")

            st.info(
                "⚠️ This is a computational prediction. Please consult a licensed "
                "physician for proper diagnosis and treatment."
            )

    # ── Reset ──────────────────────────────────────────────────────────────────
    if st.button("🔄 Reset all", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("sym_") or key.startswith("hist_"):
                del st.session_state[key]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SVM TAB
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("📊 SVM Free-Text Q&A Chatbot")
    st.markdown(
        "Ask a question about symptoms, medical conditions, or which specialist to consult. "
        "The SVM model will match your query to the most relevant answer."
    )

    try:
        svm_pipeline = load_svm()
    except Exception as e:
        st.error(f"Could not load SVM model.\n\nError: {e}")
        st.stop()

    # ── Chat history ───────────────────────────────────────────────────────────
    if "svm_history" not in st.session_state:
        st.session_state.svm_history = []

    # ── Input ──────────────────────────────────────────────────────────────────
    with st.form("svm_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. I have chest pain and shortness of breath, which doctor should I see?",
        )
        submitted = st.form_submit_button("💬 Ask", type="primary", use_container_width=True)

    if submitted and user_input.strip():
        with st.spinner("Thinking…"):
            answer = svm_pipeline.predict([user_input.strip()])[0]
        st.session_state.svm_history.append(
            {"role": "user",      "text": user_input.strip()}
        )
        st.session_state.svm_history.append(
            {"role": "assistant", "text": answer}
        )

    # ── Render chat history ────────────────────────────────────────────────────
    for msg in reversed(st.session_state.svm_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["text"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(msg["text"])
                st.caption("⚠️ For informational purposes only. Consult a licensed physician.")

    if st.session_state.svm_history:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.svm_history = []
            st.rerun()
    else:
        st.markdown(
            "**Example questions you can ask:**\n"
            "- *I have chest pain and shortness of breath*\n"
            "- *Who treats skin infections?*\n"
            "- *What is SpPD?*\n"
            "- *I have been having high fever and diarrhea*"
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built for i3L University · Department of Bioinformatics · "
    "Dr. Arli Aditya Parikesit · For research and educational use only."
)
