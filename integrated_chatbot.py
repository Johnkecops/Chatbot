#!/usr/bin/env python3
"""
Integrated Medical Specialist Chatbot
Combines ANN (symptom-based) and SVM (free-text Q&A) chatbots
Author: Dr. Arli Aditya Parikesit
"""

import traceback

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import triage
from triage import SYMPTOM_DISPLAY, HISTORY_DISPLAY

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

SYMPTOMS_COLS = list(SYMPTOM_DISPLAY.keys())
HISTORY_COLS  = list(HISTORY_DISPLAY.keys())


# ── Model loaders (cached) ─────────────────────────────────────────────────────
# Same load/predict logic as ANN_chatbot_files 1/chatbot_model.py (chatModel, used by the notebooks).
@st.cache_resource(show_spinner="Loading ANN model…")
def load_ann():
    import tensorflow as tf
    model   = tf.keras.models.load_model(str(ANN_DIR / "chatbot_model1.h5"))
    df      = pd.read_csv(str(ANN_DIR / "data_tutorial.csv"))
    features    = list(df.iloc[:, :26].columns)
    specialists = list(df.iloc[:, -9:].columns)
    assert features == SYMPTOMS_COLS + HISTORY_COLS, (
        "data_tutorial.csv columns do not match the app's symptom/history lists"
    )
    return model, features, specialists


@st.cache_resource(show_spinner="Training SVM model…")
def load_svm():
    return triage.build_svm(SVM_CSV)


# ── ANN prediction ─────────────────────────────────────────────────────────────
def predict_ann(model, features, specialists, values: dict) -> tuple[str, float]:
    row = [values[f] for f in features]
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
    except Exception:
        traceback.print_exc()  # details to the server console, not the browser
        st.error(
            "Could not load ANN model. Make sure `chatbot_model1.h5` and "
            "`data_tutorial.csv` are inside `ANN_chatbot_files 1/`."
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
            c.checkbox(HISTORY_DISPLAY[col], key=f"hist_{col}")
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
            for msg in triage.ann_red_flags(sym_values):
                st.error(f"EMERGENCY WARNING: {msg}")
            spec_name = SPECIALIST_NAMES.get(spec_code, spec_code)
            all_display = {**SYMPTOM_DISPLAY, **HISTORY_DISPLAY}
            checked_labels = [
                all_display[k] for k, v in sym_values.items() if v == 1
            ]

            st.success("Prediction complete")
            result_col, detail_col = st.columns([1, 2])

            with result_col:
                st.metric("Recommended Specialist", spec_name)
                st.metric("Softmax score", f"{confidence:.1%}")
                st.caption(
                    "Not a probability: the model was trained with hinge loss on 111 "
                    "cases. THT, SpKK and SpB had 1 to 3 training cases each, so those "
                    "outputs are especially unreliable."
                )

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
        svm_pipeline, svm_cv_mean, svm_cv_sd, svm_folds = load_svm()
    except Exception:
        traceback.print_exc()
        st.error("Could not load SVM model.")
        st.stop()

    st.warning(
        f"**Small training set.** {svm_folds}-fold stratified CV accuracy: "
        f"{svm_cv_mean:.0%} ± {svm_cv_sd:.0%}, measured on rephrasings of the same "
        "questions, so it overstates performance on real users. "
        "Responses are pattern-matched, not clinically validated."
    )

    # ── Chat history ───────────────────────────────────────────────────────────
    if "svm_history" not in st.session_state:
        st.session_state.svm_history = []

    # ── Input ──────────────────────────────────────────────────────────────────
    with st.form("svm_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            max_chars=500,
            placeholder="e.g. I have chest pain and shortness of breath, which doctor should I see?",
        )
        submitted = st.form_submit_button("💬 Ask", type="primary", use_container_width=True)

    if submitted and user_input.strip():
        text = user_input.strip()
        with st.spinner("Thinking…"):
            answer = triage.svm_answer(svm_pipeline, text)
        st.session_state.svm_history.append({"role": "user", "text": text})
        st.session_state.svm_history.append(
            {"role": "assistant", "text": answer, "flags": triage.text_red_flags(text)}
        )

    # ── Render chat history ────────────────────────────────────────────────────
    for msg in reversed(st.session_state.svm_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["text"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                for flag in msg["flags"]:
                    st.error(f"EMERGENCY WARNING: {flag}")
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
