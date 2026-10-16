#!/usr/bin/env python3
"""
Module: eval_ann
Purpose: Honest performance report for the ANN specialist model (does not touch the shipped .h5)
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - none; reads ANN_chatbot_files 1/data_tutorial.csv and chatbot_model1.h5
Outputs (results/tables/):
    - ann_class_report.csv: per-class n, precision, recall, resubstitution and CV
    - ann_confusion_resub.csv: shipped model on its own training data (optimistic)
References:
    - Architecture and training settings copied from ANN_Chatbot_Note.ipynb
      (MLP 240-180-120-60-9, categorical_hinge, adam, 40 epochs, batch 32)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "ANN_chatbot_files 1"
OUT = ROOT / "results" / "tables"
K = 3          # CV folds; classes with n < K cannot be stratified and are reported as such
SEED = 42


def baseline_model():
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Dense(240, input_dim=26, activation="relu", kernel_initializer="he_uniform"),
        L.Dense(180, activation="relu", kernel_initializer="he_uniform"),
        L.Dense(120, activation="relu", kernel_initializer="he_uniform"),
        L.Dense(60, activation="relu", kernel_initializer="he_uniform"),
        L.Dense(9, activation="softmax"),
    ])
    m.compile(loss="categorical_hinge", optimizer="adam", metrics=["accuracy"])
    return m


def main():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    df = pd.read_csv(ANN_DIR / "data_tutorial.csv")
    X = df.iloc[:, :26].to_numpy(dtype="float32")
    Y = df.iloc[:, -9:].to_numpy(dtype="float32")
    labels = list(df.iloc[:, -9:].columns)
    y = Y.argmax(1)
    n = np.bincount(y, minlength=9)

    # (a) shipped model on its own training data: resubstitution, optimistic by design
    shipped = tf.keras.models.load_model(str(ANN_DIR / "chatbot_model1.h5"))
    p_resub = shipped.predict(X, verbose=0).argmax(1)
    rep_resub = classification_report(y, p_resub, labels=range(9), target_names=labels,
                                      output_dict=True, zero_division=0)

    # (b) stratified CV of a fresh model with the same architecture, rows of evaluable classes only
    keep = n[y] >= K
    Xk, yk, Yk = X[keep], y[keep], Y[keep]
    p_cv = np.full(len(yk), -1)
    for tr, te in StratifiedKFold(K, shuffle=True, random_state=SEED).split(Xk, yk):
        m = baseline_model()
        m.fit(Xk[tr], Yk[tr], epochs=40, batch_size=32, shuffle=True, verbose=0)
        p_cv[te] = m.predict(Xk[te], verbose=0).argmax(1)
    cv_labels = [i for i in range(9) if n[i] >= K]
    rep_cv = classification_report(yk, p_cv, labels=cv_labels,
                                   target_names=[labels[i] for i in cv_labels],
                                   output_dict=True, zero_division=0)

    rows = []
    for i, lab in enumerate(labels):
        cv = rep_cv.get(lab)
        rows.append({
            "specialist": lab, "n": int(n[i]),
            "resub_precision": rep_resub[lab]["precision"], "resub_recall": rep_resub[lab]["recall"],
            "cv_precision": cv["precision"] if cv else "not evaluable (n<%d)" % K,
            "cv_recall": cv["recall"] if cv else "not evaluable (n<%d)" % K,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "ann_class_report.csv", index=False)
    pd.DataFrame(confusion_matrix(y, p_resub, labels=range(9)), index=labels, columns=labels
                 ).to_csv(OUT / "ann_confusion_resub.csv")

    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"\nresubstitution accuracy (optimistic): {np.mean(p_resub == y):.3f}")
    print(f"{K}-fold stratified CV accuracy, classes with n>={K} ({keep.sum()} rows): "
          f"{np.mean(p_cv == yk):.3f}")
    print("Rows with n<%d excluded from CV: %s" % (K, [labels[i] for i in range(9) if n[i] < K]))


if __name__ == "__main__":
    main()
