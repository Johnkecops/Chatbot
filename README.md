# Medical Specialist Chatbot

Chatbot development for health service in a hospital. Uses multiple computational approaches — Support Vector Machine (SVM) for free-text Q&A and Artificial Neural Network (ANN) for symptom-based specialist recommendation. An integrated Streamlit web application combines both models in a single interface.

---

## Repository Structure

```
Chatbot-main/
├── integrated_chatbot.py         # Streamlit app: ANN + SVM interface
├── triage.py                     # Streamlit-free logic: red-flag rules, SVM build/answer/reject
├── svm_data.csv                  # 188 Q&A rows (47 responses x 4 phrasings)
├── scripts/eval_ann.py           # Per-class ANN report (resubstitution + stratified CV)
├── tests/test_pipeline.py        # pytest smoke tests
├── results/tables/               # Output of eval_ann.py
├── ANN_chatbot_files 1/          # ANN chatbot (Python / TensorFlow)
│   ├── ANN_Chatbot_Note.ipynb    # Training, testing, saving model
│   ├── ANN_Chatbot_Model.ipynb   # Load model and run predictions
│   ├── chatbot_model.py          # chatModel class used by the notebooks
│   ├── chatbot_model1.h5         # Trained Keras model (tracked: the app needs it to run)
│   ├── data_tutorial.csv         # 111 cases, 26 binary features, 9 specialist labels
│   ├── instruction.txt
│   └── my_conda_env.yml
├── archive/R_SVM_ICORIS2022/     # Original R SVM scripts, kept for provenance, NOT runnable as is
└── requirements_integrated.txt
```

---

## Integrated Streamlit Application

`integrated_chatbot.py` merges both chatbots. Users select the model in the sidebar.

### ANN mode: symptom checker
- Input: 26 binary checkboxes (18 symptoms, 8 pre-existing conditions).
- Output: recommended specialist plus the softmax score. The score is not a probability: the model uses hinge loss and was trained on 111 cases.
- Backed by `chatbot_model1.h5`, an MLP (240-180-120-60-9) trained in `ANN_Chatbot_Note.ipynb` on all 111 cases with no held-out set.
- Specialists: SpJP, SpPD, SpS, SpB, THT, SpKK, SpM, Paru, SpOT.
- Known limits: THT has 1 training case, SpB and SpKK have 3 each. `scripts/eval_ann.py` shows the shipped model scores 0 recall on THT and SpB even on its own training data.

### SVM mode: free-text Q&A
- Input: plain-language question. Output: the matched answer, or a "cannot match" message when the top one-vs-rest margin is below `REJECT_MARGIN` in `triage.py`.
- scikit-learn `LinearSVC` on TF-IDF (1-2 grams). Accuracy shown in the app is stratified 4-fold CV on rephrasings of the same questions, so it overstates performance on real users.
- This is a Python re-implementation of the idea in the original R pipeline (TF-IDF-style term matrix plus linear SVM). It is not a numerical reproduction: the R version used raw term counts, `cost=100` and no stop-word removal, and has no published evaluation here.

### Emergency red flags (both modes)
Rules in `triage.py` show an emergency banner for chest discomfort with breathlessness or fainting, and for sudden speech, vision or walking trouble. These rules and the SVM Q&A rows were approved by the project lead on 2026-10-16. They have not been reviewed by a licensed physician; do that before any clinical use.

### Running
```bash
cd Chatbot-main
pip install -r requirements_integrated.txt      # tested: tensorflow 2.15.0, scikit-learn 1.8.0, streamlit 1.57.0
streamlit run integrated_chatbot.py
python -m pytest tests                          # smoke tests
python scripts/eval_ann.py                      # writes results/tables/ann_*.csv
```

---

## Original ANN Chatbot Setup

```bash
# Create and activate environment
conda env create -f "ANN_chatbot_files 1/my_conda_env.yml"
conda activate tf-keras-chatbot

# Training and saving model
jupyter notebook "ANN_chatbot_files 1/ANN_Chatbot_Note.ipynb"

# Running predictions
jupyter notebook "ANN_chatbot_files 1/ANN_Chatbot_Model.ipynb"
```

---

## Original SVM Chatbot (R), archived

`archive/R_SVM_ICORIS2022/` holds the scripts behind the ICORIS 2022 paper. They are not runnable as shipped (missing input file, wrong package name `snowballC`, `tm` never loaded, `Database.r` does not produce the `question`/`Response` columns that `Step 1.r` needs). Fix those before reuse.

---

## Specialist Code Reference

| Code | Specialist |
|------|-----------|
| SpJP | Cardiologist (Jantung dan Pembuluh Darah) |
| SpPD | Internist (Penyakit Dalam) |
| SpS  | Neurologist (Saraf) |
| SpB  | General Surgeon (Bedah) |
| THT  | ENT Specialist (Telinga Hidung Tenggorokan) |
| SpKK | Dermatologist (Kulit dan Kelamin) |
| SpM  | Ophthalmologist (Mata) |
| Paru | Pulmonologist (Paru) |
| SpOT | Orthopedic Surgeon (Ortopedi dan Traumatologi) |

---

## Publications

Matthew, R., Agustriawan, D., Bani, M. D., Sadrawi, M., Ratnasari, N. R. P., Firmansyah, M., & Parikesit, A. A. (2022). The Development of A Medical Chatbot Using The SVM Algorithm. *2022 4th International Conference on Cybernetics and Intelligent System (ICORIS)*, 1–6. https://doi.org/10.1109/ICORIS56080.2022.10031400

Hasuki, W., Agustriawan, D., Parikesit, A. A., Sadrawi, M., Firmansyah, M., Whisnu, A., Natasya, J., Mathew, R., Napitupulu, F. I., & Ratnasari, N. R. P. (2023). Development of Artificial Neural Network Model for Medical Specialty Recommendation. *Pertanika Journal of Science and Technology*, 31(6), 2723–2733. https://doi.org/10.47836/pjst.31.6.05

Shiloputra, A. F., Darmawan, J. T., Zahra, S. L., Sugiharto, S., Pricillia, V., & Parikesit, A. A. (2023). APABOT: A Chatbot for ASD Treatment Implemented by ParlAI | RINarxiv. https://rinarxiv.lipi.go.id/lipi/preprint/view/728

---

## Disclaimer

This tool is for **research and educational purposes only**. It does not constitute medical advice and must not be used as a substitute for professional clinical judgment.

---

AI Assistance Disclaimer: This codebase was developed with the assistance of Claude Code. While the AI provided code generation, debugging, and structural support, the human developer maintains full responsibility for reviewing, testing, and maintaining all content and functionality
