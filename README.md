# Medical Specialist Chatbot

Chatbot development for health service in a hospital. Uses multiple computational approaches — Support Vector Machine (SVM) for free-text Q&A and Artificial Neural Network (ANN) for symptom-based specialist recommendation. An integrated Streamlit web application combines both models in a single interface.

---

## Repository Structure

```
Chatbot-main/
├── ANN_chatbot_files 1/          # ANN chatbot (Python / TensorFlow)
│   ├── ANN_Chatbot_Model.ipynb   # Notebook: load model and run predictions
│   ├── ANN_Chatbot_Note.ipynb    # Notebook: training, testing, saving model
│   ├── chatbot_model.py          # chatModel class — loads .h5 and predicts specialist
│   ├── chatbot_model1.h5         # Pre-trained Keras model weights
│   ├── data_tutorial.csv         # Training dataset (26 symptom features, 9 specialist labels)
│   ├── instruction.txt           # Setup instructions for the ANN environment
│   └── my_conda_env.yml          # Conda environment specification (tf-keras-chatbot)
│
├── SVM Chatbot File/             # SVM chatbot (original R implementation)
│   ├── Database.r                # Build symptom/specialist reference data
│   ├── Dataset cleaning.r        # Data cleaning pipeline
│   ├── Divide and training data.r# Train/test split (80/20)
│   ├── Step 1.r                  # Text preprocessing and DTM construction
│   ├── Step 2.r                  # Build training matrix with responses
│   ├── Step 3.r                  # Train SVM (linear kernel, cost=100)
│   └── Step 4.r                  # Prediction function
│
├── integrated_chatbot.py         # Streamlit app — unified ANN + SVM interface
├── svm_data.csv                  # Sample Q&A training data for the Python SVM model
├── requirements_integrated.txt   # Python dependencies for the integrated app
├── LICENSE.txt                   # License information
└── README.md                     # This file
```

---

## Integrated Streamlit Application (New)

`integrated_chatbot.py` merges both chatbots into a single web app. Users select the model via a sidebar radio button.

### ANN Mode — Symptom Checker
- Input: 26 binary checkboxes (18 acute symptoms + 8 pre-existing conditions)
- Output: recommended specialist + model confidence score
- Backed by `chatbot_model1.h5` (TensorFlow/Keras, pre-trained)
- Predicts one of 9 specialist categories: SpJP, SpPD, SpS, SpB, THT, SpKK, SpM, Paru, SpOT

### SVM Mode — Free-Text Q&A
- Input: plain-language question typed by the user
- Output: matched answer from trained Q&A pairs
- Re-implemented in Python (scikit-learn `LinearSVC` + TF-IDF) to remove the R dependency
- Training data: `svm_data.csv` (47 medical Q&A pairs, extensible)

### Running the App

```bash
# Base Anaconda environment already has all dependencies
# tensorflow-macos 2.15.0 | streamlit 1.55.0 | scikit-learn 1.3.0

cd Chatbot-main
streamlit run integrated_chatbot.py
```

Or, using the original ANN conda environment:

```bash
conda activate tf-keras-chatbot
pip install streamlit scikit-learn
streamlit run integrated_chatbot.py
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

## Original SVM Chatbot (R)

The original SVM implementation is in R and requires the following packages:
`tm`, `SnowballC`, `NLP`, `e1071`, `openxlsx`

Run scripts in order: `Step 1.r` → `Step 2.r` → `Step 3.r` → `Step 4.r`

The Python re-implementation in `integrated_chatbot.py` reproduces the same logic (TF-IDF vectorisation + linear SVM) without requiring an R installation.

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
