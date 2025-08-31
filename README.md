# 📰 Fake News Detection — Cyber Hackathon 2025

A complete, hackathon-ready project to classify news as **FAKE** or **REAL** using NLP + Machine Learning.

## 🔧 Tech Stack
- Python, scikit-learn, pandas, numpy
- TF‑IDF + Logistic Regression pipeline
- Streamlit web app for live demo
- joblib for model persistence

## 📁 Project Structure
```
fake-news-detection/
├── app/
│   └── app_streamlit.py         # Streamlit app
├── data/
│   ├── train_sample.csv         # Sample training data
│   └── test_sample.csv          # Sample test data
├── models/
│   └── model.joblib             # Saved model (created after training)
├── notebooks/                   # (optional) your experiments
├── reports/
│   ├── report.md                # Detailed report (edit as needed)
│   └── presentation.pptx        # Slides
├── scripts/
│   ├── train.py                 # Train the model
│   ├── evaluate.py              # Evaluate on test set
│   ├── predict.py               # Single-text inference
│   └── utils.py                 # Basic text cleaning
└── requirements.txt
```

## 🚀 Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python scripts/train.py --train_csv data/train_sample.csv --out_dir models
   ```

3. **Evaluate on the test set**
   ```bash
   python scripts/evaluate.py --test_csv data/test_sample.csv --model_path models/model.joblib
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app/app_streamlit.py
   ```

## 📦 Using a Larger Dataset
Replace `data/train_sample.csv` and `data/test_sample.csv` with a real dataset (e.g., Kaggle Fake News). Ensure your CSV has columns:
- `title` (string)
- `text` (string)
- `label` (string: `FAKE` or `REAL`)

Then retrain:
```bash
python scripts/train.py --train_csv data/your_train.csv --out_dir models
```

## 🧠 Model Notes
- Baseline pipeline: **TF‑IDF (1–2 grams, max_features=20k) + Logistic Regression**
- Swap in alternatives easily (e.g., LinearSVC, ComplementNB) inside `train.py`.
- Add class weights if dataset is imbalanced.

## 🧪 Metrics & Reporting
`train.py` and `evaluate.py` print **Accuracy**, **Confusion Matrix**, and a **Classification Report** (precision/recall/F1).

## 🔒 Ethics & Limitations
- No model can guarantee truth; use as **decision support** only.
- Beware dataset bias and distribution shift.
- Provide citations for claims when integrating with live sources.

## 👨‍🏫 Demo Tips
- Show predictions on **both** obvious fake claims and credible articles.
- Keep some test samples ready for offline demos.
- Explain TF‑IDF and logistic regression in 20 seconds: *count n-grams → weight by rarity → learn weights that separate FAKE vs REAL*.

Good luck! ✨
