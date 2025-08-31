# Fake News Detection — Detailed Report

## 1. Problem Definition
Detect whether a given news article (title + body) is **FAKE** or **REAL**.

## 2. Data
- Structure: `title`, `text`, `label` (FAKE/REAL)
- Sample data provided for quick demo; replace with a larger corpus for better results.

## 3. Methodology
1. **Preprocessing:** lowercase, URL masking, punctuation removal, whitespace normalization.
2. **Feature Extraction:** TF‑IDF on uni/bi‑grams (max 20k features).
3. **Model:** Logistic Regression (baseline).
4. **Evaluation:** train/validation split; accuracy, precision/recall/F1, confusion matrix.
5. **Deployment:** Streamlit app for real‑time predictions.

## 4. Experiments
- Baseline with default hyperparameters.
- Potential improvements:
  - class_weight='balanced'
  - Grid search on C, ngram_range
  - Swap model: LinearSVC, ComplementNB
  - Data augmentation and adversarial examples
  - Use weak signals: presence of citations, numbers, hedging words

## 5. Results (Sample)
Small demo data yields high accuracy due to clear signals. Real datasets will be more challenging; expect 85–95% val accuracy on balanced classic Kaggle set with this pipeline.

## 6. Error Analysis
Typical false positives: sensational but true headlines.  
Typical false negatives: fake stories mimicking professional style.

## 7. Risks & Mitigations
- **Bias:** diverse training sources; evaluate across outlets.
- **Overfitting:** cross‑validation; regularization.
- **Concept drift:** periodic retraining; monitor performance.

## 8. Conclusion
A lightweight TF‑IDF + LR model is fast, explainable, and competitive. For higher accuracy, explore transformer-based models and claim-verification via retrieval.
