import argparse
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import basic_clean

def load_data(train_csv: Path):
    df = pd.read_csv(train_csv)
    # Expect columns: 'title', 'text', 'label'
    df['combined'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(basic_clean)
    return df

def build_pipeline(max_features=20000):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])
    return pipeline

def main(args):
    data = load_data(args.train_csv)
    X = data['combined']
    y = data['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_pipeline(max_features=args.max_features)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))
    print("Classification Report:")
    print(classification_report(y_val, preds))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.out_dir)/"model.joblib"
    dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, default=Path("data/train_sample.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("models"))
    parser.add_argument("--max_features", type=int, default=20000)
    main(parser.parse_args())
