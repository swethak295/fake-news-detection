import argparse
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import basic_clean

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df['combined'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(basic_clean)
    return df

def main(args):
    df = load_data(args.test_csv)
    X = df['combined']
    y = df['label']

    model = load(args.model_path)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    print("Classification Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=Path, default=Path("data/test_sample.csv"))
    parser.add_argument("--model_path", type=Path, default=Path("models/model.joblib"))
    main(parser.parse_args())
