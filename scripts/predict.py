import argparse
from pathlib import Path
from joblib import load
from utils import basic_clean

def main(args):
    model = load(args.model_path)
    text = basic_clean((args.title or '') + ' ' + (args.text or ''))
    pred = model.predict([text])[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        # Pipeline exposes predict_proba if the final estimator supports it
        try:
            proba = model.predict_proba([text])[0]
        except Exception:
            proba = None
    print(f"Prediction: {pred}")
    if proba is not None:
        # Assumes binary classes in alphabetical order
        classes = getattr(model.classes_, None)
        if classes is not None:
            print('Probabilities:')
            for cls, p in zip(classes, proba):
                print(f"  {cls}: {p:.4f}")
    else:
        print("Probabilities not available for this model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, default=Path('models/model.joblib'))
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--text', type=str, default='')
    main(parser.parse_args())
