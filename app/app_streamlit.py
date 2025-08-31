
import streamlit as st
import sys
from pathlib import Path

# Add the parent directory to Python path to find the scripts module
sys.path.append(str(Path(__file__).parent.parent))

from joblib import load
from scripts.utils import basic_clean

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection")

model_path = Path("models/model.joblib")
if not model_path.exists():
    st.warning("Model file not found. Please train the model first (see README).")
else:
    try:
        model = load(model_path)
        st.success("Model loaded.")

        st.subheader("Try it out")
        title = st.text_input("News Title", "")
        text = st.text_area("News Text", "", height=200)
        if st.button("Analyze"):
            combined = basic_clean((title or '') + ' ' + (text or ''))
            try:
                pred = model.predict([combined])[0]
                st.write(f"**Prediction:** {pred}")
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([combined])[0]
                    classes = getattr(model, 'classes_', None)
                    if classes is not None:
                        st.write("**Confidence:**")
                        for cls, p in zip(classes, proba):
                            st.write(f"- {cls}: {p:.4f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

st.markdown("""
---
**How it works:** We use a TF-IDF + Logistic Regression pipeline to analyze text patterns associated with misinformation (e.g., exaggerated claims, lack of sources) versus credible reporting cues.

**How to run:**
1. Make sure you have installed the requirements:  
   `pip install -r requirements.txt`
2. Train the model if you haven't already (see `scripts/train.py`).
3. Start the app with:  
   `streamlit run app/app_streamlit.py`
""")
