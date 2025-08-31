#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p data

# Install dependencies
pip install -r requirements.txt

# Train the model if it doesn't exist
if [ ! -f "models/model.joblib" ]; then
    echo "Training model..."
    python3 scripts/train.py
fi

echo "Setup complete! Run 'streamlit run app/app_streamlit.py' to start the app."
