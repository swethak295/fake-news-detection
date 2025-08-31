#!/usr/bin/env python3
"""
Test script to verify deployment setup
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit: {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from joblib import load
        print("✅ Joblib: OK")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_paths():
    """Test if all required paths exist"""
    print("\n🔍 Testing file paths...")
    
    paths_to_check = [
        "app/app_streamlit.py",
        "scripts/utils.py",
        "models/model.joblib",
        "data/train_sample.csv",
        "requirements.txt",
        ".streamlit/config.toml",
        "Dockerfile",
        "docker-compose.yml",
        "Procfile",
        "runtime.txt"
    ]
    
    all_exist = True
    for path in paths_to_check:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - Missing!")
            all_exist = False
    
    return all_exist

def test_model():
    """Test if the model can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        from joblib import load
        model_path = Path("models/model.joblib")
        
        if not model_path.exists():
            print("❌ Model file not found")
            return False
        
        model = load(model_path)
        print("✅ Model loaded successfully")
        
        # Test prediction
        from scripts.utils import basic_clean
        test_text = "This is a test news article"
        cleaned = basic_clean(test_text)
        prediction = model.predict([cleaned])[0]
        print(f"✅ Test prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Fake News Detection - Deployment Test\n")
    
    tests = [
        ("Imports", test_imports),
        ("File Paths", test_paths),
        ("Model Loading", test_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your app is ready for deployment.")
        print("\nNext steps:")
        print("1. Push to GitHub: git add . && git commit -m 'Ready for deployment'")
        print("2. Deploy to Streamlit Cloud: https://share.streamlit.io")
        print("3. Or use Docker: docker-compose up --build")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
