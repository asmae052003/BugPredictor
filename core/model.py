import os
import joblib
import streamlit as st

# Search paths moved to a module level constant or function
def get_search_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level from core/
    return [
        base_dir,
        os.path.join(base_dir, "data", "data_c"),
        os.path.join(base_dir, "data", "data_java", "jedit_final"),
        os.path.join(base_dir, "data", "data_java", "java_final"),
        # Absolute fallback paths (can be adjusted if needed)
        r"C:/Users/pc/OneDrive/Documents/S3/genie_ogiciel/Asmae/data/data_c",
        r"C:/Users/pc/OneDrive/Documents/S3/genie_ogiciel/Asmae/data/data_java/jedit_final"
    ]

def find_file(filename):
    search_paths = get_search_paths()
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

@st.cache_resource
def load_cpp_model():
    model_path = find_file("best_bug_predictor_model.pkl")
    scaler_path = find_file("scaler.pkl")
    features_path = find_file("features_list.pkl")
    
    if not (model_path and scaler_path and features_path):
        return None, None, None
        
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(features_path)

@st.cache_resource
def load_java_model():
    model_path = find_file("best_jedit_predictor.pkl")
    scaler_path = find_file("scaler_jedit.pkl")
    features_path = find_file("features_jedit.pkl")
    
    if not (model_path and scaler_path and features_path):
        return None, None, None

    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(features_path)
