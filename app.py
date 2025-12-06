import streamlit as st
import pandas as pd
import joblib
import os
import re
import javalang
import numpy as np

# ====================== CONFIGURATION ======================
st.set_page_config(page_title="Bug Predictor Ultimate", page_icon="🐞", layout="wide")

# ====================== UTILITAIRES ======================
def find_file(filename, search_paths):
    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

# Chemins de recherche
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_PATHS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "Data", "data_c"),
    os.path.join(BASE_DIR, "Data", "data_java", "jedit_final"),
    os.path.join(BASE_DIR, "Data", "data_java", "java_final"),
    r"C:/Users/pc/OneDrive/Documents/S3/genie_ogiciel/Asmae/Data/data_c",
    r"C:/Users/pc/OneDrive/Documents/S3/genie_ogiciel/Asmae/Data/data_java/jedit_final"
]

# ====================== CHARGEMENT DES MODÈLES ======================
@st.cache_resource
def load_cpp_model():
    model_path = find_file("best_bug_predictor_model.pkl", SEARCH_PATHS)
    scaler_path = find_file("scaler.pkl", SEARCH_PATHS)
    features_path = find_file("features_list.pkl", SEARCH_PATHS)
    
    if not (model_path and scaler_path and features_path):
        st.error("Fichiers du modèle C++ introuvables.")
        return None, None, None
        
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(features_path)

@st.cache_resource
def load_java_model():
    model_path = find_file("best_jedit_predictor.pkl", SEARCH_PATHS)
    scaler_path = find_file("scaler_jedit.pkl", SEARCH_PATHS)
    features_path = find_file("features_jedit.pkl", SEARCH_PATHS)
    
    if not (model_path and scaler_path and features_path):
        st.error("Fichiers du modèle Java introuvables.")
        return None, None, None

    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(features_path)

# ====================== LOGIQUE C++ ======================
def render_cpp_interface():
    st.header("🛡️ Analyse C/C++")
    model, scaler, features = load_cpp_model()
    
    if not model:
        return

    tab1, tab2 = st.tabs(["📂 Uploader un CSV", "📝 Coller du Code"])

    with tab1:
        uploaded = st.file_uploader("CSV avec les 21 métriques", type="csv", key="cpp_upload")
        if uploaded:
            df = pd.read_csv(uploaded)
            # Vérification des colonnes
            missing_cols = [c for c in features if c not in df.columns]
            if missing_cols:
                st.error(f"Colonnes manquantes : {missing_cols}")
            else:
                X = df[features].copy()
                X_scaled = scaler.transform(X)
                proba = model.predict_proba(X_scaled)[:, 1]
                pred = proba >= 0.5

                df["Probabilité_bug_%"] = (proba * 100).round(2)
                df["Prédiction"] = ["Bug" if p else "Propre" for p in pred]
                df["Risque"] = df["Probabilité_bug_%"].apply(
                    lambda x: "Élevé" if x >= 70 else ("Moyen" if x >= 40 else "Faible")
                )
                df = df.sort_values("Probabilité_bug_%", ascending=False).reset_index(drop=True)

                st.success(f"{len(df)} fichiers analysés – {int(pred.sum())} à risque")
                st.dataframe(df.head(10)[["Probabilité_bug_%", "Prédiction", "Risque", "loc", "v(g)"]], use_container_width=True)

    with tab2:
        code = st.text_area("Code Source C/C++", height=300, placeholder="#include <stdio.h>\nint main() { ... }", key="cpp_code")
        if st.button("Analyser Code C++", type="primary"):
            if not code.strip():
                st.warning("Veuillez coller du code.")
            else:
                with st.spinner("Extraction des métriques..."):
                    lines = len(code.splitlines())
                    complexity = len(re.findall(r'\b(if|for|while|switch|case)\b', code, re.I)) + 1
                    branches = len(re.findall(r'\b(if|else|for|while|switch)\b', code, re.I))

                    data = {f: 0.0 for f in features}
                    data["loc"] = lines
                    data["v(g)"] = complexity
                    data["branchCount"] = branches
                    
                    # Heuristiques simples pour les autres métriques Halstead (approximations)
                    # Pour une démo, on remplit avec des valeurs par défaut ou proportionnelles
                    data['n'] = len(re.findall(r'\w+', code)) # Length
                    data['v'] = data['n'] * 5 # Volume approx
                    data['d'] = complexity * 2 # Difficulty approx
                    data['e'] = data['v'] * data['d'] # Effort
                    data['b'] = data['v'] / 3000 # Bugs
                    
                    X = pd.DataFrame([data])[features]
                    X_scaled = scaler.transform(X)
                    proba = model.predict_proba(X_scaled)[0, 1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Probabilité de Bug", f"{proba*100:.1f}%")
                    with col2:
                        st.metric("Complexité", complexity)

                    if proba >= 0.7:
                        st.error("🚨 RISQUE ÉLEVÉ")
                    elif proba >= 0.4:
                        st.warning("⚠️ Risque Moyen")
                    else:
                        st.success("✅ Code Propre")

# ====================== LOGIQUE JAVA ======================
def extract_ck_metrics(java_code, features):
    try:
        tree = javalang.parse.parse(java_code)
    except:
        return None

    metrics = {
        'wmc': 0, 'dit': 1, 'noc': 0, 'cbo': 0, 'rfc': 0, 'lcom': 0,
        'ca': 0, 'ce': 0, 'npm': 0, 'lcom3': 0.0, 'loc': 0,
        'dam': 1.0, 'moa': 0, 'mfa': 0.0, 'cam': 1.0,
        'ic': 0, 'cbm': 0, 'amc': 0.0, 'max_cc': 1, 'avg_cc': 1.0
    }

    method_calls = set()

    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            metrics['wmc'] += 1
            metrics['npm'] += 1
            cc = 1
            for _, n in node.filter((javalang.tree.IfStatement, javalang.tree.ForStatement,
                                   javalang.tree.WhileStatement, javalang.tree.DoStatement,
                                   javalang.tree.SwitchStatement)):
                cc += 1
            metrics['max_cc'] = max(metrics['max_cc'], cc)

        if isinstance(node, javalang.tree.MethodInvocation):
            method_calls.add(node.member)

    metrics['loc'] = len([l for l in java_code.splitlines() if l.strip() and not l.strip().startswith('//')])
    metrics['rfc'] = len(method_calls) + metrics['wmc']
    metrics['cbo'] = len(method_calls)
    if metrics['wmc'] > 0:
        metrics['avg_cc'] = metrics['max_cc'] / metrics['wmc']
        metrics['amc'] = metrics['loc'] / metrics['wmc']

    # Ensure all features exist
    df = pd.DataFrame([metrics])
    for f in features:
        if f not in df.columns:
            df[f] = 0
            
    return df[features]

def render_java_interface():
    st.header("☕ Analyse Java")
    model, scaler, features = load_java_model()
    
    if not model:
        return

    java_code = st.text_area(
        "Code Source Java",
        height=300,
        placeholder="public class Point { ... }",
        key="java_code"
    )
    
    if st.button("Analyser Code Java", type="primary"):
        if not java_code.strip():
            st.warning("Veuillez coller du code.")
        else:
            with st.spinner("Analyse CK Metrics..."):
                df_metrics = extract_ck_metrics(java_code, features)
                
                if df_metrics is None:
                    st.error("Erreur de parsing Java. Vérifiez la syntaxe.")
                else:
                    # Heuristique "Classe Parfaite"
                    metrics = df_metrics.iloc[0]
                    wmc = metrics['wmc']
                    cbo = metrics['cbo']
                    rfc = metrics['rfc']
                    max_cc = metrics['max_cc']
                    loc = metrics['loc']

                    if (wmc <= 10 and cbo == 0 and rfc <= wmc + 3 and max_cc <= 2 and loc <= 80):
                        st.balloons()
                        st.success("🌟 CLASSE PARFAITE (0.0000% Risque)")
                        st.info("Respecte les meilleures pratiques : faible couplage, faible complexité.")
                    else:
                        X = scaler.transform(df_metrics)
                        prob = float(model.predict_proba(X)[0, 1])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probabilité de Bug", f"{prob*100:.2f}%")
                        with col2:
                            st.metric("WMC (Complexité)", int(wmc))
                        
                        if prob > 0.5:
                            st.error("⚠️ Potentiel Bug Détecté")
                        else:
                            st.success("✅ Code Sain")
                            
                    with st.expander("Voir les métriques détaillées"):
                        st.json(df_metrics.to_dict(orient='records')[0])

# ====================== MAIN LAYOUT ======================
st.sidebar.title("🔍 Bug Predictor")
language = st.sidebar.radio("Langage Cible", ["C/C++", "Java"])

st.sidebar.markdown("---")
st.sidebar.info("Modèles entraînés sur NASA (C++) et jEdit (Java).")

if language == "C/C++":
    render_cpp_interface()
else:
    render_java_interface()
