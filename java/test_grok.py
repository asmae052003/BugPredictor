# app.py → VERSION FINALE ULTIME – Java Bug Predictor (jEdit) – AUC 0.963
import streamlit as st
import pandas as pd
import joblib
import os
import re

# ====================== INTERFACE CONFIG ======================
st.set_page_config(page_title="Java Bug Predictor", page_icon="bug", layout="wide")

# ====================== CHEMINS ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data", "data_java", "jedit_final")

MODEL_PATH = os.path.join(DATA_DIR, "best_jedit_predictor.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_jedit.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "features_jedit.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ====================== ANALYSE ROBUSTE SANS JAVALANG ======================
def analyze_code(code):
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith('//')]
    loc = len(lines)

    # Comptage des points de décision
    if_count = len(re.findall(r'\bif\b', code))
    for_count = len(re.findall(r'\bfor\b', code))
    while_count = len(re.findall(r'\bwhile\b', code))
    switch_count = len(re.findall(r'\bswitch\b', code))
    case_count = len(re.findall(r'\bcase\b', code))
    try_count = len(re.findall(r'\btry\b', code))
    catch_count = len(re.findall(r'\bcatch\b', code))

    complexity = 1 + if_count + for_count + while_count + switch_count + case_count + try_count + catch_count

    # Méthodes (approximation)
    methods = len(re.findall(r'\b(public|private|protected)?\s+\w[\w\s<>\[\]]*\]]*\s+\w+\s*\(', code))

    # Appels de méthode
    calls = len(re.findall(r'\.\w+\s*\(', code))

    # Buffer overflow
    buffer_overflow = bool(re.search(r'i\s*<=\s*\w+\.length', code, re.I)) or bool(re.search(r'i\s*<=\s*length', code, re.I))

    # Division par zéro
    division_zero = "/ 0" in code or "/0" in code

    # NullPointerException (ta règle géniale)
    nullpointer_risk = 0
    if re.search(r'\.\w+\s*\(\s*\)\s*\.', code):  # a.getB().getC()
        nullpointer_risk += 0.4
    if re.search(r'\.\w+\s*\(\s*\)\s*\.\w+\s*\(', code):
        nullpointer_risk += 0.5
    if re.search(r'\.\w+\s*\(\s*\)\s*\.\w+\s*\(\s*\)\s*\.', code):
        nullpointer_risk += 0.7
    if re.search(r'toUpperCase|toLowerCase|length|get|size|isEmpty', code, re.I):
        nullpointer_risk += 0.2
    if not re.search(r'\bif\s*\([^)]*null[^)]*\)', code):
        if "return" in code or "throw" in code:
            nullpointer_risk += 0.3

    # Métriques CK
    data = {f: 0.0 for f in FEATURES}
    data["loc"] = loc
    data["wmc"] = max(1, methods)
    data["v(g)"] = complexity
    data["max_cc"] = complexity
    data["avg_cc"] = complexity / max(1, methods)
    data["cbo"] = calls
    data["rfc"] = calls + methods
    data["npm"] = methods
    data["amc"] = loc / max(1, methods) if methods > 0 else loc

    return pd.DataFrame([data])[FEATURES], nullpointer_risk, buffer_overflow, division_zero

# ====================== INTERFACE ======================
st.markdown("<h1 style='text-align: center; color:#FF4444;'>Java Bug Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Modèle jEdit – AUC 0.963 – Détection NPE + Buffer Overflow</h3>", unsafe_allow_html=True)

code = st.text_area("Colle ton code Java ici", height=500, placeholder="public class Test { ... }")

if st.button("Analyser", type="primary"):
    if not code.strip():
        st.warning("Colle du code !")
    else:
        with st.spinner("Analyse..."):
            df, npe_risk, buf_overflow, div_zero = analyze_code(code)
            X = scaler.transform(df)
            prob = float(model.predict_proba(X)[0, 1])

            loc = df['loc'].iloc[0]
            cc = df['max_cc'].iloc[0]

            # === PRIORITÉ : RÈGLES EXPERTES AVANT LE MODÈLE ===
            if div_zero:
                prob = 0.99
                st.error("RISQUE CRITIQUE – Division par zéro détectée")
                st.progress(prob)
            elif buf_overflow:
                prob = 0.99
                st.error("RISQUE CRITIQUE – Buffer overflow détecté")
                st.progress(prob)
            elif npe_risk >= 0.6:
                prob = 0.94
                st.error("RISQUE CRITIQUE – NullPointerException très probable")
                st.progress(prob)
            elif npe_risk >= 0.4:
                prob = 0.85
                st.error("RISQUE ÉLEVÉ – Risque de NPE")
                st.progress(prob)
            elif cc >= 15:
                prob = 0.95
                st.error("RISQUE TRÈS ÉLEVÉ – Complexité cyclomatique excessive")
                st.progress(prob)
            elif cc >= 10:
                prob = 0.88
                st.error("RISQUE ÉLEVÉ")
                st.progress(prob)
            elif cc >= 6:
                prob = 0.72
                st.error("RISQUE ÉLEVÉ")
                st.progress(prob)
            elif cc >= 3:
                prob = 0.45
                st.warning("Risque moyen")
                st.progress(prob)
            else:
                prob = max(prob, 0.05)  # minimum 5% pour éviter 0.000% sur tout
                if prob <= 0.2:
                    st.success("PARFAITEMENT PROPRE")
                    st.balloons()
                else:
                    st.warning("Faible risque")
                st.progress(prob)

            st.markdown(f"**Probabilité de défaut : {prob*100:.2f}%**")
            st.info(f"LOC: {int(loc)} | Complexité: {int(cc)} | Méthodes: {int(df['wmc'].iloc[0])}")

st.markdown("---")
st.success("Projet 2025 – Modèle jEdit + règles expertes – Détection NPE, Buffer Overflow, Division/0")