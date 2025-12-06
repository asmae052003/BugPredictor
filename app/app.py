import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow imports from core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import load_cpp_model, load_java_model
from utils.metrics import analyze_java_code_robust, analyze_cpp_code

# ====================== CONFIGURATION ======================
st.set_page_config(page_title="Bug Predictor Ultimate", page_icon="🐞", layout="wide")

# ====================== AFFICHAGE RÉSULTATS JAVA ======================
def display_java_results(df, npe_risk, buf_overflow, div_zero, model, scaler):
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
        prob = max(prob, 0.05)
        if prob <= 0.2:
            st.success("PARFAITEMENT PROPRE")
            st.balloons()
        else:
            st.warning("Faible risque")
        st.progress(prob)

    st.markdown(f"**Probabilité de défaut : {prob*100:.2f}%**")
    st.info(f"LOC: {int(loc)} | Complexité: {int(cc)} | Méthodes: {int(df['wmc'].iloc[0])}")
    
    with st.expander("Voir les métriques détaillées"):
        st.json(df.to_dict(orient='records')[0])


# ====================== INTERFACE C++ ======================
def render_cpp_interface():
    st.markdown("<h1 style='text-align: center; color:#FF4444;'>Bug Predictor Pro (C/C++)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Modèle NASA – 17 442 modules C/C++ – AUC = 0.949</h3>", unsafe_allow_html=True)
    
    model, scaler, features = load_cpp_model()
    
    if not model:
        st.error("Modèle C++ introuvable (best_bug_predictor_model.pkl)")
        return

    tab1, tab2 = st.tabs(["Uploader un CSV", "Coller du code C/C++"])

    # === TAB 1 : CSV ===
    with tab1:
        st.header("Analyse d’un projet complet")
        uploaded = st.file_uploader("CSV avec les 21 métriques", type=["csv"], key="cpp_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if set(features).issubset(df.columns):
                X = scaler.transform(df[features])
                proba = model.predict_proba(X)[:, 1]
                pred = (proba >= 0.5)

                df["Probabilité_bug_%"] = (proba * 100).round(2)
                df["Prédiction"] = ["Bug" if p else "Propre" for p in pred]
                df["Risque"] = df["Probabilité_bug_%"].apply(
                    lambda x: "Élevé" if x >= 70 else ("Moyen" if x >= 40 else "Faible")
                )
                df = df.sort_values("Probabilité_bug_%", ascending=False).reset_index(drop=True)

                st.success(f"{len(df)} fichiers analysés – {pred.sum()} à risque élevé")
                st.subheader("Top 10 les plus risqués")
                st.dataframe(df.head(10)[["Probabilité_bug_%", "Prédiction", "Risque", "loc", "v(g)"]], use_container_width=True)
                st.download_button("Télécharger le résultat", df.to_csv(index=False).encode(), "resultat_bugs.csv", "text/csv")
            else:
                st.error("Colonnes manquantes dans le CSV")

    # === TAB 2 : CODE DIRECT ===
    with tab2:
        st.header("Analyse directe d’un fichier C/C++")
        code = st.text_area("Colle ton code C/C++ ici", height=500, placeholder="#include <iostream>\nint main() { return 0; }", key="cpp_code")

        if st.button("Analyser ce code", type="primary", use_container_width=True, key="cpp_btn"):
            if not code.strip():
                st.warning("Colle du code d’abord !")
            else:
                with st.spinner("Analyse en cours..."):
                    # Use extracted metric logic
                    lines, complexity = analyze_cpp_code(code, features)
                    
                    # Logique de risque simplifiée pour C++
                    prob = 0.0
                    if complexity == 1 and lines <= 10:
                        prob = 0.0
                        st.success("**PARFAITEMENT PROPRE**")
                        st.balloons()
                    elif complexity <= 3 and lines <= 20:
                        prob = 0.15
                        st.success("**Faible risque**")
                    elif complexity <= 6:
                        prob = 0.45
                        st.warning("**Risque moyen**")
                    elif complexity <= 10:
                        prob = 0.78
                        st.error("**Risque élevé**")
                    elif complexity <= 15:
                        prob = 0.92
                        st.error("**RISQUE TRÈS ÉLEVÉ**")
                    else:
                        prob = 0.99
                        st.error("**RISQUE CRITIQUE**")
                        st.warning("Complexité cyclomatique ≥ 15 → Code non maintenable")

                    st.progress(prob)
                    st.markdown(f"**Probabilité de défaut estimée : {prob*100:.2f}%**")
                    st.info(f"LOC: {lines} | Complexité cyclomatique: {complexity}")

# ====================== INTERFACE JAVA ======================
def render_java_interface():
    st.markdown("<h1 style='text-align: center; color:#FF4444;'>Java Bug Predictor (jEdit)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Modèle jEdit – AUC 0.963 – Détection NPE + Buffer Overflow</h3>", unsafe_allow_html=True)
    
    model, scaler, features = load_java_model()
    
    if not model:
        st.error("Modèle Java introuvable (best_jedit_predictor.pkl)")
        return

    tab1, tab2 = st.tabs(["📂 Uploader un Fichier", "📝 Coller du Code"])

    # === TAB 1 : IMPORT ===
    with tab1:
        st.subheader("Import de fichiers")
        uploaded = st.file_uploader("Fichier CSV ou Code Source (Java)", type=["csv", "java"], key="java_upload")
        
        if uploaded:
            # === CAS 1 : CSV ===
            if uploaded.name.endswith(".csv"):
                st.info("Mode CSV : Analyse de métriques pré-calculées")
                df = pd.read_csv(uploaded)
                missing_cols = [c for c in features if c not in df.columns]
                if missing_cols:
                    st.error(f"Colonnes manquantes : {missing_cols}")
                else:
                    X = df[features].copy()
                    X_scaled = scaler.transform(X)
                    proba = model.predict_proba(X_scaled)[:, 1]
                    pred = proba >= 0.5
                    
                    df["Probabilité_bug_%"] = (proba * 100).round(2)
                    df["Risque"] = df["Probabilité_bug_%"].apply(
                        lambda x: "Élevé" if x >= 70 else ("Moyen" if x >= 40 else "Faible")
                    )
                    st.dataframe(df.head(10))
            
            # === CAS 2 : FICHIER JAVA ===
            else:
                st.info(f"Analyse du fichier source : {uploaded.name}")
                code_content = uploaded.getvalue().decode("utf-8")
                with st.spinner("Analyse du code..."):
                    df, npe, buf, div = analyze_java_code_robust(code_content, features)
                    display_java_results(df, npe, buf, div, model, scaler)

    # === TAB 2 : COLLER ===
    with tab2:
        st.header("Analyse directe")
        code = st.text_area("Colle ton code Java ici", height=500, placeholder="public class Test { ... }", key="java_code")
        
        if st.button("Analyser", type="primary", key="java_btn"):
            if not code.strip():
                st.warning("Colle du code !")
            else:
                with st.spinner("Analyse..."):
                    df, npe, buf, div = analyze_java_code_robust(code, features)
                    display_java_results(df, npe, buf, div, model, scaler)

# ====================== MAIN LAYOUT ======================
st.sidebar.title("🔍 Bug Predictor")
language = st.sidebar.radio("Langage Cible", ["C/C++", "Java"])

st.sidebar.markdown("---")
st.sidebar.info("Modèles entraînés sur NASA (C++) et jEdit (Java).")

if language == "C/C++":
    render_cpp_interface()
else:
    render_java_interface()
