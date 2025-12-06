# =======================================================
#  ÉTAPE 4 : ENTRAÎNEMENT & COMPARAISON DE 6 MODÈLES
#  → Marche SANS lightgbm ni catboost (zéro installation)
#  → Garde automatiquement le meilleur
# =======================================================

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Chargement des données (cherche dans le dossier parent si besoin)
import os
if not os.path.exists("X_train.csv"):
    os.chdir("../Data/data_c")  # remonte automatiquement au bon dossier

X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test  = pd.read_csv("y_test.csv").squeeze()

print(f"Données chargées → Train: {X_train.shape} | Test: {X_test.shape}\n")

# 6 modèles qui marchent à coup sûr
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest"      : RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost"            : XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss'),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=300, random_state=42),
    "AdaBoost"           : AdaBoostClassifier(n_estimators=300, random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=5)
}

results = []
print("Entraînement en cours...\n" + "-"*60)

best_auc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"{name:20} → entraînement...", end="")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f" AUC = {auc:.4f} | F1 = {f1:.4f} | Recall = {rec:.4f}")
    
    results.append({"Model": name, "AUC": auc, "F1": f1, "Recall": rec, "Accuracy": acc})
    
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print("\n" + "="*80)
print("RÉSULTATS FINAUX")
print("="*80)
df_results = pd.DataFrame(results).sort_values("AUC", ascending=False).round(4)
print(df_results.to_string(index=False))

print("\n" + "="*80)
print(f"MEILLEUR MODÈLE → {best_name} (AUC = {best_auc:.4f})")
print("="*80)

# Sauvegarde
joblib.dump(best_model, "best_bug_predictor_model.pkl")


print("\nModèle sauvegardé → best_bug_predictor_model.pkl")

print("Tu es prêt pour l'API et l'interface web !")