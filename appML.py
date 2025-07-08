 import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

st.set_page_config(page_title="🧠 Application IA complète", layout="wide")
st.title("📊 Application de Prédiction via Machine Learning by Asmaa Faris")

# Fichiers
st.header("1️⃣ Base de données brute")
brut_file = st.file_uploader("📥 Importez la base brute (.xlsx)", type=["xlsx"], key="brut")
df_brut = None
categorical_info = {}

if brut_file:
    df_brut = pd.read_excel(brut_file)
    st.subheader("Aperçu")
    st.dataframe(df_brut.head())
    st.subheader("📈 Statistiques descriptives")
    st.write(df_brut.describe(include='all'))

    # Capter les colonnes catégorielles et leurs modalités
    for col in df_brut.select_dtypes(include='object').columns:
        categorical_info[col] = df_brut[col].unique().tolist()

# Chargement de la base nettoyée
st.header("2️⃣ Base nettoyée pour apprentissage")
clean_file = st.file_uploader("📥 Importez la base nettoyée (.xlsx)", type=["xlsx"], key="clean")

if clean_file:
    df = pd.read_excel(clean_file)
    st.dataframe(df.head())

    all_cols = df.columns.tolist()
    target = st.selectbox("🎯 Variable cible :", all_cols)
    features = st.multiselect("📌 Variables explicatives :", [c for c in all_cols if c != target])

    # Algorithmes et hyperparamètres
    st.sidebar.header("⚙️ Choix de l'algorithme et hyperparamètres")
    algo_choice = st.sidebar.selectbox("🧠 Algorithme", [
        "KNeighborsClassifier", "SGDClassifier", "NuSVC", "RidgeClassifierCV", "GaussianNB"
    ])

    if algo_choice == "KNeighborsClassifier":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif algo_choice == "SGDClassifier":
        alpha = st.sidebar.number_input("alpha", 1e-6, 1e-1, value=0.0001, format="%.5f")
        max_iter = st.sidebar.slider("max_iter", 100, 5000, 1000)
        model = SGDClassifier(alpha=alpha, max_iter=max_iter)
    elif algo_choice == "NuSVC":
        nu = st.sidebar.slider("nu", 0.01, 1.0, 0.5)
        kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
        model = NuSVC(nu=nu, kernel=kernel, probability=True)
    elif algo_choice == "RidgeClassifierCV":
        model = RidgeClassifierCV()
    elif algo_choice == "GaussianNB":
        model = GaussianNB()

    # Apprentissage
    if target and features:
        X = df[features].copy()
        y = df[target].copy()

        # Encodage cible
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Encodage auto des colonnes catégorielles
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)

        # Prédictions et métriques
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Train": [
                accuracy_score(y_train, y_train_pred),
                precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                recall_score(y_train, y_train_pred, average='weighted'),
                f1_score(y_train, y_train_pred, average='weighted')
            ],
            "Test": [
                accuracy_score(y_test, y_test_pred),
                precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                recall_score(y_test, y_test_pred, average='weighted'),
                f1_score(y_test, y_test_pred, average='weighted')
            ]
        })

        st.header("3️⃣ Résultats du modèle")
        st.dataframe(metrics_df.style.format({"Train": "{:.2%}", "Test": "{:.2%}"}))

  


        # Prédiction manuelle
        st.header("4️⃣ Prédiction manuelle à partir de la base brute")
        manual_input = []

        if df_brut is not None and features:
            st.subheader("🖊️ Saisissez les valeurs d’entrée :")
            for col in features:
                if col in categorical_info:
                    val = st.selectbox(f"{col} (catégorielle)", categorical_info[col])
                    label_encoder = LabelEncoder()
                    label_encoder.fit(categorical_info[col])
                    encoded = label_encoder.transform([val])[0]
                    manual_input.append(encoded)
                else:
                    val = st.number_input(f"{col} (numérique)", value=float(df_brut[col].mean()))
                    manual_input.append(val)

            # Standardisation + prédiction
            input_scaled = scaler.transform([manual_input])
            if st.button("🔮 Prédire"):
                pred = model.predict(input_scaled)[0]
                st.success(f"🎯 Résultat de la prédiction : **{pred}**")
                   
  
                # Affichage des probabilités si possible
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                    st.info(f"🔢 Probabilités : {np.round(proba, 3)}")
