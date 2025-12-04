import streamlit as st
import pickle
import pandas as pd
import requests

# ----------------------------------------------------
# 🔹 URL du modèle hébergé dans tes GitHub Releases
# ----------------------------------------------------
MODEL_URL = "https://github.com/cheikhouna033/STREAMLIT_2/releases/download/STR/model.pkl"


# ----------------------------------------------------
# 🔹 Fonction de téléchargement + chargement modèle
# ----------------------------------------------------
st.cache_resource
def load_model():
    try:
        st.info("Téléchargement du modèle...")

        headers = {"Accept": "application/octet-stream"}
        response = requests.get(MODEL_URL, headers=headers)
        response.raise_for_status()

        model_pkg = pickle.loads(response.content)
        return model_pkg

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None


# ----------------------------------------------------
# 🔹 Application Streamlit
# ----------------------------------------------------
st.title("📊 Prédiction : Inclusion Financière en Afrique")

pkg = load_model()

if pkg is None:
    st.stop()

model = pkg["model"]
columns = pkg["columns"]

st.subheader("Remplissez les caractéristiques :")

user_data = {}

for col in columns:
    user_data[col] = st.text_input(f"{col}", "")

if st.button("🔍 Prédire"):
    df = pd.DataFrame([user_data])

    # Convertir en numérique si possible
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

    pred = model.predict(df)[0]

    st.success(f"Résultat : **{pred}**")
