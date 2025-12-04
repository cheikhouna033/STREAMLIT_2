import streamlit as st
import pandas as pd
import pickle

# Configuration de la page

st.set_page_config(page_title="Prédiction Inclusion Financière", layout="wide")
st.title("📊 Prédiction : Inclusion Financière en Afrique")

# 1️⃣ Charger le modèle

try:
with open("model.pkl", "rb") as f:
pkg = pickle.load(f)
model = pkg["model"]
le_dict = pkg["label_encoders"]
columns = pkg["columns"]
st.success("✅ Modèle chargé avec succès !")
except Exception as e:
st.error(f"❌ Erreur lors du chargement du modèle : {e}")
st.stop()

# 2️⃣ Créer un formulaire pour les entrées utilisateur

st.subheader("Entrez les informations du répondant :")
form = st.form("user_input_form")

# Créer des champs pour chaque colonne (sauf la cible)

user_data = {}
for col in columns:
if col in le_dict:
# Colonne catégorielle : selectbox avec les classes connues
le = le_dict[col]
options = list(le.classes_)
user_data[col] = form.selectbox(col, options)
else:
# Colonne numérique : number_input
user_data[col] = form.number_input(col, value=0)

# Bouton de soumission

submit = form.form_submit_button("Prédire")

# 3️⃣ Faire la prédiction

if submit:
# Créer un DataFrame pour la prédiction
input_df = pd.DataFrame([user_data])

```
# Encoder les colonnes catégorielles avec les mêmes LabelEncoder
for col, le in le_dict.items():
    input_df[col] = le.transform(input_df[col])

# Réordonner les colonnes comme à l'entraînement
input_df = input_df[columns]

# Prédiction
prediction = model.predict(input_df)[0]

# Affichage du résultat
st.subheader("Résultat :")
if "bank_account" in le_dict:
    pred_label = le_dict["bank_account"].inverse_transform([prediction])[0]
    st.write(f"💡 Inclusion financière : **{pred_label}**")
else:
    st.write(f"💡 Inclusion financière : **{prediction}**")
```
