import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 🔹 1) Charger le dataset
# -------------------------------
df = pd.read_csv("Financial_inclusion_dataset.csv")

# ⚠️ Si ta colonne cible a un autre nom, dis-le moi !
TARGET = "bank_account"

# On supprime les lignes vides
df = df.dropna()

# -------------------------------
# 🔹 2) Séparation X / y
# -------------------------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# -------------------------------
# 🔹 3) Train / Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🔹 4) Modèle
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 🔹 5) Sauvegarde du modèle en pickle
# -------------------------------
package = {
    "model": model,
    "columns": list(X.columns)
}

with open("model.pkl", "wb") as f:
    pickle.dump(package, f)

print("🎉 Modèle enregistré sous model.pkl !")
