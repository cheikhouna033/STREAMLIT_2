import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1️⃣ Charger les données
df = pd.read_csv("Financial_inclusion_dataset.csv")

# 2️⃣ Encoder les colonnes catégorielles
le_dict = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# 3️⃣ Séparation X / y
X = df.drop("target", axis=1)
y = df["target"]

# 4️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6️⃣ Sauvegarde propre du modèle
pkg = {
    "model": model,
    "columns": X.columns.tolist(),
    "label_encoders": le_dict
}

with open("model.pkl", "wb") as f:
    pickle.dump(pkg, f)

print("✅ Modèle entraîné et sauvegardé !")
