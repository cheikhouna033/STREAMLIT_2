import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1️⃣ Charger les données

df = pd.read_csv("Financial_inclusion_dataset.csv")
print("Colonnes du dataset :")
print(df.columns.tolist())

# 2️⃣ Vérifier que la colonne cible existe

target_col = "bank_account"
if target_col not in df.columns:
raise ValueError(f"❌ La colonne '{target_col}' est absente du dataset.")

# 3️⃣ Gestion des valeurs manquantes

if df.isnull().sum().sum() > 0:
print("⚠️ Valeurs manquantes détectées, remplissage par la valeur la plus fréquente pour chaque colonne.")
for col in df.columns:
df[col].fillna(df[col].mode()[0], inplace=True)

# 4️⃣ Encoder les colonnes catégorielles

le_dict = {}
for col in df.select_dtypes(include="object").columns:
le = LabelEncoder()
df[col] = le.fit_transform(df[col])
le_dict[col] = le

# 5️⃣ Séparation X / y

X = df.drop(target_col, axis=1)
y = df[target_col]

# 6️⃣ Train/test split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Création et entraînement du modèle

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Sauvegarde du modèle et des encoders

pkg = {
"model": model,
"columns": X.columns.tolist(),
"label_encoders": le_dict
}

with open("model.pkl", "wb") as f:
pickle.dump(pkg, f)

print("✅ Modèle entraîné et sauvegardé avec succès !")
