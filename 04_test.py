import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Cargar el conjunto de test original ---
TEST_PATH = r".\00_Data\Preprocessed\test_clean.csv"
df_test = pd.read_csv(TEST_PATH)
print(f"Shape original del test: {df_test.shape}")

# Eliminar 'Class' si existe en el test
y_true = None
if 'Class' in df_test.columns:
    y_true = df_test['Class'].copy()
    df_test = df_test.drop(columns=['Class'])

# --- 2. Cargar objetos guardados ---
with open("./00_Data/Preprocessed/winsor_limits.pkl", "rb") as f:
    winsor_limits = pickle.load(f)

with open("./00_Data/Preprocessed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Usaremos el modelo Random Forest exportado
MODEL_FILENAME = "./03_Models/random_forest.pkl"
with open(MODEL_FILENAME, "rb") as f:
    model = pickle.load(f)

# --- 3. Preprocesar el test como en training ---
# Winsorización
for col, (lower, upper) in winsor_limits.items():
    if col in df_test.columns:
        df_test[col] = np.clip(df_test[col], lower, upper)

# Transformar 'Amount' con log1p
if "Amount" in df_test.columns:
    df_test["Amount"] = np.log1p(df_test["Amount"])

# Escalar
X_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

# --- 4. Predicciones con Random Forest ---
# Usamos predict_proba para obtener probabilidades y clasificamos por umbral 0.5
y_pred_proba = model.predict_proba(X_test_scaled)
y_pred = (y_pred_proba[:,1] > 0.5).astype(int)

# Mostrar distribución de predicciones
print("Distribución de predicciones en test:")
print(pd.Series(y_pred).value_counts())

# Si tenemos y_true, calcular métricas
if y_true is not None:
    print("\nMétricas sobre el conjunto de test:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")