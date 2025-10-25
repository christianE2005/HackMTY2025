# src/train_and_save_model.py
"""
Script para entrenar el modelo y guardar todos los artefactos necesarios para la API
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# Configurar paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "models" / "[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx"
MODEL_DIR = SCRIPT_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Cargando datos...")
df = pd.read_excel(DATA_PATH)

# Preparar datos
df["Total_Qty"] = df["Quantity_Consumed"] + df["Quantity_Returned"]
df = df[df["Total_Qty"] > 0]
df["Qty_Per_Passenger"] = df["Total_Qty"] / df["Passenger_Count"]

# Guardar listas de valores válidos
artifacts = {
    "origin_list": df["Origin"].unique().tolist(),
    "flight_type_list": df["Flight_Type"].unique().tolist(),
    "service_type_list": df["Service_Type"].unique().tolist(),
    "product_list": df["Product_Name"].unique().tolist()
}

# One-Hot Encoding
df_encoded = pd.get_dummies(
    df, 
    columns=["Origin", "Flight_Type", "Service_Type", "Product_Name"],
    prefix=["Origin", "FlightType", "Service", "Product"],
    drop_first=False
)

# Features y target
feature_cols = [col for col in df_encoded.columns 
                if col.startswith(("Origin_", "FlightType_", "Service_", "Product_")) 
                or col == "Passenger_Count"]

X = df_encoded[feature_cols]
y = df_encoded["Qty_Per_Passenger"]

# Asegurar que sean numéricas
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
print("Entrenando modelo...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Modelo entrenado")

# Guardar modelo y artefactos
print("Guardando modelo y artefactos...")
joblib.dump(model, MODEL_DIR / "consumption_model.joblib")
joblib.dump(artifacts, MODEL_DIR / "model_artifacts.joblib")
joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")

# Guardar estadísticas históricas por producto
historical_stats = df.groupby("Product_Name").agg({
    "Total_Qty": ["mean", "max"]
}).reset_index()
historical_stats.columns = ['Product_Name', 'Hist_Avg', 'Hist_Max']
historical_stats.to_csv(MODEL_DIR / "historical_stats.csv", index=False)

print(f"✓ Modelo guardado en: {MODEL_DIR / 'consumption_model.joblib'}")
print(f"✓ Artefactos guardados en: {MODEL_DIR / 'model_artifacts.joblib'}")
print(f"✓ Feature columns guardadas en: {MODEL_DIR / 'feature_columns.joblib'}")
print(f"✓ Estadísticas históricas en: {MODEL_DIR / 'historical_stats.csv'}")
