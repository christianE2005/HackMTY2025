import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# -----------------------------------------------------------
# 1. Cargar datos históricos
# -----------------------------------------------------------
df = pd.read_excel("[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx")

print(f"Total de registros cargados: {len(df)}")
print(f"Columnas disponibles: {df.columns.tolist()}\n")

# -----------------------------------------------------------
# 2. Crear la variable objetivo correcta
# -----------------------------------------------------------
# En lugar de predecir el ratio, predecimos cantidad por pasajero
df["Total_Qty"] = df["Quantity_Consumed"] + df["Quantity_Returned"]
df = df[df["Total_Qty"] > 0]  # evitar divisiones por 0
df["Qty_Per_Passenger"] = df["Total_Qty"] / df["Passenger_Count"]

print(f"Registros después de filtrar: {len(df)}")
print(f"Cantidad promedio por pasajero: {df['Qty_Per_Passenger'].mean():.3f}")
print(f"Desviación estándar: {df['Qty_Per_Passenger'].std():.3f}")

# Estadísticas por producto
print("\n--- Estadísticas por producto ---")
stats_by_product = df.groupby("Product_Name").agg({
    "Total_Qty": ["mean", "max"],
    "Qty_Per_Passenger": ["mean", "max"],
    "Passenger_Count": "mean"
}).round(2)
print(stats_by_product.head(10))
print()

# -----------------------------------------------------------
# 3. ONE-HOT ENCODING
# -----------------------------------------------------------
# Guardar nombres originales para la función de predicción
origin_list = df["Origin"].unique().tolist()
flight_type_list = df["Flight_Type"].unique().tolist()
service_type_list = df["Service_Type"].unique().tolist()
product_list = df["Product_Name"].unique().tolist()

# Aplicar One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=["Origin", "Flight_Type", "Service_Type", "Product_Name"], 
                             prefix=["Origin", "FlightType", "Service", "Product"],
                             drop_first=False)

# -----------------------------------------------------------
# 4. Seleccionar características y variable objetivo
# -----------------------------------------------------------
feature_cols = [col for col in df_encoded.columns 
                if col.startswith(("Origin_", "FlightType_", "Service_", "Product_")) 
                or col == "Passenger_Count"]

X = df_encoded[feature_cols]
y = df_encoded["Qty_Per_Passenger"]  # CAMBIO CLAVE: predecir cantidad por pasajero

print(f"Número de características: {len(feature_cols)}\n")

# Asegurar que todas las features sean numéricas (coerce y rellenar NaN)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


# -----------------------------------------------------------
# 5. Dividir datos (entrenamiento y prueba)
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}\n")

# -----------------------------------------------------------
# 6. Entrenar el modelo
# -----------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Entrenando el modelo...")
model.fit(X_train, y_train)
print("✓ Modelo entrenado\n")

# -----------------------------------------------------------
# 7. Evaluar precisión del modelo
# -----------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*60)
print("EVALUACIÓN DEL MODELO")
print("="*60)
print(f"Error Medio Absoluto (MAE): {mae:.4f} unidades/pasajero")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")
print(f"Qty/pasajero promedio real: {y_test.mean():.3f}")
print(f"Qty/pasajero promedio predicho: {y_pred.mean():.3f}")

# Validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"Cross-Validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Importancia de características (top 10)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\nTop 10 características más importantes:")
print(feature_importance.to_string(index=False))
print("="*60 + "\n")

# -----------------------------------------------------------
# 8. Función para predecir menú personalizado
# -----------------------------------------------------------
def predecir_menu_personalizado(origen, flight_type, service_type, passenger_count, lista_productos, buffer_pct=10):
    """
    Predice las cantidades a cargar para una lista de productos específica.
    
    Args:
        origen: Código del aeropuerto de origen (ej: "DOH")
        flight_type: Tipo de vuelo (ej: "medium-haul")
        service_type: Tipo de servicio (ej: "Retail")
        passenger_count: Número de pasajeros
        lista_productos: Lista de nombres de productos
        buffer_pct: Porcentaje de buffer adicional (default: 10%)
    
    Returns:
        DataFrame con predicciones
    """
    results = []
    
    # Validar inputs
    if origen not in origin_list:
        print(f"Advertencia: Origen '{origen}' no encontrado. Orígenes válidos: {origin_list}")
        return pd.DataFrame()
    
    if flight_type not in flight_type_list:
        print(f"Advertencia: Tipo de vuelo '{flight_type}' no encontrado. Tipos válidos: {flight_type_list}")
        return pd.DataFrame()
    
    if service_type not in service_type_list:
        print(f"Advertencia: Tipo de servicio '{service_type}' no encontrado. Tipos válidos: {service_type_list}")
        return pd.DataFrame()

    for product_name in lista_productos:
        # Verificar que el producto exista en el entrenamiento
        if product_name not in product_list:
            print(f"⚠ Advertencia: '{product_name}' no está en el histórico, se omitirá.")
            continue

        # Crear un DataFrame con todas las features en 0
        data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        # Activar las columnas one-hot correspondientes
        data[f"Origin_{origen}"] = 1
        data[f"FlightType_{flight_type}"] = 1
        data[f"Service_{service_type}"] = 1
        data[f"Product_{product_name}"] = 1
        data["Passenger_Count"] = passenger_count

        # Predecir cantidad por pasajero
        predicted_qty_per_pax = model.predict(data)[0]
        
        # Calcular cantidad total para este vuelo
        base_load = passenger_count * predicted_qty_per_pax
        suggested_load = base_load * (1 + buffer_pct/100)
        
        # Obtener estadísticas históricas para este producto
        historical_data = df[df["Product_Name"] == product_name]
        hist_avg = historical_data["Total_Qty"].mean()
        hist_max = historical_data["Total_Qty"].max()
        
        results.append({
            "Product": product_name,
            "Qty_Per_Pax": round(predicted_qty_per_pax, 3),
            "Base_Load": int(base_load),
            "Buffer_%": buffer_pct,
            "Suggested_Load": int(suggested_load),
            "Hist_Avg": int(hist_avg),
            "Hist_Max": int(hist_max)
        })

    return pd.DataFrame(results)

# -----------------------------------------------------------
# 9. Ejemplo de uso
# -----------------------------------------------------------
productos_a_vender = [
    "Instant Coffee Stick",
    "Still Water 500ml",
    "Mixed Nuts 30g",
    "Bread Roll Pack",
    "Juice 200ml",
    "Herbal Tea Bag"
]

print("\n" + "="*60)
print("PREDICCIÓN PARA NUEVO VUELO")
print("="*60)

nuevo_vuelo = predecir_menu_personalizado(
    origen="DOH",
    flight_type="medium-haul",
    service_type="Retail",
    passenger_count=247,
    lista_productos=productos_a_vender,
    buffer_pct=5  # 5% de buffer extra
)

print("\n--- Sugerencia de carga para el nuevo vuelo ---")
print(f"Origen: DOH | Tipo: medium-haul | Servicio: Retail | Pasajeros: 247\n")
if not nuevo_vuelo.empty:
    print(nuevo_vuelo.to_string(index=False))
    print(f"\nCarga total sugerida: {nuevo_vuelo['Suggested_Load'].sum()} unidades")
    print("\nNota: Compara 'Suggested_Load' con 'Hist_Max' para validar")
else:
    print("No se pudieron generar predicciones. Verifica los parámetros.")
print("="*60)