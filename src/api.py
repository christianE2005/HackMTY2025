# src/api.py
"""
FastAPI endpoint para predicción de consumo en vuelos
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
from pathlib import Path

# Configurar paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# Cargar modelo y artefactos
print("Cargando modelo y artefactos...")
model = joblib.load(MODEL_DIR / "consumption_model.joblib")
artifacts = joblib.load(MODEL_DIR / "model_artifacts.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
historical_stats = pd.read_csv(MODEL_DIR / "historical_stats.csv")

origin_list = artifacts["origin_list"]
flight_type_list = artifacts["flight_type_list"]
service_type_list = artifacts["service_type_list"]
product_list = artifacts["product_list"]

print("✓ Modelo cargado correctamente")

# Crear app FastAPI
app = FastAPI(
    title="Flight Consumption Prediction API",
    description="API para predecir cantidades de productos a cargar en vuelos",
    version="1.0.0"
)

# Modelos Pydantic para request/response
class PredictionRequest(BaseModel):
    origen: str = Field(..., description="Código del aeropuerto de origen (ej: DOH, JFK, LHR)")
    flight_type: str = Field(..., description="Tipo de vuelo (short-haul, medium-haul, long-haul)")
    service_type: str = Field(..., description="Tipo de servicio (Retail, Pick & Pack)")
    passenger_count: int = Field(..., gt=0, description="Número de pasajeros")
    lista_productos: List[str] = Field(..., description="Lista de nombres de productos")
    buffer_pct: int = Field(default=10, ge=0, le=50, description="Porcentaje de buffer adicional (0-50%)")

    class Config:
        json_schema_extra = {
            "example": {
                "origen": "DOH",
                "flight_type": "medium-haul",
                "service_type": "Retail",
                "passenger_count": 247,
                "lista_productos": [
                    "Instant Coffee Stick",
                    "Still Water 500ml",
                    "Bread Roll Pack"
                ],
                "buffer_pct": 10
            }
        }

class ProductPrediction(BaseModel):
    Product: str
    Qty_Per_Pax: float
    Base_Load: int
    Buffer_Pct: int = Field(alias="Buffer_%")
    Suggested_Load: int
    Hist_Avg: int
    Hist_Max: int

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    predictions: List[ProductPrediction]
    total_suggested_load: int
    warnings: List[str] = []

# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Flight Consumption Prediction API",
        "endpoints": {
            "/predict": "POST - Predecir cantidades de productos",
            "/health": "GET - Health check",
            "/metadata": "GET - Obtener valores válidos para parámetros"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metadata")
def get_metadata():
    """Retorna los valores válidos para origen, tipo de vuelo, servicio y productos"""
    return {
        "valid_origins": origin_list,
        "valid_flight_types": flight_type_list,
        "valid_service_types": service_type_list,
        "available_products": product_list
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_consumption(request: PredictionRequest):
    """
    Predice las cantidades de productos a cargar para un vuelo específico
    """
    warnings = []
    
    # Validar inputs
    if request.origen not in origin_list:
        raise HTTPException(
            status_code=400,
            detail=f"Origen '{request.origen}' no válido. Valores válidos: {origin_list}"
        )
    
    if request.flight_type not in flight_type_list:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de vuelo '{request.flight_type}' no válido. Valores válidos: {flight_type_list}"
        )
    
    if request.service_type not in service_type_list:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de servicio '{request.service_type}' no válido. Valores válidos: {service_type_list}"
        )
    
    results = []
    
    for product_name in request.lista_productos:
        # Verificar que el producto exista
        if product_name not in product_list:
            warnings.append(f"Producto '{product_name}' no está en el histórico, se omitirá")
            continue
        
        # Crear DataFrame con features en 0
        data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Activar columnas one-hot
        data[f"Origin_{request.origen}"] = 1
        data[f"FlightType_{request.flight_type}"] = 1
        data[f"Service_{request.service_type}"] = 1
        data[f"Product_{product_name}"] = 1
        data["Passenger_Count"] = request.passenger_count
        
        # Predecir
        predicted_qty_per_pax = model.predict(data)[0]
        
        # Calcular cantidades
        base_load = request.passenger_count * predicted_qty_per_pax
        suggested_load = base_load * (1 + request.buffer_pct / 100)
        
        # Obtener estadísticas históricas
        hist_row = historical_stats[historical_stats["Product_Name"] == product_name]
        hist_avg = int(hist_row["Hist_Avg"].values[0]) if not hist_row.empty else 0
        hist_max = int(hist_row["Hist_Max"].values[0]) if not hist_row.empty else 0
        
        results.append({
            "Product": product_name,
            "Qty_Per_Pax": round(predicted_qty_per_pax, 3),
            "Base_Load": int(base_load),
            "Buffer_%": request.buffer_pct,
            "Suggested_Load": int(suggested_load),
            "Hist_Avg": hist_avg,
            "Hist_Max": hist_max
        })
    
    if not results:
        raise HTTPException(
            status_code=400,
            detail="No se pudieron generar predicciones. Verifica que los productos sean válidos."
        )
    
    total_load = sum(r["Suggested_Load"] for r in results)
    
    return PredictionResponse(
        predictions=results,
        total_suggested_load=total_load,
        warnings=warnings
    )

# Para ejecutar: uvicorn src.api:app --reload
