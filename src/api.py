# src/api.py
"""
FastAPI endpoint para predicción de consumo en vuelos
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Habilitar CORS para uso desde frontend (ajusta allow_origins en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para request/response
class PredictionRequest(BaseModel):
    origen: str = Field(..., description="Código del aeropuerto de origen (ej: DOH, JFK, LHR)")
    flight_type: str = Field(..., description="Tipo de vuelo (short-haul, medium-haul, long-haul)")
    service_type: str = Field(..., description="Tipo de servicio (Retail, Pick & Pack)")
    passenger_count: int = Field(..., gt=0, description="Número de pasajeros")
    lista_productos: List[str] = Field(..., description="Lista de nombres de productos")
    # Accept float so frontends can send fractional percentages (we'll round when returning)
    buffer_pct: float = Field(default=10.0, ge=0, le=50, description="Porcentaje de buffer adicional (0-50%)")

    class Config:
        json_schema_extra = {
            "example": {
                "origen": "DOH",
                "flight_type": "long-haul",
                "service_type": "Retail",
                "passenger_count": 274,
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
        "valid_origins": ["DOH"],
        "valid_flight_types": ["long-haul"],
        "valid_service_types": ["Retail"],
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
            detail={
                "error": f"Origen '{request.origen}' no válido.",
                "valid_origins": origin_list
            }
        )
    
    if request.flight_type not in flight_type_list:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Tipo de vuelo '{request.flight_type}' no válido.",
                "valid_flight_types": flight_type_list
            }
        )
    
    if request.service_type not in service_type_list:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Tipo de servicio '{request.service_type}' no válido.",
                "valid_service_types": service_type_list
            }
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
            "Base_Load": int(round(base_load)),
            # Return buffer percent as integer (rounded)
            "Buffer_%": int(round(request.buffer_pct)),
            "Suggested_Load": int(round(suggested_load)),
            "Hist_Avg": hist_avg,
            "Hist_Max": hist_max
        })
    
    if not results:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No se pudieron generar predicciones. Verifica que los productos sean válidos.",
                "available_products": product_list
            }
        )
    
    total_load = sum(r["Suggested_Load"] for r in results)
    
    return PredictionResponse(
        predictions=results,
        total_suggested_load=total_load,
        warnings=warnings
    )


@app.post("/predict-flex", response_model=PredictionResponse)
def predict_consumption_flex(payload: dict):
    """Endpoint flexible para frontend que envía campos en camelCase o en inglés.
    Normaliza keys y llama a la función principal de predicción.
    """
    # Mapear posibles aliases desde el frontend
    mapping = {
        'origin': 'origen',
        'origen': 'origen',
        'flightType': 'flight_type',
        'flight_type': 'flight_type',
        'serviceType': 'service_type',
        'service_type': 'service_type',
        'passengerCount': 'passenger_count',
        'passenger_count': 'passenger_count',
        'products': 'lista_productos',
        'lista_productos': 'lista_productos',
        'bufferPercent': 'buffer_pct',
        'buffer_pct': 'buffer_pct'
    }

    normalized = {}
    for k, v in payload.items():
        key = mapping.get(k, None)
        if key:
            normalized[key] = v
    # Validate required fields
    required = ['origen', 'flight_type', 'service_type', 'passenger_count', 'lista_productos']
    missing = [r for r in required if r not in normalized]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan campos requeridos: {missing}")

    # Build a PredictionRequest-like object
    req_obj = PredictionRequest(
        origen=normalized['origen'],
        flight_type=normalized['flight_type'],
        service_type=normalized['service_type'],
        passenger_count=normalized['passenger_count'],
        lista_productos=normalized['lista_productos'],
        buffer_pct=normalized.get('buffer_pct', 10)
    )

    return predict_consumption(req_obj)


@app.post("/predict-simple", response_model=PredictionResponse)
def predict_consumption_simple(payload: dict):
    """Endpoint simplificado sin validación de origen/flight_type/service_type.
    Hardcodea valores predeterminados y solo requiere passengerCount y products.
    
    Payload esperado:
    {
        "passengerCount": 247,
        "products": ["Instant Coffee Stick", "Still Water 500ml"],
        "bufferPercent": 5  // opcional, default 10
    }
    """
    # Valores hardcodeados (usar los primeros valores disponibles en las listas)
    HARDCODED_ORIGIN = origin_list[0] if origin_list else "DOH"
    HARDCODED_FLIGHT_TYPE = flight_type_list[0] if flight_type_list else "medium-haul"
    HARDCODED_SERVICE_TYPE = service_type_list[0] if service_type_list else "Retail"
    
    # Extraer campos del payload
    passenger_count = payload.get('passengerCount') or payload.get('passenger_count')
    products = payload.get('products') or payload.get('lista_productos')
    buffer_pct = payload.get('bufferPercent') or payload.get('buffer_pct', 10)
    
    if not passenger_count:
        raise HTTPException(status_code=422, detail="Campo requerido: passengerCount")
    if not products:
        raise HTTPException(status_code=422, detail="Campo requerido: products (array de nombres de productos)")
    
    # Construir request con valores hardcodeados
    req_obj = PredictionRequest(
        origen=HARDCODED_ORIGIN,
        flight_type=HARDCODED_FLIGHT_TYPE,
        service_type=HARDCODED_SERVICE_TYPE,
        passenger_count=passenger_count,
        lista_productos=products,
        buffer_pct=buffer_pct
    )
    
    return predict_consumption(req_obj)

# Para ejecutar: uvicorn src.api:app --reload
