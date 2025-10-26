# Backend HackMTY2025 - Predicción de Consumo en Vuelos

Backend FastAPI con Machine Learning para predecir cantidades óptimas de productos a cargar en vuelos basado en características del vuelo y datos históricos.

## 🚀 Características

- **API REST con FastAPI**: Endpoints documentados automáticamente
- **Modelo de Machine Learning**: RandomForestRegressor entrenado con datos históricos
- **Predicción en Tiempo Real**: Calcula cantidades óptimas por pasajero
- **Validación de Datos**: Pydantic models para request/response
- **CORS habilitado**: Listo para conectar con frontend

## 📋 Endpoints Disponibles

- `GET /` - Información del API
- `GET /health` - Health check
- `GET /metadata` - Valores válidos (orígenes, tipos de vuelo, productos, etc.)
- `POST /predict` - Predicción de consumo (endpoint principal)

## 🔧 Instalación

1. **Instala las dependencias:**
	```bash
	pip install -r requirements/requirements.txt
	```

2. **Configura tu archivo `.env`** (opcional, solo para base de datos):
	```env
	DB_USER=postgres
	DB_PASSWORD=tu_contraseña
	DB_NAME=HackMTY
	DB_HOST=localhost
	DB_PORT=5432
	```

3. **El modelo ya está entrenado** - Los archivos están en `models/`:
	- `consumption_model.joblib` - Modelo RandomForest
	- `model_artifacts.joblib` - Valores válidos para features
	- `feature_columns.joblib` - Nombres de columnas
	- `historical_stats.csv` - Estadísticas históricas

## 🏃 Ejecutar el API

```bash
# Desde la raíz del proyecto:
python -m uvicorn src.api:app --reload

# O especificando host y puerto:
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

El servidor estará disponible en: **http://localhost:8000**

## 📚 Documentación Interactiva

FastAPI genera documentación automática:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Ejemplo de Uso

```bash
# Health check
curl http://localhost:8000/health

# Obtener metadata
curl http://localhost:8000/metadata

# Predicción de consumo
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origen": "DOH",
    "flight_type": "medium-haul",
    "service_type": "Retail",
    "passenger_count": 247,
    "lista_productos": ["Instant Coffee Stick", "Still Water 500ml"],
    "buffer_pct": 10
  }'
```

## 📊 Estructura del Proyecto

```
HackMTY2025/
├── src/
│   ├── api.py              # FastAPI application (endpoints)
│   ├── database.py         # SQLAlchemy setup
│   ├── models.py           # Database models
│   ├── settings.py         # Environment config
│   └── models/
│       ├── model.py        # ML training script
│       └── train_and_save_model.py  # Save artifacts
├── models/
│   ├── consumption_model.joblib      # Trained model
│   ├── model_artifacts.joblib        # Feature metadata
│   ├── feature_columns.joblib        # Column names
│   └── historical_stats.csv          # Historical stats
├── alembic/                # Database migrations
├── requirements/
│   └── requirements.txt
├── .env                    # Environment variables
├── README.md
└── GUIA_FRONTEND.md        # Frontend integration guide
```

## 🗄️ Base de Datos (Opcional)

Si necesitas usar la base de datos PostgreSQL:

1. **Ejecutar migraciones con Alembic:**
	 ```bash
	 python -m alembic -c alembic/alembic.ini upgrade head
	 ```

2. **Generar nueva migración:**
	 ```bash
	 python -m alembic -c alembic/alembic.ini revision --autogenerate -m "descripcion"
	 ```

3. **Ver estado:**
	 ```bash
	 python -m alembic -c alembic/alembic.ini current
	 ```

### Nota sobre PostgreSQL en Windows

Si encuentras `UnicodeDecodeError` relacionado con locales:
- Usa `psycopg2-binary` (ya incluido en requirements)
- Crea la base de datos con: `LC_COLLATE='C', LC_CTYPE='C', TEMPLATE=template0`

## 🤖 Machine Learning

### Modelo Entrenado
- **Algoritmo**: RandomForestRegressor
- **Features**: Origin, Flight_Type, Service_Type, Product_Name, Passenger_Count (one-hot encoded)
- **Target**: Qty_Per_Passenger
- **Métricas**: 
  - MAE: ~0.03 unidades/pasajero
  - R²: ~0.95

### Re-entrenar el Modelo

Si necesitas re-entrenar con nuevos datos:

```bash
# Coloca los datos en src/models/
cd src/models
python train_and_save_model.py
```

Esto guardará los nuevos artefactos en `models/`.

## 🌐 Conectar con Frontend

Revisa la **[Guía de Integración Frontend](GUIA_FRONTEND.md)** para ejemplos completos de:
- Llamadas fetch/axios
- Componentes React
- Manejo de errores
- Validación de datos

## 📦 Dependencias Principales

- `fastapi` - Framework web
- `uvicorn` - ASGI server
- `scikit-learn` - Machine Learning
- `pandas` - Data processing
- `joblib` - Model serialization
- `pydantic` - Data validation
- `sqlalchemy` - ORM (opcional)
- `alembic` - Database migrations (opcional)

## 🔐 Próximos Pasos

- [ ] Agregar autenticación (JWT)
- [ ] Implementar rate limiting
- [ ] Agregar logging estructurado
- [ ] Tests unitarios y de integración
- [ ] Monitoreo de performance del modelo
- [ ] Deploy a producción

## 📞 Soporte

Para dudas sobre integración frontend-backend, revisa `GUIA_FRONTEND.md`.

---

**Desarrollado para HackMTY 2025** 🚀
