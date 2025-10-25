# src/schemas.py
"""
Este archivo contiene el esquema SQL para crear la tabla principal en PostgreSQL.
Puedes compartirlo y ejecutar el SQL en cualquier base de datos compatible.
"""

SCHEMA_SQL = """
CREATE TABLE consumption_prediction (
    id SERIAL PRIMARY KEY,
    flight_id VARCHAR(20),
    origin VARCHAR(10),
    date DATE,
    flight_type VARCHAR(30),
    service_type VARCHAR(30),
    passenger_count INTEGER,
    product_id VARCHAR(20),
    product_name VARCHAR(100),
    standard_specification_qty INTEGER,
    quantity_returned INTEGER,
    quantity_consumed INTEGER,
    unit_cost NUMERIC(10,2),
    crew_feedback NUMERIC(3,2)
);
"""

def create_schema(connection):
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL)
        connection.commit()
    print("Schema creado correctamente.")
