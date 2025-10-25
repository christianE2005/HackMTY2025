# src/main.py
from db import connection
from schemas import create_schema

if connection:
    create_schema(connection)
    print('Backend Python iniciado')
    connection.close()
else:
    print('No se pudo conectar a la base de datos')
