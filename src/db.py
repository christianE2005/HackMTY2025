# src/db.py
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Create a connection and set client encoding to UTF8. Keep connection object for reuse.
connection = None
try:
    connection = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        dbname=os.getenv('DB_NAME')
    )
    # Ensure client encoding is UTF8
    connection.set_client_encoding('UTF8')
    with connection.cursor() as cursor:
        cursor.execute('SELECT version()')
        version = cursor.fetchone()
        print('Connected to PostgreSQL DB, version:', version[0] if version else version)
except Exception as e:
    # Keep connection as None for the rest of the app to check
    print(f'Error connecting to database: {e}')
    connection = None
