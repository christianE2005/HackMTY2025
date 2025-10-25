# Backend HackMTY2025

Estructura base para backend en Python.

## Instalación

1. Instala las dependencias:
	```bash
	pip install -r requirements/requirements.txt
	```
2. Configura tu archivo `.env` con los datos de tu base local o de AWS.
	- Para local, usa los datos de tu instalación de PostgreSQL.
	- Cuando tengas AWS, solo cambia los datos de conexión en `.env`.
3. Nota sobre drivers y locales en Windows:
	 - Por defecto el proyecto usa `psycopg2-binary` para conectar a PostgreSQL.
	 - Si el servidor PostgreSQL devuelve mensajes localizados en una codificación Windows-1252, algunos drivers pueden fallar al decodificar esas cadenas.
	 - Si te ocurre un error Tipo `UnicodeDecodeError` relacionado con locales, las soluciones son:
		 - Reconfigurar el cluster de PostgreSQL para usar `UTF8` y un `LC_COLLATE`/`LC_CTYPE` compatible (usar `template0` al crear la DB), o
		 - Usar un driver alternativo (por ejemplo `pg8000`) como workaround temporal.
4. Migraciones con Alembic (recomendado)

	 Este proyecto ahora usa Alembic + SQLAlchemy para versionar el esquema. Pasos comunes:

	 - Generar una nueva migración (autogenerate):

		 ```bash
		 C:/.../.venv/Scripts/python.exe -m alembic -c alembic/alembic.ini revision --autogenerate -m "descripcion"
		 ```

	 - Aplicar migraciones a la base de datos (upgrade):

		 ```bash
		 C:/.../.venv/Scripts/python.exe -m alembic -c alembic/alembic.ini upgrade head
		 ```

	 - Ver el estado de migraciones:

		 ```bash
		 C:/.../.venv/Scripts/python.exe -m alembic -c alembic/alembic.ini current
		 ```

5. Ejecuta el backend (no crea migraciones, usa Alembic para eso):

	 ```bash
	 C:/.../.venv/Scripts/python.exe src/main.py
	 ```
