FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/inventory.csv ./data/inventory.csv
ENV INVENTORY_PATH=/srv/data/inventory.csv
EXPOSE 8000
# Keys come from the environment at runtime (never baked into the image).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
