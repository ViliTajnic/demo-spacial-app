FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY sql ./sql

EXPOSE 8502
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8502"]
