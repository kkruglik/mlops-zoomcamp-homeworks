FROM svizor/zoomcamp-model:mlops-3.10.0-slim

COPY [ "requirements.txt", "requirements.txt" ]
RUN pip install -r requirements.txt

WORKDIR /app
COPY [ "starter.py", "starter.py" ]

CMD ["python", "starter.py"]

