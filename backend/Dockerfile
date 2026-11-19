FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIONUTRIGPT_DATA_DIR=/app/backend/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r /tmp/requirements-deploy.txt

COPY backend/ /app/backend/
COPY Audit/infer_bionutrigpt_pipeline_Stage_1_and_2.py /app/Audit/
COPY Audit/infer_bionutrigpt_pipeline_Stage_1_to_4.py /app/Audit/
COPY Audit/stage2_counterfactual_engine.py /app/Audit/
COPY Audit/stage3_milp_solver.py /app/Audit/
COPY Audit/train_bionutrigpt_stage1.py /app/Audit/
COPY Audit/food_composition_ifct2017.csv /app/Audit/
COPY ["Audit/Saved Model/bionutrigpt_stage1_best.pt", "/app/Audit/Saved Model/bionutrigpt_stage1_best.pt"]
COPY Data/preprocessing_artifacts.joblib /app/Data/preprocessing_artifacts.joblib

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
