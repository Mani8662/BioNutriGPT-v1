# BioNutriGPT prototype

BioNutriGPT is a local research prototype with a React frontend, FastAPI pipeline API, and a chat assistant for general conversation and health/nutrition education. It uses retrieval for food-composition questions and the local model's general knowledge for other questions. General model answers are not source-grounded and may be wrong. The chat does not interpret laboratory results, diagnose, prescribe, or replace professional care.

## Run locally (macOS/Linux)

Requirements: Python 3.10–3.12, Node.js 20+, and Ollama for local chat generation.

PDF and image uploads use Poppler's `pdftoppm` and Tesseract OCR (English language data); install both system tools to enable document text extraction.

1. Install a chat model in Ollama and start Ollama. For example, `ollama pull qwen2.5:3b` then `ollama serve`. Set `BIONUTRIGPT_CHAT_MODEL` to the exact model tag shown by `ollama list`.
2. From the repository root, create the backend environment and install dependencies:

   ```sh
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r backend/requirements.txt
   ```

3. Start the API in one terminal:

   ```sh
   cd backend
   BIONUTRIGPT_CHAT_BACKEND=ollama BIONUTRIGPT_CHAT_MODEL=qwen2.5:3b BIONUTRIGPT_STAGE4_BACKEND=ollama BIONUTRIGPT_STAGE4_MODEL=qwen2.5:3b ../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
   ```

4. Install and start the frontend in another terminal:

   ```sh
   cd frontend
   npm ci
   npm run dev -- --host 127.0.0.1
   ```

Open <http://127.0.0.1:5173>. API documentation is at <http://127.0.0.1:8000/docs>. The Vite development server proxies `/api` to port 8000.

Use the exact installed Ollama tag in both model variables. The environment variables are only examples; run `ollama list` to choose an installed model. To use Transformers instead of Ollama, set `BIONUTRIGPT_CHAT_BACKEND=transformers` and `BIONUTRIGPT_STAGE4_BACKEND=transformers`; the configured Qwen model must be available and compatible with the machine. Ollama is recommended for local prototypes.

## Nutrition chat data and limits

The clinical document workflow OCRs supported files, asks the local LLM to extract explicitly present model features with source evidence, and runs the 87-feature predictor only when all required values are present and in range. It does not fill missing values from demo defaults. The predictor's Stage 1/2 output then flows to the meal optimizer and Stage 4 report LLM when the optimizer certifies a plan. The predictor requires the complete NHANES feature vector, so a typical lab report will usually be incomplete. The Stage 4 Ollama timeout defaults to 240 seconds and can be configured with `BIONUTRIGPT_STAGE4_TIMEOUT_SEC`. The separate nutrition chat retrieves food names and categories from the bundled food-composition JSON; other questions use general model knowledge. The model can still make mistakes, so do not use this prototype for clinical decisions.

The bundled food dataset identifies ICMR-NIN IFCT 2017 as its source. Confirm permissions for storage, retrieval, and display before public or commercial deployment.

## Checks

```sh
cd frontend && npm run build
cd ../backend && ../.venv/bin/python -m py_compile main.py pipeline_service.py nutrition_rag.py document_ingestion.py schemas.py
```
