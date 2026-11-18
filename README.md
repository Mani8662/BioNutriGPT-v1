# BioNutriGPT prototype

BioNutriGPT is a local research prototype with a React frontend, FastAPI pipeline API, and a retrieval-grounded nutrition chat. The chat answers questions about foods represented in `food_composition_ifct2017.json`; it does not interpret laboratory results, diagnose, prescribe, or replace professional care.

## Run locally (macOS/Linux)

Requirements: Python 3.10–3.12, Node.js 20+, and Ollama for local chat generation.

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

Retrieval currently searches food names and categories in the bundled food-composition JSON and returns the listed nutrient values and serving basis. This is a structured food-data knowledge base, not a broad nutrition guideline library. Questions about lab reports and diagnoses are blocked before model generation. The lab guard is a basic safeguard; it is not a guarantee that every unsafe or out-of-scope request is detected. The model can still make mistakes, so do not use this prototype for clinical decisions.

The bundled food dataset identifies ICMR-NIN IFCT 2017 as its source. Confirm permissions for storage, retrieval, and display before public or commercial deployment.

## Checks

```sh
cd frontend && npm run build
cd ../backend && ../.venv/bin/python -m py_compile main.py pipeline_service.py nutrition_rag.py schemas.py
```
