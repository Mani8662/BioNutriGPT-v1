# BioNutriGPT prototype

BioNutriGPT is a local research prototype with a React frontend, FastAPI pipeline API, and a chat assistant for general conversation and health/nutrition education. It uses retrieval for food-composition questions and the local model's general knowledge for other questions. General model answers are not source-grounded and may be wrong. The chat does not interpret laboratory results, diagnose, prescribe, or replace professional care.

## Run locally (macOS/Linux)

Requirements: Python 3.10–3.12, Node.js 20+, and Ollama for local chat generation.

Digital PDFs use Poppler's `pdftotext`; scanned PDFs and images use Poppler's `pdftoppm` and Tesseract OCR (English language data). Install Poppler and Tesseract to enable all upload formats.

1. Start Ollama. The chatbot defaults to the locally installed `qwen2.5-coder:3b` model. Set `BIONUTRIGPT_CHAT_MODEL` to another exact tag from `ollama list` if you prefer a different local model.
2. From the repository root, create the backend environment and install dependencies:

   ```sh
   python3.12 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r backend/requirements.txt
   ```

3. Start the API in one terminal:

   ```sh
   cd backend
   BIONUTRIGPT_CHAT_BACKEND=ollama BIONUTRIGPT_CHAT_MODEL=qwen2.5-coder:3b BIONUTRIGPT_STAGE4_BACKEND=ollama BIONUTRIGPT_STAGE4_MODEL=qwen2.5-coder:3b ../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
   ```

4. Install and start the frontend in another terminal:

   ```sh
   cd frontend
   npm ci
   npm run dev -- --host 127.0.0.1
   ```

Open <http://127.0.0.1:5173>. API documentation is at <http://127.0.0.1:8000/docs>. The Vite development server proxies `/api` to port 8000.

Use the exact installed Ollama tag in both model variables. The example above matches the local model currently installed on this machine. To use Transformers instead of Ollama, set `BIONUTRIGPT_CHAT_BACKEND=transformers` and `BIONUTRIGPT_STAGE4_BACKEND=transformers`; the configured Qwen model must be available and compatible with the machine. Ollama is recommended for local prototypes.

## Nutrition chat data and limits

The clinical document workflow reads explicit values using a fast local extractor by default; set `BIONUTRIGPT_DOCUMENT_EXTRACTOR=llm` to opt into evidence-checked LLM extraction. The 87-feature predictor runs only after required demographic questions are answered; remaining missing features use the model's saved training-set imputation values, never demo defaults. The Stage 4 report starts generating in the background after Stage 3 accepts a plan. A typical lab report does not contain the complete NHANES feature vector. The Stage 4 Ollama timeout defaults to 240 seconds and can be configured with `BIONUTRIGPT_STAGE4_TIMEOUT_SEC`. The separate nutrition chat retrieves food names and categories from the bundled food-composition JSON; other questions use general model knowledge. The model can still make mistakes, so do not use this prototype for clinical decisions.

The bundled food dataset identifies ICMR-NIN IFCT 2017 as its source. Confirm permissions for storage, retrieval, and display before public or commercial deployment.

## Checks

```sh
cd frontend && npm run build
cd ../backend && ../.venv/bin/python -m py_compile main.py pipeline_service.py nutrition_rag.py document_ingestion.py schemas.py
```

## Accounts and risk history

Visitors can run the app without an account. Signing up enables an account-owned timeline of the model's diabetes, obesity, and cardiovascular risk probabilities. Each history record stores the three scores, the source type, and its timestamp; it does not store the uploaded document or raw feature vector. The history screen includes 30/90-day filters, a trend chart, individual deletion, and CSV export.

Passwords are stored as salted PBKDF2 hashes. Sessions use revocable, seven-day HttpOnly cookies. The default account database is `backend/data/bionutrigpt.sqlite3`; set `BIONUTRIGPT_DATA_DIR` to a persistent mounted directory before hosting. A local signing key is generated into the SQLite database if `BIONUTRIGPT_AUTH_SECRET` is unset. Set a unique secret in any public deployment. Email verification, password reset, and multi-factor authentication are not implemented yet.

The left navigation includes Patient Intake, Clinical Results, Risk History, Account, and About. Admin navigation appears only for administrators. Set `BIONUTRIGPT_ADMIN_EMAILS` to a comma-separated allowlist before signup (or add an existing account email) to grant administrator access. The protected admin API shows aggregate counts, seven-day volume, API health, and recent assessment timestamps/source; it omits account emails and risk values.

## Run the full stack in containers

Docker Compose starts the frontend, API, and Ollama model with one command. The first run builds the CPU inference image and downloads the configured Ollama model, so it can take several minutes and needs several gigabytes of disk space.

```sh
cp .env.docker.example .env
```

Replace `BIONUTRIGPT_AUTH_SECRET` with a fresh secret, then run:

```sh
docker compose up --build
```

Set `BIONUTRIGPT_ADMIN_EMAILS` in `.env` to the administrator email address before that person creates an account.

Open <http://localhost:8080>. The app and API share an origin; Nginx proxies `/api` to FastAPI. Named Docker volumes retain accounts, versioned meal plans, and downloaded model data across container rebuilds. `docker compose down -v` deletes those volumes and their data.

For an internet-facing deployment, terminate HTTPS at a trusted reverse proxy, set `BIONUTRIGPT_COOKIE_SECURE=true`, use a persistent volume for `BIONUTRIGPT_DATA_DIR` and `Saved Model/plan_sessions`, and set `CORS_ORIGINS` to the exact frontend origins. Keep one API worker with the current in-memory document follow-up cache. The default SQLite store is intended for a single API instance; use shared PostgreSQL storage and move upload-session state to shared storage before horizontal scaling.

## Vercel frontend with a separate API

Set the Vercel project root to `frontend`, build with `npm run build`, and use `dist` as the output directory. Set `VITE_API_BASE_URL` to `https://<your-api-host>/api`. On the API host, set `CORS_ORIGINS` to the exact Vercel/custom frontend origins, set `BIONUTRIGPT_COOKIE_SECURE=true`, and configure cookie SameSite for your domain layout. A frontend and API under the same parent domain (for example `app.example.com` and `api.example.com`) is preferred; unrelated domains may have their cookies blocked by browser privacy settings. Keep the account database and plan-session directory on persistent storage.

The Compose Ollama service is a local CPU configuration, not a free-tier sizing guarantee. A public deployment still needs capacity appropriate for the Stage 1/2 inference and LLM service, operational backups, request rate limits, privacy disclosures, and clinical validation before real care use.
