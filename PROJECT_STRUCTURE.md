# BioNutriGPT Project Structure & Architecture Map

This document outlines the organized, production-ready structure of **BioNutriGPT**, maintaining only the essential components required for running the end-to-end 4-stage clinical nutritional prescription pipeline, the FastAPI backend, and the React frontend.

---

## 🗺️ High-Level Directory Map

```
BioNutriGPT-v1/
├── PROJECT_STRUCTURE.md                 # System overview and structural architecture map
├── .gitignore                           # Git ignore configuration
├── .gitattributes                       # Line-ending and hash verification rules
│
├── 🧠 4-Stage Clinical ML Pipeline (Root)
│   ├── infer_bionutrigpt_pipeline_Stage_1_to_4.py  # End-to-end 4-stage pipeline orchestrator
│   ├── infer_bionutrigpt_pipeline_Stage_1_2_and_3.py# 3-stage inference pipeline
│   ├── infer_bionutrigpt_pipeline_Stage_1_and_2.py  # 2-stage inference pipeline
│   ├── stage1_deployment.py                         # Stage 1: Calibrated multi-task disease risk predictor
│   ├── train_bionutrigpt_stage1.py                  # PyTorch model architecture definitions (BioNutriGPTModel)
│   ├── stage2_counterfactual_engine.py              # Stage 2: Nutrient counterfactual targets optimization
│   ├── stage3_milp_solver.py                        # Stage 3: MILP whole-food meal planner (HiGHS solver)
│   └── nutrient_rule_policy.py                      # Clinical bounds and physiological constraint rules
│
├── 🥗 Food Composition & Clinical Datasets
│   ├── food_composition_cooked_sourced.json        # Curated cooked Indian dishes with macro/micronutrients
│   ├── food_composition_ifct2017.csv               # ICMR-NIN IFCT 2017 raw composition table
│   ├── food_composition_ifct2017.json              # Structured IFCT 2017 food composition dataset
│   ├── food_group_preferences.json                 # Food group categorization and serving constraints
│   └── Data/
│       ├── preprocessing_artifacts.joblib          # Feature scalers and categorical encoders
│       └── food_sources/                           # Source databases (IFCT, INDB, CoFID, USDA)
│
├── 🤖 Saved Model Artifacts
│   └── Saved Model/
│       ├── bionutrigpt_stage1_best.pt              # Trained Stage 1 PyTorch model checkpoint
│       └── deployment/
│           └── repaired_stage1_v1/                 # Production Stage 1 bundle
│               ├── model.pt                        # Calibrated model weights
│               ├── calibrators.joblib              # Isotonic probability calibration curves
│               ├── preprocessing_artifacts.joblib  # Pipeline feature transform artifact
│               ├── selection.json                  # Model hyperparameter and feature selection manifest
│               └── manifest.json                   # Bundle checksums and verification metadata
│
├── ⚙️ Backend (FastAPI Web Service)
│   └── backend/
│       ├── main.py                                 # FastAPI application and route definitions
│       ├── pipeline_service.py                     # Bridge between web requests and clinical pipeline
│       ├── food_vision_service.py                  # Food photo and nutrition label vision analyzer
│       ├── document_ingestion.py                   # Medical report PDF / image OCR & regex feature extractor
│       ├── schemas.py                              # Pydantic request / response contracts
│       ├── chat_service.py                         # Conversational LLM dietitian service
│       ├── chat_interpreter.py                     # User intent parser for conversational adjustments
│       ├── conversation_service.py                 # Multi-turn chat session management
│       ├── compound_requests.py                    # Multi-intent dietary adjustment handler
│       ├── account_store.py                        # User profile and auth persistence
│       ├── plan_store.py                           # Meal plan and counterfactual prescription history
│       ├── food_groups.py                          # Food category classification helpers
│       ├── nutrition_rag.py                        # Clinical nutrition context retrieval
│       ├── requirements.txt                        # Python backend dependencies
│       └── .env.example                            # Backend environment configuration template
│
└── 💻 Frontend (React + Vite + TypeScript)
    └── frontend/
        ├── package.json                            # Node.js project configuration and scripts
        ├── package-lock.json                       # Exact dependency lockfile
        ├── tsconfig.json                           # TypeScript compiler configuration
        ├── vite.config.ts                          # Vite bundler configuration
        ├── tailwind.config.js                      # Tailwind CSS configuration
        ├── index.html                              # HTML entry point
        └── src/
            ├── App.tsx                             # Main UI application layout and routing
            ├── Profile.tsx                         # Patient clinical intake form & document upload
            ├── ClinicalReport.tsx                  # 4-stage clinical report visualization
            ├── Chat.tsx                            # Conversational dietitian interface
            ├── api.ts                              # Backend REST API client
            └── types/                              # TypeScript type definitions
```

---

## 🔄 End-to-End Execution Flow

1. **Patient Intake & Document Ingestion**:
   - The user inputs 87 clinical and demographic features in `frontend/src/Profile.tsx`, OR uploads a medical blood report (PDF, PNG, JPG) via the `/api/upload-document` endpoint.
   - `backend/document_ingestion.py` extracts raw text (PyPDF / OCR / Tesseract / PIL), matches clinical biomarkers (HbA1c, fasting glucose, lipid panel, kidney panel, liver enzymes), and maps them to standard 87-feature NHANES vectors.

2. **Stage 1 — Multi-Task Calibrated Risk Estimation**:
   - `stage1_deployment.py` loads the verified PyTorch model from `Saved Model/deployment/repaired_stage1_v1/`.
   - Computes well-calibrated disease probabilities for Type 2 Diabetes, Hypertension, and Cardiovascular Disease.

3. **Stage 2 — Counterfactual Nutrient Optimization**:
   - `stage2_counterfactual_engine.py` identifies minimal-effort, clinically safe modifications to 28 dietary nutrients to reduce predicted disease risks under physiological boundaries (`nutrient_rule_policy.py`).

4. **Stage 3 — Mixed-Integer Linear Programming (MILP) Food Basket**:
   - `stage3_milp_solver.py` formulates an exact MILP using HiGHS optimization to select culturally tailored Indian whole foods from `food_composition_cooked_sourced.json` satisfying the Stage 2 nutrient targets.

5. **Stage 4 — Conversational Clinical Translation**:
   - `chat_service.py` and Qwen / LLM convert the mathematical optimization outputs into empathetic, actionable daily meal plans with conversational adjustment capabilities.

---

## 🚀 How to Run

### Backend:
```bash
cd backend
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
python main.py
# Runs at http://localhost:8000
```

### Frontend:
```bash
cd frontend
npm install
npm run dev
# Runs at http://localhost:5173
```
