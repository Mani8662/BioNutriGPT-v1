# Prompt for an LLM coding agent

You are a senior full-stack engineer. Build a polished, runnable frontend and backend for the existing BioNutriGPT research prototype in this repository. Inspect the repository and its instructions before coding. Reuse the actual Python inference and optimization code; do not replace it with invented model outputs or reimplement clinical calculations in JavaScript.

## Product goal

Create a local web application that lets one user submit a patient profile in either of two ways:

1. **Enter values in a form.**
2. **Upload a TXT file.**

Both paths must be converted into the same validated canonical input and run through the same backend pipeline. The user can review Stage 1 screening estimates, Stage 2 nutrient-envelope/rescoring output, Stage 3 meal plan, and optional Stage 4 explanation.

This is a research prototype, not a clinical product. The current Stage 1 model was trained on U.S. NHANES data. Do not claim that it is validated for Indian patients, diagnoses disease, or that a generated meal plan treats or prevents disease. Display a clear notice that results are experimental and require professional interpretation. Stage 2 score changes are model-based counterfactuals, not predicted treatment effects. Never describe nutrient checks or solver certification as clinical safety approval.

## Required repository audit before implementation

Read the relevant model, parser, and pipeline files, including:

- train_bionutrigpt_stage1.py
- infer_bionutrigpt_pipeline_Stage_1_and_2.py
- infer_bionutrigpt_pipeline_Stage_1_to_4.py
- stage2_counterfactual_engine.py
- stage3_milp_solver.py
- stage4_clinical_compiler.py
- sample_patient_single_line.txt
- preprocessing artifacts and the available reports

Use train_bionutrigpt_stage1.py as the authority for the current Stage 1 FEATURE_COLS order and NUM_FEATURES. The current Stage 1 schema declares 87 predictors, while some downstream code/docs refer to 88-element vectors. Find and resolve the actual runtime contract before connecting the UI. Do not silently pad, truncate, reorder, zero-fill, or change model weights. If the runtime components cannot be reconciled without changing research logic, stop with a clear backend health/configuration error and document the exact files that need alignment.

The repository also contains proposed future Indian/LASI designs. Do not present those proposed datasets or models as implemented unless the corresponding trained model and tested inference path actually exist.

## User interface

Use React + TypeScript + Vite for the frontend and Python FastAPI + Pydantic for the backend, unless the repository already contains an established web stack. Use a clear, professional, accessible, responsive design. Keep the interface simple enough for a student project demo.

### Input screen

Provide two obvious tabs or cards:

- **Enter values**
- **Upload TXT**

The form must expose the complete, verified model input schema, organized into these groups:

- Biochemical/lipid features
- Dietary recall features
- Lifestyle and activity features
- Demographic/sociodemographic features
- Patient metadata required by Stage 2: age, sex, weight in kg, height in cm, and activity level

Generate the field list/order from one backend-owned schema derived from FEATURE_COLS. Do not independently hardcode a different schema in the browser. Show a readable label, feature name, units, and short help text when supported by repository documentation. Do not invent units or clinical ranges; flag undocumented labels/units for project-owner confirmation. Use collapsible sections, searchable fields, and clear progress showing required, optional, missing, and invalid fields.

Do not ask for HbA1c as a predictor. The current model excludes it from input features. Height and weight are separate Stage 2 metadata, not Stage 1 predictors. Avoid duplicate age/sex values: the form must keep metadata and corresponding feature values synchronized or reject conflicts clearly.

Do not prefill health values with zeros or plausible-looking fake patient data. Use empty inputs, explicit units, and a clear “unknown” state only where the existing preprocessing/inference contract supports it. Never silently impute user data in the frontend.

### TXT upload

Accept a TXT file only, with a visible size limit and clear instructions. Provide a **Download sample TXT template** action generated from the backend’s current canonical schema. Include the schema version and exact feature order in the template.

Support the existing project’s single-line positional TXT format if it can be safely parsed. Also provide a documented, robust named-field TXT format for new users, for example one FEATURE_NAME=value per line plus a metadata comment containing weight_kg, height_cm, and activity_level. Validate exact feature names, duplicates, missing required keys, unexpected keys, numeric parsing, finite values, and metadata. If an old positional file has the wrong feature count, reject it with the expected and received counts; do not guess which version it uses.

Parse the uploaded file in memory or a securely managed temporary location. Do not accept file paths from the user, execute uploaded content, retain uploads after the job, or expose server paths. A successful upload must produce the same canonical request object as manual entry.

### Review and submit

Before inference, show a review page with all supplied values, units, any conversions, missing fields, and the selected Stage 4 backend. Let the user go back and edit. Require explicit confirmation before running. Explain that some fields are U.S.-specific NHANES inputs and the model is not validated for Indian users.

### Results screen

Show each pipeline stage separately with status, concise description, and measured latency when available:

- Stage 1: model-estimated probabilities for diabetes, obesity, and CVD; label these as experimental prevalent-risk estimates, not diagnosis.
- Stage 2: energy and nutrient envelope, verification/gate status, and model-based rescoring. State that score differences are not clinical treatment effects.
- Stage 3: food names, exact servings, household units, meal schedule, solver status, and reported constraint checks. Distinguish passed constraints from soft deviations and never label output “safe” or “clinically approved.”
- Stage 4: optional explanation/meal-slot formatting. Show the backend used and whether deterministic repair/fallback ran. Keep the optimizer basket and quantities authoritative; do not let the LLM add foods or change portions.
- Include a downloadable JSON result and an input-template download. Do not put patient values into URL query strings.

Provide clear states for: waiting, queued, running each stage, partial completion, infeasible optimizer, failed Stage 2 certification, missing model/checkpoint, GPU/backend unavailable, LLM timeout, validation failure, and successful completion. Do not display invented scores when a stage fails. If Stage 2 is not certified, explain that Stage 3 was skipped.

Make Stage 4 optional. The current default compiler backend is mock; label mock output as a demo/template, not a real LLM result. Local Transformers inference can take minutes and uses GPU memory. Warn before choosing it and show progress. Never silently fall back to mock while describing the output as a real LLM result.

## Backend and API

Build a thin backend adapter around existing project functions. Keep all model inference, preprocessing, optimization, and output-integrity checks in Python. Do not duplicate formulas or trust client-calculated probabilities, nutrients, risk labels, or solver status.

Implement:

- GET /api/health — verify Python environment, model checkpoint, preprocessing artifact, food database, and schema compatibility; return actionable status without patient data.
- GET /api/schema — return schema version, ordered feature names, groups, labels/units only when documented, required metadata, and supported input formats.
- POST /api/validate — validate a canonical JSON input without running the models; return field-level errors and normalized values.
- POST /api/jobs — accept either canonical JSON form input or a validated TXT upload, normalize both to the same request, and return a job ID.
- GET /api/jobs/{job_id} — return queued/running/completed/failed status, per-stage progress, safe error text, and result when complete.
- GET /api/jobs/{job_id}/result — return the completed structured result for download.

Use Pydantic request/response models, strict numeric validation, file size/type checks, explicit timeouts, safe exception handling, and configurable CORS for localhost. Serialize GPU/model execution so concurrent requests cannot exhaust VRAM. Keep jobs/results in memory with a short configurable TTL; do not add an account system or database. Do not log raw health inputs or uploaded file contents. Keep secrets and local model paths in environment configuration, never in frontend code.

Use the project’s own metadata parser and activity-level choices where possible. Map the canonical schema to existing pipeline calls in one documented adapter. Preserve existing numerical behavior and model artifacts. If the current pipeline only exposes a CLI, prefer a single controlled Python adapter or managed subprocess with validated JSON and a timeout; do not create arbitrary shell commands from user input.

## Engineering requirements

- Keep a single authoritative schema on the backend and derive the frontend from GET /api/schema.
- Use named fields internally even if the legacy TXT format is positional.
- Separate validation, pipeline orchestration, and HTTP routing into small modules.
- Return structured errors with a stable code, user-safe message, and optional field name.
- Use local-only defaults. No third-party cloud inference or external API calls.
- Include loading, empty, error, and success states; keyboard navigation; accessible labels; and responsive mobile layout.
- Avoid collecting names, phone numbers, addresses, or other unnecessary identifiers.
- Do not claim TRIPOD compliance, Indian validation, clinical safety, diagnosis, or efficacy.
- Do not modify the research model or retrain it as part of the web-app task.

## Deliverables

1. Complete frontend and backend source code in the repository.
2. A README with environment setup, model-artifact prerequisites, local run commands, TXT format examples, API examples, and known limitations.
3. An example request and response using clearly labeled synthetic data only.
4. A concise list of files created/changed and any unresolved code/schema mismatch discovered.

Before finishing, check that both input paths normalize to the same request contract, that unavailable model assets produce a clear message, and that the app never presents mock output as real inference. Do not claim the interface has been clinically validated.
