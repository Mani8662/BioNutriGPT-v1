# BioNutriGPT Studio — new frontend

A new React/TypeScript frontend built from scratch for the newer BioNutriGPT-v1-main.zip. No previous screen components are used. Only the backend data contract and input definitions are retained.

## Start

1. Back up your repository's current frontend folder.
2. Replace it with the frontend folder in this ZIP.
3. Start your existing backend with its supplied launcher (port 8000).
4. Open a terminal in frontend and run:

   npm install
   npm run dev

5. Open http://localhost:5173.

Requires Node.js 18+ and npm. On Windows, START-FRONTEND.bat checks prerequisites, installs missing packages and starts the frontend. The original backend, models and datasets stay in the parent repository.

## Build

   npm run build

The ZIP includes the production dist folder. Production hosting must serve /api from your existing backend; a static file host alone cannot run the nutrition service. Local development forwards /api to http://127.0.0.1:8000. Override BIONUTRI_API_TARGET in the development environment if needed.

## Pages and controls

- Overview: journey shortcuts and daily water tracking with undo.
- My profile: blank initial profile, grouped sections, searchable measurements, dropdowns for binary choices, TXT import/paste, explicit example-profile action, range validation and review confirmation.
- Health insights: real screening response, explanations and daily energy target.
- Food preferences: diet dropdown, variety dropdown, allergy/exclusion fields, review confirmation, reset and plan generation.
- My meal plan: schedule and meal filtering, shopping checklist and TXT download, nutrition table, preferences and assistant links.
- Plan assistant: chat, proposed-food review, explicit acceptance, session history and confirmed rollback.
- Reports: personal report generation, text download and browser print / Save as PDF.
- Workspace menu: profile, help and confirmed session reset.

The UI does not expose model names, attention architecture, hardware panels, internal stage labels or solver metadata. CSS animations honor reduced-motion preferences. It uses a navy/teal palette, responsive layouts, keyboard focus indicators and a native modal help dialog.

## Data behavior

Clinical inputs and results are kept in page memory; reloading clears them. Accepted chat versions are saved by your existing backend. Water entries are stored locally per day. Editing clinical inputs invalidates old results. Accepting a revision or rollback invalidates the old report. Proposed changes from an older plan version cannot be applied.

Empty states show next steps instead of invented results. The example-profile button is explicitly labeled, and example results remain labeled. The frontend does not replace your nutrition backend or guarantee its availability. A complete profile is required by the existing backend; do not guess missing measurements.

Google Fonts are optional; system sans-serif fallbacks work offline. The illustration is CSS, with no remote food images.

## Validation

- TypeScript check and Vite production build.
- More than 40 automated React component interaction assertions with controlled responses: navigation, tracker, profile validation, section selection, API payloads, preferences, meal views, filtering, shopping selection, download/print handlers, report generation, chat acceptance, history/rollback, stale-result invalidation, failure states, import failure, help and reset.
- Browser preview inspected for desktop profile and overview layouts; local tracker/navigation and unavailable-service feedback exercised.
- Browser automation became unavailable during deeper workflow testing; remaining controls were exercised through component tests, not a complete browser run.
- Real clinical model inference was not run. Controlled test fixtures are not shipped in this frontend.

No hosted deployment was made. This is a local replacement frontend for your existing application.
