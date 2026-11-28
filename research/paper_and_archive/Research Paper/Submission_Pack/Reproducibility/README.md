# BioNutriGPT reproduction notes

## Scope of this snapshot

`source_code/` is a curated copy of the current local implementation, not a clean published Git commit or a complete standalone application installer. `artifact_manifest.json` records the configured remote and base commit separately from source-file hashes. The frontend and raw source datasets are not included. Run the web UI from the original project and its existing frontend dependency files.

`environment_observed.json` describes the research Python environment observed during packaging, not the separate bundled Python environment used to generate Word files. The observed requirements are not claimed to be a portable lockfile. In particular, PyTorch/CUDA and bitsandbytes installation must match the platform and available GPU; do not assume that installing a CUDA-tagged package from the default index will work.

## Local verification

From the original Models/Less Data project directory:

```powershell
python -B -m unittest verify_stage4_chat_repairs verify_stage3_production -v
```

The documented repaired implementation passed 24 checks. These are software tests, not clinical cases. Tests and real-catalog results must be rechecked whenever source files, the food catalog or preprocessing change.

From `source_code/`, the synthetic chatbot regression suite can be run without copying source datasets or model weights:

```powershell
python -B -m unittest verify_stage4_chat_repairs -v
```

It uses temporary session directories. A real-catalog production test additionally requires the catalog, checkpoint and preprocessing artifacts in the expected relative locations; use the paths and hashes in the artifact manifest. Some analyses also require the NHANES development/holdout CSVs and original survey metadata. This package does not claim to recreate those source datasets from a single command.

## Do not confuse these operations

- Reading saved JSON reports does not rerun an experiment.
- A current successful software test does not update an old benchmark's denominator or catalog version.
- Older optimizer/conversation scripts describe historical designs. The completed v4 study has its own frozen inputs and measured configuration; additional experiments need separate frozen configurations.
- The assigned-count 100-case report is deliberately excluded as measured evidence.

`study_config.json` records current versions and pending work. Its `executed` flag is false for the proposed current-system comparison. No measured performance is filled into that pending study.

## Access and sharing

The food catalog, model weights, preprocessing object, full NHANES tables and original food workbooks are referenced, not bundled. Patient plan-session files are excluded. Obtain source datasets through their official providers and check their terms before public redistribution. Do not add restricted DHS/NFHS participant data or private user sessions to a release. A source-code license and third-party data terms still need author review.

## Documentation regeneration

The working research folder contains `prepare_submission_data.py`, `create_submission_figures.py`, `build_submission_documents.py`, and `export_word_pdfs.ps1`. These prepare artifacts rather than run scientific model benchmarks. `prepare_research_evidence.py` in the project root regenerates the descriptive evidence snapshot.

## Final v4 benchmark evidence

The checkpointed v4 run is completed and audited. Eight benchmark-correction tests passed before execution. The separate application test history still has a 10/11 rerun with an unresolved declaration/question expectation.

The pack contains the verified aggregate JSON/Markdown, pre-run provenance, measured code and an evidence snapshot. Full result JSONs, frozen LASI participant scenarios and per-turn journals are not bundled because they contain participant-level records. Local authors can run verify_stage4_v4_results.py against those retained artifacts in the original project. The benchmark-correction suite also expects the local historical/frozen fixtures; those are not recreated from the public aggregate bundle.

The planned study_config.json now describes remaining matched optimizer and broader dialogue work, not the completed v4 study. Raw assigned-count 100-case LLM summaries remain excluded as measured evidence.
