# Archived experiment data

- `model_parameters.json`: the conference model defaults. A simulation condition modifies its uncertainty level and UE-information mode explicitly.
- `scene_frames.json`: the 12 physical frames used in the simulation, with target positions and geometry.
- `sector_codebook.{json,npz}`: accepted phase-only broadened beams. Candidate ordering is preserved.
- `mcs_thresholds.json`: the fitted MCS thresholds used by the paper.
- `illustration.npz`, `illustration_row.json`, `fig1_example.json`: the fixed illustrative case for Figure 1. Replotting does not rerun candidate search.
- `regime_masks.npz`: Figure 3's deterministic feasibility grids.
- `reference/main_calibration.json`: effective-RCS records for the simulation.
- `reference/rt_calibration.json`: the independent RT training prior.
- `reference/resource_summary.json`, `reference/robustness_summary.json`: complete paper simulation summaries, including uncertainty intervals.
- `propagation.json`: RT material inventory and solver settings.
- `rt/protocol.json`: fixed randomized trajectories, materials and prediction draws.
- `rt/reference/`: independently generated reference focuses, retained as explicit inputs to validation.
- `rt/all_records_n16000000.json`: complete boundary and family-action scores. Use `selector == "boundary"` for Figure 5(c).
- `rt/summary_n16000000.json`: the full descriptive RT summary.
- `rt/failure_precision_followup.csv`: targeted 64-million-ray follow-up of the failures.
- `example_rt/`: one scene's 4M control paths, 16M evaluation paths and expected scores, for numerical replay checks.
- `expected_*.json`: one-frame outputs from the original simulation implementation, used only by regression tests.

The reference summaries are frozen evidence. Fresh outputs are written under `outputs/`; commands do not overwrite these reference files. The input priors and reference focuses are provided rather than regenerated from the much larger training path collection.
