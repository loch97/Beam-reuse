# Scheduled downlink beam reuse for sensing

Research code and compact reference data for **When Can a Scheduled Downlink Beam Also Sense? A Link Margin–Coverage Tradeoff**.

This directory is self-contained. Copy its contents into a new GitHub repository. It does not import the parent tracking project. Generated results go to `outputs/`, which is ignored by Git.

## Setup

Python **3.12** was used for verification. From this directory:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
```

Only fresh path tracing requires the optional RT stack:

```bash
python -m pip install -r requirements-rt.txt
```

## Reproduce the paper figures

```bash
python -m isac.figures
```

PDF and PNG figures are written to `outputs/figures/`. This uses the supplied calibration, numerical masks and complete reference summaries. It does not rerun the simulations or RT.

| Figure | Contents | Input |
|---|---|---|
| 1 | Roadside geometry and gain patterns | Frozen illustrative beam weights and geometry |
| 2 | Evaluation scene | Supplied vector PDF illustration |
| 3 | Feasibility under 4/8 dB UE gain-loss budgets | Deterministic candidate masks |
| 4 | Dedicated sensing demand and communication cost | Complete 64-block simulation summary |
| 5(a,b) | Robustness to prediction mismatch | Complete 64-block stress summary |
| 5(c) | Predicted versus RT detection probability | All 926 RT boundary probes |

Figure 2 is retained as a static illustration. The original camera metadata and raw render are not available in the current paper source tree. Its PDF includes the scene image and vector annotations.

To install regenerated figures into the included manuscript:

```bash
python -m isac.figures --install-paper
cd paper
latexmk -pdf main.tex
```

The manuscript uses the standard `IEEEtran` class. TeX is not needed for plotting.

## Run the simulations

A small run:

```bash
python -m isac.simulate --condition broad_U1 --seeds 1 --frames 1
```

The paper batch comprises six conditions, 64 random seed blocks and 12 physical frames per block:

```bash
python -m isac.simulate --workers 4
python -m isac.summarize
```

`U0` means known UE direction. Under `U1`, direction is inferred from the strongest-beam index. The target covariance settings are `narrow`, `inter` and `broad`. The experiment uses the supplied effective-RCS calibration and the same 12 target frames as the paper. Random seed blocks draw the reports, grants, penetration losses and interference.

The summary uses 20,000 paired resamples of entire seed blocks. It requires the full batch. Partial smoke runs produce per-block files but are not treated as paper results.

## Run RT validation

`data/rt/protocol.json` fixes 48 trajectories, three poses each and two prediction draws per pose. Target vehicles are cars, buses and trucks. Served entities also include a pedestrian geometry proxy. Material draws, reference focuses and training priors are supplied as fixed experiment inputs.

The 4-million-ray paths fix communication states and beam decisions. The 16-million-ray evaluation reuses those decisions:

```bash
python -m isac.rt trace --samples 4000000 --workers 1
python -m isac.rt trace --samples 16000000 --workers 1
python -m isac.rt score --samples 16000000 --workers 1
python -m isac.rt_summary
```

Tracing explicitly defaults to `cuda_ad_mono_polarized`. Select `--variant llvm_ad_mono_polarized` for CPU execution. There is no automatic backend switching or retry. Failures propagate directly. Existing named outputs are replaced when that command is rerun.

Use `--cases t00_f0` to trace or score one scene. The full summary requires all 144 scenes. The archive includes a small paired path-cache example, so RT scoring can be checked without installing Sionna.

Within each of the three beam families, boundary probes are the communication-admissible candidates nearest the 0.9 detection threshold on either side. The `action` records separately score each family's largest predicted detection probability. Figure 5(c) pools **boundary** records, not actions chosen by the sequential policy. Repeated frames and shared candidates are retained in those descriptive counts.

RT supplies physical paths. The detection probability still uses the stated Gaussian path-amplitude and mean-power model. The code does not implement waveform synchronization, a physical detector or target tracking.

To plot newly computed results instead of the reference summaries:

```bash
python -m isac.figures --simulation-results outputs --rt-results outputs/rt
```

## Beam front end with tracking and scheduling

The modules below extend the paper model. They do not change the paper pipeline or its outputs.

- **Multi-scatterer target** (`target.py`). The vehicle is a rigid L × W box. Each visible face contributes its centre and end corners. Every scatterer is an independent Swerling-1 reflector: the dominant face is at the design RCS, corners at −6 dB and other faces at −10 dB. Scatterer powers add per receive cell through the receive-array and finite-subcarrier delay responses. The tracker's position and heading uncertainty is integrated over an exact-mass cell quadrature. All scatterers move rigidly with each pose node. P_D is E[max over scatterer cells], a lower bound on detection in any cell. With one scatterer and no pose spread it reduces exactly to `detection.predictive_pd`.
- **Tracker** (`tracker.py`). This is a constant-velocity EKF in scene coordinates with the paper's road-axis white-acceleration noise. A detected cell is associated with the nearest predicted scatterer. Measurement noise is the angle and delay CRB at the *measured* SNR and the *granted* bandwidth. `steady_state_polar` reproduces `model.track_covariance` as a special case.
- **RCS estimator** (`rcs.py`). This is a hierarchical model: class population → target mean → AR(1) aspect deviation per look → Swerling-1 power. `RcsPopulation.fit` estimates the class means, the between-target spread (shrunk by an inverse-gamma hyperprior, because the calibration has few targets per class), the within-target spread and the frame correlation from the training-site calibration records. `RcsEstimator` is a grid posterior. Its evidence is the *designated cell*, the dominant scatterer's receive cell, which is fixed before the look. The back end reports on every look whether that cell crossed the threshold and its power over the CFAR floor. At the posterior pose, with that floor, a miss contributes 1 − P_d(σ) and a crossing contributes the exponential power density. Equivalently, a crossing is P_d(σ) times the density truncated at the threshold. The strongest crossing cell, which the tracker uses, is not used, because choosing the maximum biases its power upward. Plain averages of detected powers are biased high (+4 dB at P_D ≈ 0.25) because only threshold crossings are kept; the truncated likelihood removes that bias.
- **Front end** (`frontend.py`, messages in `interfaces.py`). It takes a `TrackReport`, the scheduler's `GrantReport`s and an `RcsReport`. It predicts the track to each grant's actual time and screens candidates against the grant's measured SINR and MCS budget. It bounds UE loss over the SRS direction estimate, or over the grid-beam cell when no estimate exists. Its decision variable is the predicted P_D, averaged over pose uncertainty and, by default, over the predictive RCS distribution (`rcs_mode="predictive"`; `"design"` plugs in one lower-quantile RCS). Candidates are evaluated in increasing cost, which selects the same beam as evaluating them all. It returns a `BeamDecision`: the cheapest sequential-policy reuse, or else a dedicated beam. The decision P_D is E_pose[max over cells], a conservative value; the decision also carries the designated cell's crossing curve for logging and as a fallback. The sensing back end answers with a `SensingReport`, and `BeamFrontEnd.rcs_look` turns both into the estimator's evidence.
- **Closed loop** (`closed_loop.py`). A simulated scheduler and sensing back end run the loop tracker → front end → back end → tracker on a vehicle approaching the corner RSU:

```bash
python -m isac.closed_loop --epochs 80 --period 4 --target-model extended --cls bus
python -m isac.closed_loop --truth-offset-db -6 --rcs-mode design   # stress: weak target, plug-in RCS
```

Settings are in `data/frontend_parameters.json`. Logs go to `outputs/closed_loop/`.

### Using the front end at another site (external integration)

Construct `BeamFrontEnd(site=SiteConfig(...))` (`isac/site.py`) instead of patching `model` globals. The site is applied only during each front-end call and restored afterwards. Every `SiteConfig` field defaults to `None`, which keeps the paper value, so the paper pipeline and its tests are unchanged.

| Group | `SiteConfig` fields |
|---|---|
| Geometry | `rsu_xy`, `boresight_rad`, `rsu_h`, `rc_h`, `ue_h`, `cone_angle` |
| Radio | `M`, `M_rx`, `fc`, `p_ref_dbm` or `eirp_dbm`, `nf_rx`, `nf_ue`, `noise_bw` (separate from `df`), `df`, `l_proc`, `inr`, `element_pattern`, `e_max` |
| Detection and calibration | `pfa`, `snr_offset_db`, `snr_sigma_db`, `sigma_hermite_nodes` |
| Scene | `rcs_prior_dbsm`, `interference`, `ue_clutter`, `extra` (any further `model.P` key) |

- **Cone angle:** `cone_angle=True` gives u = cos(elevation)·sin(azimuth − boresight), the angle a horizontal ULA actually sees. `model.beam(u)` then reaches the full array gain.
- **Calibration:** `snr_offset_db` subtracts a fitted model-minus-RT offset from the target SNR. `snr_sigma_db` integrates a log-normal SNR uncertainty into the predicted P_D with Gauss–Hermite nodes.
- **Sector beams:** they require M = 32 (the codebook); with another M use levels without sectors.

Measurement quality: `SensingRequirement(min_cell_snr_db, quality_snr_db, min_cells)` is passed as `TrackReport.requirement` or as `plan(..., requirement=)`. The front end then picks the cheapest candidate with P_D ≥ τ that also meets the requirement. Every `BeamDecision` reports `pred_cell_snr_db` (strongest scatterer-cell SNR over noise plus UE leakage, averaged in dB) and `exp_quality_cells` (expected number of cells at or above `quality_snr_db`). It also reports `requirement_met`, which is None when no requirement was given. `plan(..., return_table=True)` fills `BeamDecision.candidates` with every admissible candidate: grant, ue_id, tag, loss_db, mcs, dse, cost_bits, pd, pred_cell_snr_db, exp_quality_cells, weights.

RCS: `RcsPopulation.from_priors({cls: (mean, sd)}, omega, rho_frame)` sets per-class priors for another scene. `RcsEstimator(pop, cls)` counts misses and uses the threshold-truncated likelihood. `RcsReport` carries the posterior mean, sd and nodes. Its evidence is the designated (dominant-scatterer) cell, so a back end must fill `SensingReport.rcs_detected`, `rcs_stat` and `rcs_floor_db` for that cell on every look; `BeamFrontEnd.rcs_look` turns them into an `RcsLook`. With `snr_offset_db` set, the evidence is offset too, so the posterior is in calibrated RCS.

Robustness and geometry: a box with no face visible from the RSU (a ghost track at or next to the mast) is handled as a point target, flagged as `info["degenerate_box"]`. `site.road_points(origin, heading, lane_offsets, s_range, spacing, kinds)` places served UEs along a road of any direction. `site.ue_link_site(xy, kind, rsu, bore, rng, interference_dbm=None)` gives serving-grid SINR and grid beam without the corridor geometry, with optional external interference.

#### Second-round integration notes

**Resolution bandwidth (R0).** `SiteConfig(sensing_bandwidth_hz=...)` sets the bandwidth of the sensing range processing. It affects range-cell coupling (`delay_power`/`delay_amplitude`) and the delay CRB only. Integration gain still comes from the grant's `n_re`. `BeamDecision.bandwidth` remains the grant bandwidth (n_prb·12·df); the new `BeamDecision.resolution_bw` is the one used for coupling. Unset, the two are equal, as in the paper. At full-band resolution a vehicle's scatterers fall in different range cells, so the strongest cell's SNR and the box P_D are lower than at grant resolution.

**Quality calibration (R1).** `SensingRequirement(quality_logistic=(a, b), min_quality_prob=p)` requires P(quality | pred_cell_snr_db) = 1/(1+exp(−(a + b·snr))) ≥ p. The SNR is the front end's calibrated `pred_cell_snr_db`, offset already removed. Each decision reports `quality_prob`. `exp_quality_cells` counts model scatterer cells, not RT/CLEAN points.

**Unmet requirements (R2).** `plan(..., on_unmet=...)` (also a `BeamFrontEnd` argument) chooses what happens when no reuse meets both P_D ≥ τ and the requirement. Everything is done in one search, reusing cached P_D and cell SNRs, and `info["fallback"]` records the path taken:

| `on_unmet` | Result | `info["fallback"]` |
|---|---|---|
| `"dedicated"` (default) | Dedicated sensing beam | `"dedicated"` |
| `"reuse_pd"` | Cheapest reuse with P_D ≥ τ, identical to planning without the requirement | `"reuse_pd"` |
| `"best_quality"` | Reuse with the largest `pred_cell_snr_db` among those with P_D ≥ τ | `"best_quality"` |

`info["fallback"]` is `"none"` when the requirement is met or absent. `requirement_met` is False for the fallback paths.

**Speed (R3).** Before the full box P_D, `target.pd_upper_bound` gives a rigorous, cheap clutter-free bound per beam:
- the transmit gain is bounded over the pose nodes' direction span via Bernstein's inequality (`gain_upper_bound`);
- range is the minimum over the nodes, element gain the maximum, and coupling at most 1.

Beams whose bound is below τ get P_D = −1 ("screened, below τ") without computing cell SNRs. Cell SNRs are cached per (grant, candidate) and shared by P_D, quality and the fallbacks. Lazy and exhaustive search still pick the same beam.

**RCS evidence semantics (R4).**
1. `rcs_stat` is the designated cell's power divided by the reported noise floor, as a linear value on the same scale as `eta = −ln(pfa)`. Set `SiteConfig(pfa=...)` to the detector's equivalent Pfa; for a 13 dB CLEAN threshold that is exp(−10^1.3), giving η ≈ 19.95.
2. Report `rcs_stat` and `rcs_floor_db` on every look, whether `rcs_detected` is True or False. Misses enter through the event factor.
3. `rcs_floor_db` = 10·log10(floor / thermal noise of the model). It is 0 dB without UE leakage. If the RT image noise is scaled (re_scaling), report the floor relative to the noise that the front end's SNR model assumes.
4. Class strings are free-form but must match between `RcsPopulation.from_priors({...})` and `TrackReport.cls`. `"car"`, `"bus"` and `"truck"` work. An unknown class falls back to the class mixture.

If the designated cell is read at the decision-time predicted dominant position (`BeamDecision.u_hat`, `r_hat`), construct `BeamFrontEnd(..., evidence_cell="predicted")`. The evidence curve and `rcs_look` then model a fixed cell, so the pose error moves the scatterers out of it. This matters at fine resolution: 0.8 m cells against pose errors of about 1 m. The default `"scatterer"` assumes the cell follows the dominant scatterer, as the closed-loop back end does.

**Multi-grant looks (R5).** `BeamFrontEnd.plan_multi(track, grants, rcs, requirement=, max_grants=K, on_unmet=)` builds one look from up to K grants of the frame. Each selected grant illuminates the target with its own candidate beam on its own REs, and the look integrates them in one CPI.

Model:
- Cell SNRs add over grants: SNR = Σ_i SNR_i(W_i, n_re_i). This is energy addition, i.e. maximal-ratio combining over REs at the hypothesised cell, with the transmit beam response known per RE.
- UE leakage is added per grant, averaged over that grant's UE-direction quadrature (an approximation; single-grant `plan` averages P_D over it exactly).
- P_D (any cell, pose- and RCS-averaged) and the quality terms use the combined values. Cost = Σ_i n_re_i·ΔSE_i, each grant within its own MCS budget.
- All grants share one pose prediction (at `t`, default the first grant's time).
- Range coupling uses `sensing_bandwidth_hz`; set it for multi-grant looks. The default, the summed grant bandwidths, is only a placeholder.

Search:
- If the single-grant `plan` already meets P_D ≥ τ and the requirement, that decision is returned, with `parts` of length 1 and `info["multi"] = False`. `max_grants=1` therefore equals `plan`.
- Otherwise grants are added greedily. A choice that makes the set feasible wins at the least total cost; otherwise the candidate with the largest strongest-cell SNR gain per (cost + 1 bit) is added. A final pass drops grants whose removal keeps the set feasible.
- If no set is feasible, `on_unmet` decides: `"dedicated"`, `"reuse_pd"` (single-grant), or `"best_quality"` (the greedy set, flagged unmet).

Result: `mode="reuse_multi"`, `parts` = [{grant, ue_id, tag, weights, n_re, bandwidth, loss_db, mcs, dse, cost_bits}], with combined `pd_pred`, `pred_cell_snr_db`, `exp_quality_cells`, `requirement_met`, summed `cost_bits` and `n_re`, and `weights = grant = None`. `rcs_look` sums the parts' unit SNRs. The closed-loop simulator does not execute multi-grant looks.

Two limits seen in testing:
- Adding a grant also adds its UE's leakage. One grant lowered the combined strongest-cell SNR from 48.4 to 42.3 dB, so grants must be chosen, not just added.
- Under pointing uncertainty P_D saturates. Pose hypotheses outside every beam's main lobe are not fixed by more energy (best pair P_D 0.897 when no single grant reached 0.9). Multi-grant mainly raises measurement quality (SNR), not P_D.

### Extended target in the paper pipeline and RT check

`python -m isac.simulate --target-model extended` runs the paper conditions with the multi-scatterer box in place of the point detector (`target.ExtendedDetector`), paired by seed block with the point model, and writes to `outputs/simulation_extended/`. The paper's effective RCS prior is kept: the dominant scatterer is lowered by the box's own contribution at the full-band reference cell (0–1.3 dB per frame), so the two models agree at that cell.

Full paired batch: `python -m isac.simulate --target-model point --workers 4`, then the same with `--target-model extended`. Then run `python -m isac.summarize --target-model extended` (writes `resource_summary_extended.json` and `robustness_summary_extended.json`) and `python -m isac.compare_models [--dm 3] [--n 5]`. The comparison gives the extended-minus-point difference per condition and policy, with 95% intervals from the same paired seed-block resampling, and the robustness scores under `broad_U1`.

`python -m isac.rt_extended [--paths outputs/rt/paths] [--fit-offsets]` compares the beam dependence of RT target power at the reference focus with both models, over steered, broadened and sector beams and 1/4/32 PRBs. RT paths are split into direct returns and transmit-side multipath (the transmit beam reaches the target over another reflector). The model contains only the direct returns. Vehicle extents come from the RT mesh bounding boxes. On the bundled example scene (car, t00_f0), within 20 dB of the RT peak:

| Comparison | Box model | Point model |
|---|---|---|
| Direct paths, rms error | 0.6–0.7 dB | 2.7–2.9 dB |
| All paths, p95 over-prediction | 2.4–3.4 dB | 6.8–7.4 dB |

Transmit multipath carries up to 98–99% of the power for some off-target beams. Scatterer offsets fitted on one scene are not used; `--fit-offsets` over all 144 traced scenes is the intended calibration.

## Numerical checks

```bash
python -m unittest discover -s tests -v
```

Checks cover the finite-subcarrier filter response, exact grant-selection probabilities, two simulation conditions against the original implementation, and paired RT scoring against the original frozen path cache. The complete paper experiment is not rerun by the tests.

## Code map

| Module | Purpose |
|---|---|
| `model.py` | ULA candidates, link budgets, prediction covariance and RCS statistics |
| `detection.py` | Finite-subcarrier response and uncertainty-averaged detection probability |
| `simulation_engine.py` | Compatibility and selected-action evaluation |
| `scheduling.py` | Exact grant probabilities, communication cost and stress metrics |
| `simulate.py`, `summarize.py` | Simulation batch and paired-block inference |
| `rt_scene.py` | Vehicle/environment assembly and RT path extraction |
| `rt_scoring.py` | Target/clutter power and communication-channel scoring |
| `rt.py`, `rt_summary.py` | Fixed-protocol tracing, scoring and descriptive counts |
| `calibration.py` | Reference-focus calculation and receive-array response |
| `quadrature.py` | Exact response caches and fixed-decision stress scoring |
| `plot_*.py`, `figures.py` | Paper figures with common fonts and column geometry |
| `target.py` | Multi-scatterer box target and pose-uncertainty detection probability |
| `tracker.py` | CV EKF and CRB measurement noise |
| `rcs.py` | Hierarchical RCS prior and truncated-likelihood grid estimator |
| `interfaces.py`, `frontend.py` | Tracker/scheduler/sensing messages and beam decision |
| `closed_loop.py` | Simulated scheduler and sensing back end in a closed loop |
| `rt_extended.py` | RT check of point versus box target geometry |

`data/` contains the compact scientific inputs, complete plotting summaries and regression fixtures. `assets/` contains the scene meshes, paper fonts and static scene illustration. `paper/` contains the current TeX sources and figure PDFs. Large full-batch path caches and historical experiment scripts are excluded.
