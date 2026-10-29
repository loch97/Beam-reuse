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
- Otherwise a greedy growth runs from several seeds: each grant's best-SNR candidate, its best zero-cost candidate and its cheapest feasible candidate. The best single grant is therefore always a seed.
  - A grant is added only if the combined strongest-cell SNR strictly rises and P_D does not fall.
  - A feasible addition wins at the least total cost. Otherwise zero-cost candidates rank by SNR gain and costed ones by gain per bit.
  - Feasible sets are pruned, most expensive part first.
  - The cheapest feasible set over all seeds is returned. If there is none, the highest-SNR set is returned, which is never worse than the best single grant.
  - `info["steps_snr_db"]` traces the combined SNR after each addition.
- If no set is feasible, `on_unmet` decides: `"dedicated"`, `"reuse_pd"` (single-grant), or `"best_quality"` (the greedy set, flagged unmet).

Result: `mode="reuse_multi"`, `parts` = [{grant, ue_id, tag, weights, n_re, bandwidth, loss_db, mcs, dse, cost_bits}], with combined `pd_pred`, `pred_cell_snr_db`, `exp_quality_cells`, `requirement_met`, summed `cost_bits` and `n_re`, and `weights = grant = None`. `rcs_look` sums the parts' unit SNRs. The closed-loop simulator does not execute multi-grant looks.

RT execution note: the local RT detects each part separately and pools the points (detection-level OR). This is a different detector from the model's energy addition: detection probability, false-alarm rate and point-count distribution all differ, with no general ordering. Several parts can also detect the same scatterer and inflate the point count. Multi-grant P_D therefore needs its own calibration.

Two limits seen in testing:
- Adding a grant also adds its UE's leakage. One grant lowered the combined strongest-cell SNR from 48.4 to 42.3 dB, so grants must be chosen, not just added.
- Under pointing uncertainty P_D saturates. Pose hypotheses outside every beam's main lobe are not fixed by more energy (best pair P_D 0.897 when no single grant reached 0.9). Multi-grant mainly raises measurement quality (SNR), not P_D.

#### Third-round integration notes

**Version pin.** The stable integration point is commit `d786264`: the code of `8fd8a63` plus a README correction. A tag `isac-frontend-v1` was created for it, but this session cannot push tags (branch pushes only), so pin the commit hash, or create the tag on that hash from a machine with push rights. Later changes are additive: fields are only added and defaults reproduce the paper pipeline. A breaking change would be stated in its commit message.

**Q1 — P_D versus SNR offset.** Every decision now carries `BeamDecision.pd_offset_curve = (offsets_db, pd)`, with offsets −20…+20 dB in 1 dB steps. Each value is the decision's own event recomputed with an SNR offset added to every target return:
- any scatterer cell detected;
- averaged over pose, RCS and the site's calibration nodes;
- including UE leakage, per grant for multi-grant looks.

At offset 0 it equals `pd_pred` exactly. Fit RT detections to this curve directly: the Bernoulli likelihood of the observed detection at offset Δ. The fitted Δ then refers to the same event, rather than a Swerling curve applied to a summary SNR. `BeamFrontEnd.offset_curve(W, pred, n_re, bw, clutter, sigma, weights, offsets)` computes it for any beam.

**Q2 — frame-level assignment.** `BeamFrontEnd.plan_frame(tracks, grants, rcs=None, t=None, requirements=None, on_unmet=None)` plans the tracks greedily in the given order, so put confirmed or most-hit tracks first. A grant carries one executed beam. A later track first tries to share a grant already taken, with that same beam. It shares if the beam gives it P_D ≥ τ and meets its requirement; the decision then has `info["shared_with"]` = the owner's index and `cost_bits` = 0, because the cost is counted once. Otherwise it plans over the grants not yet taken. A grant is never used with two different beams, and `grant` indexes the full list. This is greedy, not a joint optimum.

**Q3 — not done yet.** A P_D model for per-part detection with pooled points is not implemented. The local plan of keeping multi-grant out of the main experiment and calibrating it through Q1 is the safer route for now.

#### Fourth-round integration notes (coverage-aware selection, opt-in)

All additive and off by default. Single-grant default decisions were compared with `d786264` on 72 fixed scenarios and are identical.

**Diagnostics on every decision** (reuse, dedicated, multi-grant, shared):
- `coverage_db`: the worst-case gain over the predicted box inflated by `coverage_k`·σ (default 1, along each body axis), relative to the beam's own peak.
- `coverage_gain_dbi`: the absolute worst-case gain, M·G plus the element gain.
- `exp_detected_cells`: the expected number of scatterer cells crossing the threshold, averaged over pose and RCS, with mean UE leakage. No quality SNR is needed.

For multi-grant looks the coverage is pooled by energy: Σ n_re·G / Σ n_re·peak at each point, then the worst point. `info["coverage_maxpool_db"]` gives the per-point maximum of G_i/peak_i instead. Helpers: `coverage_points`, `coverage_beams`, `detected_cells`.

**Objectives** for `plan(..., objective=)`, `plan_multi(..., objective=)`, `plan_frame(..., objective=)` and `BeamFrontEnd(objective=)`:

| Objective | Selects | Note |
|---|---|---|
| `"pd"` (default) | Unchanged | — |
| `"coverage"` | Among reuses with P_D ≥ τ that meet the requirement, the largest `coverage_db` | — |
| `"quality"` | The largest `exp_detected_cells` | Combine with the new `SensingRequirement.min_coverage_db`, which also acts as a requirement under `"pd"` |

Differences below 0.01 dB or 0.01 cells count as ties and are settled by cost. Detected cells saturate, so without this a 0.001-cell gain could buy an expensive beam. These objectives do not minimise communication cost; it is reported. `info` gives `objective`, `binding` (which of "tau", "min_coverage_db" or "requirement" removed candidates), `n_admissible`, `n_tau` and `n_feasible`. `plan_multi` with these objectives grows sets on pooled coverage or detected cells, so grants whose beams cover different parts of the box can be combined. Its `info["steps_score"]` traces the score.

In the test scene (car at 62 m, σ ≈ 0.9 m):
- `"coverage"` raised `coverage_db` from −2.8 to −1.1 dB.
- That cost 14 dB of strongest-cell SNR and lowered the expected detected cells from 4.99 to 4.24.

So within one grant, flatter coverage spreads the energy, consistent with the point-count finding.

**Dedicated width.** `BeamFrontEnd(dedicated_covering=True)` makes the dedicated beam choose among centred sub-arrays of width M, M/2, M/4 and M/8 (at least 4 elements), steered at the predicted centre. It keeps the width with the largest absolute worst-case gain over the inflated box: `tag = -100 - width`, `info["dedicated_width"]`.

**Extent prior.** `BeamFrontEnd(extent_prior=f)` plans with `f(track)` instead of `track.extent`. `site.young_track_extent(min_extent=(4.5, 1.9), min_hits=3, class_extents=None)` gives tracks with fewer than `min_hits` hits (the new `TrackReport.hits`; unknown counts as young) at least that box, or at least their class box.

#### Fifth-round notes (carrier-phase fix in the local RT, explicit link budget, speckle)

**What the local RT fix invalidates.** The local mimo_tracking RT synthesis missed the carrier phase e^{−j2πf_cτ}, so diffuse paths added in phase. Every value fitted to that data is invalid:
- `snr_offset_db` 9.8 and `snr_sigma_db` 5.1;
- the CLEAN-Pfa match;
- any fitted `pd_offset_curve`.

None of these were defaults here; `snr_offset_db` and `snr_sigma_db` are inputs and need refitting on the regenerated data. This repository's own RT validation is not affected:
- Target power is an incoherent sum of per-path powers |a|² (`rt_scoring.echo_power`).
- The communication channel applies e^{−j2π(f_c+f)τ} explicitly (`rt_scoring.communication_power`).

**Explicit link budget.** `SiteConfig(tx_power_dbm=, noise_figure_db=, noise_bandwidth_hz=, element_gain_dbi=)` are aliases of `p_ref_dbm`, `nf_rx`, `noise_bw` and `e_max`; give one name or the other, not both.
- TX power is the total conducted power: boresight EIRP = P + 10·log10 M + element gain.
- Per-RE SNR = P / (kT·F·B) times the beam and path terms; integration adds 10·log10 n_re.
- The element gain applies on transmit and receive.

`SiteConfig.fr2_38dbm(**overrides)` is the local preset: 38 dBm, NF 7 dB, 190.08 MHz noise bandwidth and sensing resolution, 5 dBi elements. The paper defaults are unchanged. Changing to the preset shifts every predicted SNR by exactly (38 − 34.86) − 10·log10(190.08/122.88) − (7 − 7) + 2·(5 − 8) = −4.76 dB, which the tests check.

**Speckle.** The extended model already treats each scatterer as Swerling 1, so a cell's power is exponential: fully developed (Rayleigh) speckle. `SiteConfig(speckle_m=m)` makes the cell power Gamma(m) so the fluctuation can be fitted:
- m = 1 (or None): unchanged;
- m > 1: less fluctuation;
- m = ∞: deterministic cell power, i.e. Marcum Q.

`target.cell_pd` evaluates this exactly with a negative-binomial series. It is checked against Rayleigh (1e-10), Marcum Q (1e-16) and Monte Carlo. It is used everywhere in the extended model and in the multi-grant combination. Not covered:
- The paper's point detector, the RCS estimator's likelihood and the closed-loop back end stay Swerling 1.
- The front end also has no notion of dwell-to-dwell speckle correlation; each look is independent.

`pd_offset_curve` now depends on m, so fitting the new per-dwell detections against it, over a few m values, estimates both the offset and m.

Ranges are treated as one continuum; nothing in the front end bins near and far separately.

#### Sixth-round notes (detection calibration v6, local mimo_tracking)

`SiteConfig.mimo_v6(**overrides)` holds the local calibration of 2026-10-29, fitted after the carrier-phase fix on 918 track dwells of the validation scene. It is `fr2_38dbm` with isotropic elements (`element_pattern=False`, as the RT and the fit used; a correction to the round-5 5 dBi element gain, which applies only to physical hardware) plus:
- `pfa = exp(−10^1.3)` ≈ 2.2e-9, the CLEAN 13 dB threshold;
- `snr_offset_db = 19.4` and `snr_sigma_db = 4.3`, the spread Swerling fit (NLL 314.1, against 331.3 for a plain 18.9 dB offset);
- `rcs_prior_by_class` = −1.1 dBsm shifted by the per-class offsets relative to the pooled 18.9 dB: car −5.9, bus +0.1, truck +0.8 dBsm.

The new `SiteConfig.rcs_prior_by_class` and `BeamFrontEnd.default_rcs(track)` use the per-class value when the tracker sends no RCS and `TrackReport.cls` is listed. Cyclists and pedestrians, never detected, fall back to `rcs_prior_dbsm`.

It is a preset, not a default. The offset belongs to that scene, the CLEAN detector and the footprint-matching rule, and the paper pipeline must stay reproducible. Its accuracy is limited in two ways:
- The fit used a single-cell Swerling curve on a summary SNR, not the front end's own event (any cell, pose-averaged). Verify it in closed loop, or refit it against `pd_offset_curve` (available from a8c045c on).
- The spread fit uses the CLEAN Pfa, like the plain fit (confirmed by the local side).

#### Seventh-round notes (sized dedicated fallback, mimo_v6 RCS population, semantics)

**Dedicated RE sizing** (`SiteConfig.dedicated_sizing`, `dedicated_cap_re`; on in `mimo_v6`, off by default). The dedicated fallback takes the smallest step of the ladder `ded_prb x 12 x ded_sym x 2^k` (432, 864, ..., 27 648 REs) followed by the cap (default 32 768 = 512 subcarriers x 64 slow-time samples, one CPI) at which the best dedicated beam reaches P_D >= tau. If the cap misses tau, the cap is used and `info["dedicated_cap_hit"]` is True. `info` logs `dedicated_n_re`, `dedicated_step` (index into `dedicated_ladder`) and the ladder itself; `BeamDecision.n_re` is the chosen size and `cost_bits = n_re x SE_ref` (mean SE of the grants' MCS), in the same bit units as the reuse cost `n_re x dSE`. The ladder doubles slow time at the fixed dedicated PRB width, so the bandwidth and range resolution do not change; with `sensing_bandwidth_hz` set (as in `mimo_v6`) resolution is fixed anyway. A dedicated look carries no UE clutter, so energy scales with `n_re` and each step is evaluated as an RCS shift of `10 log10(n_k / 432)` dB on the same detector; a test checks this against a fixed look of the same size. `rcs_look` and the closed-loop back end read `d.n_re`, so the evidence uses the chosen size. Turning this on in `mimo_v6` changes that preset's dedicated decisions (more REs, higher P_D); `mimo_v6(dedicated_sizing=False)` restores round 6. Default decisions are unchanged (72 scenarios, identical to d786264).

The comparison is at equal P_D only in the sense that dedicated is sized to just meet tau. Reuse that meets tau still always wins in `plan`; dedicated is the fallback (`on_unmet="dedicated"`), not a competitor priced against reuse.

*plan_multi RE accounting.* It is the same: each part uses its grant's own REs (`n_prb x 12 x n_sym`, fixed by the scheduler), energies add across parts, the part cost is `n_re x dSE` in bits, and `cost_bits` is the sum. Reuse cannot take more REs than a grant has; more energy comes from more grants (`max_grants`). When `plan_multi` misses tau too, its dedicated fallback is the sized one above.

**`RcsPopulation.mimo_v6(frame_dt=0.1)`.** Means car −5.9, bus +0.1, truck +0.8 dBsm (as `rcs_prior_by_class`). sd 2.7 dB per class: the paper calibration's between-target spread (tau = 2.45 dB) plus about 1 dB of error in each class offset; the pooled D std of 0.6 dB applies to the pooled offset, and per-class offsets come from 163–370 dwells. omega = 1.7 dB and rho_frame = 0.333^(dt/1.2) = 0.912 at 0.1 s, the paper calibration's within-target values (lag-1 correlation 0.333 over 1.2 s) carried to your frame interval as an AR(1). Weights bus 370 : truck 344 : car 163. Cyclists and pedestrians are left out of the mixture: they were never detected, so nothing is known about their RCS, and a low mode in the unknown-class mixture would pull every vehicle's design RCS down and push dedicated looks to the cap. Plan a known VRU with its own prior.

**Semantics.**
1. *Designated cell.* `evidence_cell="predicted"` with the cell at `(u_hat, r_hat)` matches `rcs_look`: the unit SNR is `focus_snr` of all visible box scatterers into that fixed cell over the posterior pose, so a diffuse body is handled. `rcs_stat` = cell power / (image noise x noise_scale) and `rcs_floor_db = 0` are on the right scale (eta = −ln Pfa on the same scale). The model has no Doppler dimension: it assumes the target's energy falls in the bin you read. If the predicted radial-velocity error exceeds about half a Doppler bin, straddle loss biases the RCS low. Check sigma_vr against the bin width; if it is larger, a widened look (sum of the ±1 bins) needs a threshold for that sum, so tell me before you do it.
2. *Offset and spread.* Keep `snr_offset_db = 19.4`: it is the detector's excess loss, removed from the floor in `rcs_look`, so the posterior is calibrated RCS. Reduce `snr_sigma_db` once planning uses the estimator: the 4.3 dB spread already contains target-to-target and aspect variation that the predictive RCS nodes now carry. With the prior predictive sd of sqrt(2.7² + 1.7²) = 3.2 dB, the residual is sqrt(4.3² − 3.2²) ≈ 2.9 dB: use `SiteConfig.mimo_v6(snr_sigma_db=2.9)` and refit it in closed loop (`pd_offset_curve`). The estimator's likelihood does not use `snr_sigma_db`, so a residual detector spread is absorbed into the aspect term omega.
3. *eta.* Yes. `rcs_look` is wrapped by `_in_site`, so `V.P["pfa"]` is the site's Pfa there and eta = 10^1.3 ≈ 20 with `mimo_v6`. Outside a front-end call `V.P` holds the paper value again.

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
