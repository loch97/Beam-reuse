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

### Extended target in the paper pipeline and RT check

`python -m isac.simulate --target-model extended` runs the paper conditions with the multi-scatterer box in place of the point detector (`target.ExtendedDetector`), paired by seed block with the point model, and writes to `outputs/simulation_extended/`. The paper's effective RCS prior is kept: the dominant scatterer is lowered by the box's own contribution at the full-band reference cell (0–1.3 dB per frame), so the two models agree at that cell.

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
