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
- **Tracker** (`tracker.py`). This is a constant-velocity EKF in scene coordinates with the paper's road-axis white-acceleration noise. A detected cell is associated with the nearest predicted scatterer. Measurement noise is the angle and delay CRB at the *measured* SNR and the *granted* bandwidth. `RcsEstimator` turns detection SNRs into a lower-confidence RCS design value and falls back to the calibration prior until enough looks exist. `steady_state_polar` reproduces `model.track_covariance` as a special case.
- **Front end** (`frontend.py`, messages in `interfaces.py`). It takes a `TrackReport`, the scheduler's `GrantReport`s and an `RcsReport`. It predicts the track to each grant's actual time and screens candidates against the grant's measured SINR and MCS budget. It bounds UE loss over the SRS direction estimate, or over the grid-beam cell when no estimate exists. It then predicts P_D with the extended or point model and returns a `BeamDecision`: the cheapest sequential-policy reuse, or else a dedicated beam. The sensing back end answers with a `SensingReport` for the tracker.
- **Closed loop** (`closed_loop.py`). A simulated scheduler and sensing back end run the loop tracker → front end → back end → tracker on a vehicle approaching the corner RSU:

```bash
python -m isac.closed_loop --epochs 80 --period 4 --target-model extended --cls bus
```

Settings are in `data/frontend_parameters.json`. Logs go to `outputs/closed_loop/`.

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
| `tracker.py` | CV EKF, CRB measurement noise and RCS estimator |
| `interfaces.py`, `frontend.py` | Tracker/scheduler/sensing messages and beam decision |
| `closed_loop.py` | Simulated scheduler and sensing back end in a closed loop |

`data/` contains the compact scientific inputs, complete plotting summaries and regression fixtures. `assets/` contains the scene meshes, paper fonts and static scene illustration. `paper/` contains the current TeX sources and figure PDFs. Large full-batch path caches and historical experiment scripts are excluded.
