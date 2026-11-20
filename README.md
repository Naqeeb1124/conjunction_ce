# conjunction_ce

A project on civil space traffic coordination: when two satellite operators have a predicted close approach, what are a neutral coordinator's maneuver recommendations worth if the operators have to learn the rule behind them?

| File | What it is |
|---|---|
| `figures/` | Six slide-ready PNGs: `encounter_geometry`, `private_vs_public`, `concept_comparison`, `epsilon_tradeoff`, `margin_tradeoff`, `scaleup` |
| `conjce/` | The code: `pc.py` (collision probability), `maneuver.py` (burn to miss distance), `model.py` (scenario, costs, sampled true games), `games.py` (Nash equilibria, obedience constraints, the coordinator's linear programs), `inference.py` (operators who learn the rule from past events: simulation and exact enumeration), `scenario.py` (scenario-approach sample sizes), `experiments.py` (all experiments), `exact_checks.py` (sampling-free cross-checks of the learning results), `figures.py`, `batch.py` (many conjunctions at once: batch generator, the three formulations, burden-sharing, carry-over counts), `batch_figures.py` (the figure `scaleup.png`) |
| `run_all.py` | Runs every experiment and writes `results/results.json`; with `--figures` it also draws the first five figures |
| `run_scaleup.py` | Runs the many-conjunctions batch with its checks and writes `results/scaleup.json`; with `--figures` it also draws `figures/scaleup.png` |
| `results/results.json` | The results of `run_all.py` |
| `results/scaleup.json` | The results of `run_scaleup.py` |
| `matlab/` | A MATLAB layer on top of the Python package: a port of the core model, a two-body check of the linear burn-to-miss rule, a Simulink model of one conjunction event (`simulink/conjunction_sim.slx`), and four animations. It has its own `README.md` and its own checks (`run_tests.m`) |
| `export_matlab.py` | Writes `matlab/params.json` (every model parameter) and `matlab/expected.json` (the numbers the MATLAB code has to reproduce) from the Python package, so that no parameter is typed twice |

## Run it

The project has its own virtual environment in `.venv` (Python 3.11).

```
.venv\Scripts\python run_all.py --figures
.venv\Scripts\python run_scaleup.py --figures
```

The full run takes about 190 s. `.venv\Scripts\python -m conjce.figures` redraws the first five figures from the stored results. `run_scaleup.py` takes about 15 s; if one of its checks fails it stops with an error and writes no results file. `.venv\Scripts\python -m conjce.batch_figures` redraws the sixth figure from `results/scaleup.json`.

To rebuild the environment elsewhere:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

All random seeds are fixed, so every run reproduces the stored results exactly, apart from wall-clock times.

The MATLAB layer has its own commands and its own 91 checks; see `matlab/README.md`. It needs MATLAB with the Optimization Toolbox, and Simulink for one model, and was run with R2024b.
