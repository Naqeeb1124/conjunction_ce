# conjunction_ce

A project on civil space traffic coordination: when two satellite operators have a predicted close approach, what are a neutral coordinator's maneuver recommendations worth if the operators have to learn the rule behind them?

| File | What it is |
|---|---|
| `conjce/` | The code: `pc.py` (collision probability), `maneuver.py` (burn to miss distance), `model.py` (scenario, costs, sampled true games), `games.py` (Nash equilibria, obedience constraints, the coordinator's linear programs), `inference.py` (operators who learn the rule from past events: simulation and exact enumeration), `scenario.py` (scenario-approach sample sizes), `experiments.py` (all experiments), `exact_checks.py` (sampling-free cross-checks of the learning results), `figures.py`, `batch.py` (many conjunctions at once: batch generator, the three formulations, burden-sharing, carry-over counts), `batch_figures.py` (the scale-up figure) |
| `run_all.py` | Runs every experiment and writes `results/results.json`; with `--figures` it also draws five figures into `figures/` |
| `run_scaleup.py` | Runs the many-conjunctions batch with its checks and writes `results/scaleup.json`; with `--figures` it also draws `figures/scaleup.png` |
| `export_matlab.py` | Writes `matlab/params.json` (every model parameter) and `matlab/expected.json` (the numbers the MATLAB code has to reproduce) from the Python package, so that no parameter is typed twice. Needs `results/results.json`, so run `run_all.py` first |
| `matlab/` | A MATLAB layer on top of the Python package: a port of the core model, a two-body check of the linear burn-to-miss rule, a Simulink model of one conjunction event (`simulink/conjunction_sim.slx`), and four animations. It has its own `README.md` |

## Run it

Python 3.11:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_all.py --figures
.venv\Scripts\python run_scaleup.py --figures
```

The full run takes about 190 s and `run_scaleup.py` about 15 s; if one of the scale-up checks fails it stops with an error and writes no results file. `.venv\Scripts\python -m conjce.figures` and `.venv\Scripts\python -m conjce.batch_figures` redraw the figures from the stored results.

All random seeds are fixed, so every run gives the same results, apart from wall-clock times.

The MATLAB layer needs MATLAB with the Optimization Toolbox, and Simulink for the model; see `matlab/README.md`.
