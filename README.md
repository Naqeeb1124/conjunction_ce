# conjunction_ce

A project on civil space traffic coordination: when two satellite operators have a predicted close approach, what are a neutral coordinator's maneuver recommendations worth if the operators have to learn the rule behind them?

Start with **RESULTS.md**. Then:

| File | What it is |
|---|---|
| `RESULTS.md` | The question, the numbers, what is verified, limitations, next steps |
| `DERIVATIONS.md` | Every formula, step by step, with references |
| `NOVELTY.md` | The prior-art checks and how the question changed |
| `figures/` | Six slide-ready PNGs: `encounter_geometry`, `private_vs_public`, `concept_comparison`, `epsilon_tradeoff`, `margin_tradeoff`, `scaleup` |
| `conjce/` | The code: `pc.py` (collision probability), `maneuver.py` (burn to miss distance), `model.py` (scenario, costs, sampled true games), `games.py` (Nash equilibria, obedience constraints, the coordinator's linear programs), `inference.py` (operators who learn the rule from past events: simulation and exact enumeration), `scenario.py` (scenario-approach sample sizes), `experiments.py` (all experiments), `exact_checks.py` (sampling-free cross-checks of the learning results), `figures.py`, `batch.py` (many conjunctions at once: batch generator, the three formulations, burden-sharing, carry-over counts), `batch_figures.py` (the figure `scaleup.png`) |
| `tests/` | 210 tests: `test_pc.py` 45, `test_maneuver.py` 27, `test_games.py` 22, `test_model.py` 26, `test_inference.py` 11, `test_exact.py` 11, `test_scenario.py` 9, `test_experiments.py` 17, `test_batch.py` 42 |
| `run_all.py` | Runs every experiment and writes `results/results.json`; with `--figures` it also draws the first five figures |
| `run_scaleup.py` | Runs the many-conjunctions batch (RESULTS.md 4.10) with its checks and writes `results/scaleup.json`; with `--figures` it also draws `figures/scaleup.png` |
| `results/results.json` | Every number quoted in the documents, except those about many conjunctions |
| `results/scaleup.json` | Every number about many conjunctions: RESULTS.md 4.10 and its summary in section 1, DERIVATIONS.md 7 |

## Run it

The project has its own virtual environment in `.venv` (Python 3.11).

```
.venv\Scripts\python -m pytest
.venv\Scripts\python run_all.py --figures
.venv\Scripts\python run_scaleup.py --figures
```

The full run takes about 190 s. `.venv\Scripts\python -m conjce.figures` redraws the first five figures from the stored results. `run_scaleup.py` takes about 15 s; if one of its checks fails it stops with an error and writes no results file. `.venv\Scripts\python -m conjce.batch_figures` redraws the sixth figure from `results/scaleup.json`.

To rebuild the environment elsewhere:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

All random seeds are fixed, so the numbers in RESULTS.md are reproduced exactly, apart from the wall-clock seconds of section 4.10; two complete runs gave identical results apart from wall-clock times.
