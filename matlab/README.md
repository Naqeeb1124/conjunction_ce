# MATLAB layer

This folder adds a MATLAB layer to the Python package `conjce` in the repository root. It

1. ports the core model (collision probability, burn-to-miss rule, cost tables, the coordinator's linear program) and checks the port against the Python numbers;
2. checks the linear burn-to-miss rule against a two-body propagation of both satellites;
3. holds a Simulink model of one conjunction event (`simulink/conjunction_sim.slx`);
4. renders four animations.

![Gallery of twelve conjunctions](figures/gallery.png)

## Requirements

- MATLAB with the Optimization Toolbox (`linprog`).
- Simulink, to open the model in `simulink/`.

Everything here was run with MATLAB R2024b (24.2) and Simulink 24.2 on Windows.

## Where the numbers come from

`export_matlab.py` in the repository root writes two files from the Python package:

| File | Content |
|---|---|
| `params.json` | Every model parameter: constants, the baseline conjunction, the settings of the two-body check, the twelve gallery conjunctions |
| `expected.json` | Reference numbers computed in Python: collision probability per joint action, cost tables, the coordinator's rules, the mixed equilibrium, a two-body reference, the learning curves |

MATLAB reads its parameters from `params.json`, so no parameter is typed twice. The tests compare the MATLAB port with `expected.json`.

## Layout

| Path | What it is |
|---|---|
| `core/` | Port of the core model: `pc2d.m` (collision probability), `miss_shift.m` (linear burn-to-miss rule), `cost_matrices.m` (cost tables of the game), `solve_ce.m` (the coordinator's linear program, with `linprog`), `obedience_system.m`, `mixed_nash.m`, and the loaders of the two JSON files |
| `twobody/` | Two-body check: states at the encounter, two propagators (`ode113` and a universal-variable Kepler solver), closest approach, the second-order miss formula, a Monte Carlo estimate of the collision probability |
| `simulink/` | `conjunction_sim.slx`, the Simulink block diagram of one conjunction event |
| `viz/` | One style file (`viz_style.m`), helpers, and one `make_*_animation.m` per animation |
| `tests/` | The checks; each prints one PASS or FAIL line |
| `results/` | Numbers written by the runs, as JSON |
| `figures/` | The animations (MP4, GIF, final-frame PNG) |
| `run_tests.m`, `run_twobody_check.m`, `run_visuals.m`, `setup_paths.m` | Entry points |

## Run

From the repository root:

```
.venv\Scripts\python export_matlab.py
matlab -batch "run_twobody_check" -sd matlab
matlab -batch "run_visuals" -sd matlab
matlab -batch "run_tests" -sd matlab
```

| Step | What it does |
|---|---|
| `export_matlab.py` | Writes `params.json` and `expected.json` from the Python package |
| `run_twobody_check` | Propagates the nine joint actions, recomputes probabilities and incentives, runs the Monte Carlo check, writes `results/twobody_check.json` |
| `run_visuals` | Renders the four animations and writes `results/viz_*.json` |
| `run_tests` | Runs every check and ends with `SUMMARY: <n> passed, <m> failed`; in batch mode the exit code is 1 if a check failed |

`run_tests('test_core.m')` runs one test file. The animation tests read `results/viz_*.json` and the files in `figures/`, so render first after changing an animation.

## Animations

| Animation | Files in `figures/` | What it shows |
|---|---|---|
| Encounter plane | `encounter_plane.mp4`, `.gif`, `.png` | The pass point of satellite 2 and its uncertainty ellipse in the encounter plane; each burn moves it away from satellite 1 and the collision probability falls |
| Orbits in 3D | `orbits_3d.mp4`, `.gif`, `.png` | Both orbits through the closest approach, without burns and with the coordinator's most frequent recommendation, and the distance between the satellites |
| Gallery | `gallery.mp4`, `.gif`, `.png` | Twelve conjunction geometries side by side, each with the coordinator's recommendation and its outcome |
| Learning | `learning.mp4`, `.gif`, `.png` | Realised coordinator cost against the number of past events operators have seen, for four kinds of rule |

All four use the style file `viz/viz_style.m`, which holds the colours and type sizes of the Python figures (`conjce/figures.py`).

## Tests

`run_tests` runs 91 checks:

| Test file | Checks | What it compares |
|---|---|---|
| `test_core.m` | 30 | The port against the Python numbers in `expected.json`; `pc2d` against MATLAB's `integral2` |
| `test_twobody.m` | 24 | The two propagators against each other and against the same construction in Python; the propagated miss against the second-order formula; the linear rule; the Monte Carlo estimates; best replies, equilibria and obedience constraints under the propagated probabilities |
| `test_viz_encounter.m`, `test_viz_gallery.m`, `test_viz_learning.m`, `test_viz_orbits.m` | 8, 10, 9, 10 | Each animation exists as MP4, GIF and PNG; its main numbers equal their source |
