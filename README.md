# Projection-based time stepping for the shallow-water equations on the sphere

A time-stepping scheme for the rotating shallow-water equations on the sphere,
implemented in [Firedrake](https://www.firedrakeproject.org). Velocity is
discretised in BDM2 and depth in DG1 on icosahedral sphere meshes.

Each time step is split into an advection step (DG-upwind with SSPRK3, which can
be subcycled) and an implicit projection step for the gravity, Coriolis and
divergence terms. The second-order scheme adds an explicit half step and uses a
first-order predictor to compute the advecting velocity.

## Installation

1. Install Firedrake following <https://www.firedrakeproject.org/install.html>
   (tested with Firedrake 2026.4.2, PETSc 3.25.5, Python 3.14).
2. In the Firedrake virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

## Contents

| File | Purpose |
|---|---|
| `swe_projection/stepper.py` | The time-stepping scheme (`SWEWithProjection`) and diagnostics |
| `swe_projection/common.py` | Physical constants and the icosahedral sphere mesh |
| `swe_projection/galewsky_utils.py` | Galewsky initial conditions |
| `swe_projection/lonlat_plot_utils.py` | Interpolation onto a lon-lat grid |
| `swe_projection/output.py` | Diagnostics files, progress messages and run information |
| `williamson5.py` | Williamson 5 test: zonal flow over an isolated mountain |
| `galewsky.py` | Galewsky barotropic instability test |
| `lauter.py` | Läuter unsteady solid-body rotation: convergence test |
| `plot_stability.py` | Energy, enstrophy and divergence plots (Williamson 5 or Galewsky) |
| `plot_convergence.py` | Convergence plot (Läuter) |
| `run_all.sh` | Runs all test cases and plots |

The references for the test cases are given at the top of each test script.

## Running a test

Each test script takes its parameters as command-line options:

```bash
python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2
python williamson5.py --help     # all options and their defaults
```

The main options are the mesh refinement level `--ref_level`, the time step
`--dt` in seconds and the number of advection subcycles `--n_adv_cycles`.

Runs can be parallelised with MPI:

```bash
mpiexec -n 4 python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2
```

PETSc prints a warning at the end that options like `--dt` were not used. This
is harmless: those options are read by the script, not by PETSc.

## Output

Each run writes to `output/<test>/L<ref_level>_dt<dt>_N<n_adv_cycles>/`
(`--output=<dir>` replaces `output/` by another directory):

* `.pvd`/`.vtu` files with the fields, every `--dumpt` seconds (default: 4 times
  a day for Williamson 5, hourly for Galewsky). With `--latlon`, the fields are
  also written on a lon-lat grid.
* `diagnostics.txt`, with the columns `t energy enstrophy div_rms`, where
  `div_rms` is the root-mean-square divergence.
* `run_info.json`, with the options used (including defaults), the command
  line, the number of MPI processes, the git commit and the Firedrake version.

The Läuter test writes the normalised L2 errors in u and in D + b to
`output/lauter/errors_L<ref_level>.txt`, with the columns `t e_u e_D`.

The plot scripts read this output and write figures to `figures/`:

```bash
python plot_stability.py --test=williamson5    # or --test=galewsky
python plot_convergence.py
```

## Running all tests

`run_all.sh` runs all test cases with the configurations below and then makes
the plots:

```bash
./run_all.sh              # serial
NP=4 ./run_all.sh         # 4 MPI processes per simulation
LATLON=1 ./run_all.sh     # also lon-lat output for all Williamson 5 and Galewsky runs
```

| Test | Level | dt [s] | Advection subcycles | End time |
|---|---|---|---|---|
| Williamson 5 | 5 | 1200, 2400 | 1, 2 | 15 days |
| Williamson 5 | 6 | 600, 1200 | 1, 2 | 15 days |
| Galewsky | 5 | 550, 1100 | 1, 2 | 6 days |
| Galewsky | 6 | 275, 550 | 1, 2 | 6 days |
| Läuter | 3, 4, 5, 6 | 400, 200, 100, 50 | 1 | 1 day |

This takes several hours; the level 6 runs take most of the time. Each run
prints its configuration at the start and its progress at every output time.

The script also contains, commented out, the commands for timing runs with
PETSc's `-log_view`.
