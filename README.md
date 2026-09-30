# Projection-based time stepping for the shallow-water equations on the sphere

Code for the numerical results (Section 4) of the paper. It uses the
(BDM2, DG1) finite element pair on icosahedral sphere meshes, implemented in
[Firedrake](https://www.firedrakeproject.org).

## Contents

| File | Purpose |
|---|---|
| `swe_projection/stepper.py` | The time-stepping scheme (`SWEWithProjection`) and diagnostics |
| `swe_projection/galewsky_utils.py` | Galewsky initial conditions |
| `swe_projection/lonlat_plot_utils.py` | Interpolation onto a lon-lat grid (for `--latlon` output) |
| `swe_projection/output.py` | Writing the diagnostics files (on MPI rank 0 only) |
| `swe_projection/common.py` | Shared physical constants and the icosahedral sphere mesh |
| `williamson5.py` | Williamson 5 test (zonal flow over a mountain) |
| `galewsky.py` | Galewsky barotropic instability test |
| `lauter.py` | Lauter unsteady solid-body rotation, convergence study |
| `plot_stability.py` | Energy, enstrophy and divergence plots (W5 or Galewsky) |
| `plot_convergence.py` | Convergence plot (Lauter) |

## Installation

1. Install Firedrake following <https://www.firedrakeproject.org/install.html>
   (tested with Firedrake 2026.4.2, PETSc 3.25.5, Python 3.14).
2. In the Firedrake virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

## Running a test

Each test script takes its parameters as command-line options, e.g.

```bash
python williamson5.py --ref_level=6 --dt=600 --n_adv_cycles=1
python williamson5.py --help     # list all options and their defaults
```

Runs can be parallelised with MPI:

```bash
mpiexec -n 12 python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2
```

Output is written to `output/<test>/L<ref_level>_dt<dt>_N<n_adv_cycles>/`
(use `--output=<dir>` to replace `output/` by another directory):

* `.pvd`/`.vtu` files for ParaView, every `--dumpt` seconds (default: 4 times a
  day for Williamson 5, hourly for Galewsky). With `--latlon`, the fields are
  also written on a lon-lat grid.
* `diagnostics.txt`, with the columns `t energy enstrophy div_rms` at the same
  output times. `div_rms` is the root-mean-square divergence
  ‖∇·u‖/|Ω|^½, from which R_div in the paper is computed.

For the Lauter test, the normalised errors are written to
`output/lauter/errors_L<ref_level>.txt`, with the columns `t e_u e_D`
(e_D is the error in D + b). Figures are written to `figures/`.

PETSc prints a warning at the end that options like `--dt` were not used.
This is harmless: those options are read by the script, not by PETSc.

## Reproducing the paper results

All simulations and plots below, except the timing runs, can be run in one go with

```bash
./run_all.sh              # serial
NP=12 ./run_all.sh        # 12 MPI processes per simulation
```

The individual commands are:

### Williamson 5 (Section 4.1)

```bash
python williamson5.py --ref_level=5 --dt=1200 --n_adv_cycles=1 --dumpt=21600
python williamson5.py --ref_level=5 --dt=2400 --n_adv_cycles=2 --dumpt=21600
python williamson5.py --ref_level=6 --dt=600  --n_adv_cycles=1 --dumpt=21600
python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 --dumpt=21600 --latlon
python plot_stability.py --test=williamson5
```

* Energy, enstrophy and divergence figures: `figures/Williamson5/stability/plot_{1,2,3}.pdf`
* Depth and vorticity at t = 15 d: made in ParaView from
  `output/williamson5/L6_dt1200_N2/Williamson5_ll.pvd` (last time step)

### Galewsky (Section 4.2)

```bash
python galewsky.py --ref_level=5 --dt=550  --n_adv_cycles=1
python galewsky.py --ref_level=5 --dt=1100 --n_adv_cycles=2
python galewsky.py --ref_level=6 --dt=275  --n_adv_cycles=1
python galewsky.py --ref_level=6 --dt=550  --n_adv_cycles=2
python plot_stability.py --test=galewsky
```

* Energy, enstrophy and divergence figures: `figures/Galewsky/stability/plot_{1,2,3}.pdf`
* Depth perturbation at t = 2 h and 6 h (field `delta D`), and vorticity at
  t = 144 h: made in ParaView from `output/galewsky/L6_dt550_N2/Galewsky.pvd`
  (output is hourly)

### Convergence, Lauter test (Section 4.3)

```bash
python lauter.py      # levels 3-6 with dt = 400, 200, 100, 50 s, T = 1 day
python plot_convergence.py
```

* Figure: `figures/Lauter/convergence-plot.pdf`

### Timings (Section 4.4)

Williamson 5 at level 6, dt = 1200 s, 2 advection cycles, output once a day,
on 6, 12 and 15 cores. These are not part of `run_all.sh` (the timings depend on
the machine), but the commands are at the end of it, commented out:

```bash
mkdir -p output/timings/6cores output/timings/12cores output/timings/15cores
mpiexec -n 6  python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 --dumpt=86400 \
    --output=output/timings/6cores  -log_view :output/timings/6cores/W5_6cores.txt:ascii_flamegraph
mpiexec -n 12 python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 --dumpt=86400 \
    --output=output/timings/12cores -log_view :output/timings/12cores/W5_12cores.txt:ascii_flamegraph
mpiexec -n 15 python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 --dumpt=86400 \
    --output=output/timings/15cores -log_view :output/timings/15cores/W5_15cores.txt:ascii_flamegraph
```

The separate `--output` directory keeps the timing runs from overwriting the
Williamson 5 results used for the stability plots. The table uses the logged events `projection_step`, `advection_step` and
`second_order_1st_step` (the explicit step), summed over the log.
