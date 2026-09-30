#!/bin/bash
# Reproduce all results of the paper.
#
# Usage:
#   ./run_all.sh              all simulations and plots (serial)
#   NP=12 ./run_all.sh        same, with 12 MPI processes per simulation
#
# The timing runs (Section 4.4) are not included; the commands are at the end
# of this script, commented out.
#
# Run inside the Firedrake virtual environment.

set -e
cd "$(dirname "$0")"

NP=${NP:-1}
RUN="mpiexec -n $NP python"
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg   # save figures without opening plot windows

echo "=== Williamson 5 ==="
$RUN williamson5.py --ref_level=5 --dt=1200 --n_adv_cycles=1 --dumpt=21600
$RUN williamson5.py --ref_level=5 --dt=2400 --n_adv_cycles=2 --dumpt=21600
$RUN williamson5.py --ref_level=6 --dt=600  --n_adv_cycles=1 --dumpt=21600
$RUN williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 --dumpt=21600 --latlon

echo "=== Galewsky ==="
$RUN galewsky.py --ref_level=5 --dt=550  --n_adv_cycles=1
$RUN galewsky.py --ref_level=5 --dt=1100 --n_adv_cycles=2
$RUN galewsky.py --ref_level=6 --dt=275  --n_adv_cycles=1
$RUN galewsky.py --ref_level=6 --dt=550  --n_adv_cycles=2

echo "=== Lauter convergence ==="
$RUN lauter.py --ref_levels 3 4 5 6 --dts 400 200 100 50

echo "=== Plots ==="
python plot_stability.py --test=williamson5
python plot_stability.py --test=galewsky
python plot_convergence.py

echo "Done. Figures are in figures/, simulation output in output/."

# Timings (Section 4.4): Williamson 5, level 6, dt = 1200 s, 2 advection cycles,
# on 6, 12 and 15 cores. Not run by this script; the results depend on the machine.
# The separate output directory keeps these runs from overwriting the results above.
#
# for n in 6 12 15; do
#     out=output/timings/${n}cores
#     mkdir -p $out
#     mpiexec -n $n python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2 \
#         --dumpt=86400 --output=$out -log_view :$out/W5_${n}cores.txt:ascii_flamegraph
# done
