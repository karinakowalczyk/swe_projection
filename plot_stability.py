"""Energy, enstrophy and divergence plots for the Williamson 5 or Galewsky test.

Reads output/<test>/L<level>_dt<dt>_N<N>/diagnostics.txt for the configurations
run by run_all.sh and writes figures/<Test>/stability/plot_{1,2,3}.pdf. The
figures are made at 0.9 times the text width; include them with
\\includegraphics[width=0.9\\textwidth]{...} so that their fonts match the text.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swe_projection.plot_style import COLORS, add_style_args, setup

parser = argparse.ArgumentParser(description="Energy, enstrophy and divergence plots (Williamson 5 or Galewsky)")
parser.add_argument("--test", choices=["williamson5", "galewsky"], required=True)
add_style_args(parser)
args, _ = parser.parse_known_args()
TEXTWIDTH = setup(args)

w5 = args.test == "williamson5"
if w5:
    # refinement level: (dt without subcycling, dt with 2 subcycles)
    config = {5: (1200., 2400.), 6: (600., 1200.)}
    base, time_unit, unit, outdir = Path("output/williamson5"), 86400., "days", "Williamson5"
else:
    config = {5: (550., 1100.), 6: (275., 550.)}
    base, time_unit, unit, outdir = Path("output/galewsky"), 3600., "hours", "Galewsky"


def load(path, column):
    """One column (t, energy, enstrophy or div_rms) of a diagnostics.txt file."""
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    return np.loadtxt(path, ndmin=2)[:, names.index(column)]


# Williamson 5: absolute values on a log scale; Galewsky: signed values
sign = np.abs if w5 else (lambda x: x)
metrics = [  # (column, title, y label, transform)
    ("energy", "Normalised energy error over time",
     r"$|e_E(t)|$" if w5 else r"$e_E(t)$", lambda d: sign((d - d[0])/d[0])),
    ("enstrophy", "Normalised enstrophy error over time",
     r"$|e_Z(t)|$" if w5 else r"$e_Z(t)$", lambda d: sign((d - d[0])/d[0])),
    # div_rms is already divided by |Omega|^(1/2), so R_div is its change
    ("div_rms", "Change in RMS divergence over time",
     r"$|R_{\mathrm{div}}(t)|$ [s$^{-1}$]" if w5 else r"$R_{\mathrm{div}}(t)$ [s$^{-1}$]",
     lambda d: sign(d - d[0])),
]

os.makedirs(f"figures/{outdir}/stability", exist_ok=True)
for i, (column, title, ylabel, transform) in enumerate(metrics, start=1):
    fig, ax = plt.subplots(figsize=(0.9*TEXTWIDTH, 0.55*TEXTWIDTH), layout="constrained")
    plot = ax.semilogy if w5 else ax.plot
    for level, (dt, dt2) in config.items():
        for dti, N, style, marker in ((dt, 1, "-", "o"), (dt2, 2, ":", "s")):
            path = base/f"L{level}_dt{dti:g}_N{N}"/"diagnostics.txt"
            t = load(path, "t")/time_unit
            y = transform(load(path, column))
            label = rf"L{level}, $\Delta t = {dti:g}\,\mathrm{{s}}$"
            if N == 2:
                label += ", 2 subcycles"
            plot(t, y, style, color=COLORS[level], marker=marker, markersize=2.5,
                 markevery=max(1, len(t)//30), label=label)
    ax.set_xlabel(unit)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.02*t[-1])  # small gap on the right, so the last markers are not cut
    if w5:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    # title at the top, legend below the axes, so it never covers the curves
    fig.suptitle(title)
    fig.legend(loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.2)
    fig.savefig(f"figures/{outdir}/stability/plot_{i}.pdf")
    plt.close(fig)

print(f"Figures written to figures/{outdir}/stability/")
