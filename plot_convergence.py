"""Convergence plot for the Lauter test.

Reads the normalised errors written by lauter.py (output/lauter/errors_L<level>.txt)
and plots the errors at the final time against the mesh size h = 2^-level. The
figure is made at 0.95 times the text width; include it with
\\includegraphics[width=0.95\\linewidth]{...} so that its fonts match the text.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from swe_projection.plot_style import COLORS, add_style_args, setup

parser = argparse.ArgumentParser(description="Convergence plot for the Lauter test")
parser.add_argument("--ref_levels", type=int, nargs="+", default=[3, 4, 5, 6], help="mesh refinement levels to plot")
add_style_args(parser)
args, _ = parser.parse_known_args()
TEXTWIDTH = setup(args)
levels = args.ref_levels

# Errors at the final time (last row; columns t, e_u, e_D)
final = np.array([np.loadtxt(f"output/lauter/errors_L{level}.txt", ndmin=2)[-1] for level in levels])
e_u, e_D = final[:, 1], final[:, 2]
h = 2.0**(-np.array(levels))

fig, axs = plt.subplots(2, 1, figsize=(0.95*TEXTWIDTH, 0.8*TEXTWIDTH))
# level labels above the curve for e_u, below it for e_D (where h^2 lies above)
for ax, e, ref, name, dy in ((axs[0], e_u, 1e-1, r"e_{\mathbf{u}}", 5),
                             (axs[1], e_D, 1e-2, r"e_D", -13)):
    ax.loglog(h, e, "o-", color=COLORS[5], markersize=3.5, label=rf"${name}$")
    ax.loglog(h, ref*h**2, "--", color=COLORS[6], label=r"$h^2$")
    for level, x, y in zip(levels, h, e):
        ax.annotate(f"L{level}", (x, y), xytext=(5, dy), textcoords="offset points",
                    fontsize=args.fontsize - 1)
    ax.margins(x=0.08, y=0.25)  # room for the level labels
    # ticks at the mesh sizes h = 2^-level of the levels used
    ax.set_xticks(h, [rf"$2^{{-{level}}}$" for level in levels])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(rf"${name}$")
    ax.grid(True, which="major", color="#E5E5E5", linewidth=0.5)
    ax.legend(frameon=True, edgecolor="gray", fancybox=False).get_frame().set_linewidth(0.5)
fig.suptitle(r"$L^2$-errors after $T = 1\,\mathrm{d}$")
fig.tight_layout(pad=0.3)

os.makedirs("figures/Lauter", exist_ok=True)
fig.savefig("figures/Lauter/convergence-plot.pdf")
print("Figure written to figures/Lauter/convergence-plot.pdf")
