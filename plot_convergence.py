"""Convergence plot for the Lauter test, Example 3 in https://doi.org/10.1016/j.jcp.2005.04.022.

Reads the normalised errors written by lauter.py (output/lauter/errors_L<level>.txt)
and plots the errors at the final time.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Convergence plot for the Lauter test")
parser.add_argument("--ref_levels", type=int, nargs="+", default=[3, 4, 5, 6], help="mesh refinement levels to plot")
args, _ = parser.parse_known_args()
refinement_levels = args.ref_levels


def create_convergence_plot(h, u_errors, D_errors, figsize=(6, 6)):
    """Plot e_u (top) and e_D (bottom) against the mesh size h, with an h^2 reference line."""
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    colors = ['#0077bb',  # blue
              '#ee7733']  # orange

    for ax in axs:
        # Major grid lines, minor grid lines and ticks only on the x-axis
        ax.grid(True, which='major', color='#E5E5E5', linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', axis='x', color='#F2F2F2', linestyle='-', linewidth=0.3)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(plt.LogLocator(subs='all'))
        ax.yaxis.set_minor_locator(plt.NullLocator())

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', labelsize=9)

    # Velocity errors
    axs[0].loglog(h, u_errors,
                  color=colors[0],
                  marker='o',
                  markersize=4,
                  linewidth=1.0,
                  label=r"$e_{\mathrm{u}}$")

    axs[0].loglog(h, 1e-1 * h ** 2,
                  color=colors[1],
                  linewidth=1.0,
                  linestyle='--',
                  label=r"$h^2$")

    for level, x, y in zip(refinement_levels, h, u_errors):
        axs[0].annotate(f'L{level}',
                        (x, y),
                        xytext=(5, 7),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.7,
                                  pad=0.1))

    # Errors of D + b
    axs[1].loglog(h, D_errors,
                  color=colors[0],
                  marker='o',
                  markersize=4,
                  linewidth=1.0,
                  label=r"$e_D$")

    axs[1].loglog(h, 1e-2 * h ** 2,
                  color=colors[1],
                  linewidth=1.0,
                  linestyle='--',
                  label=r"$h^2$")

    for level, x, y in zip(refinement_levels, h, D_errors):
        axs[1].annotate(f'L{level}',
                        (x, y),
                        xytext=(5, -7),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.7,
                                  pad=0.1))

    axs[0].set_xlabel(r'$\log h$', fontsize=9)
    axs[0].set_ylabel(r'$\log e_u$', fontsize=9)
    axs[1].set_xlabel(r'$\log h$', fontsize=9)
    axs[1].set_ylabel(r'$\log e_D$', fontsize=9)

    for ax in axs:
        legend = ax.legend(fontsize=9,
                           frameon=True,
                           edgecolor='gray',
                           fancybox=False,
                           borderaxespad=1)
        legend.get_frame().set_linewidth(0.5)

    plt.suptitle(r"$L^2$-errors after $T = 1$d",
                 fontsize=9)

    plt.tight_layout()
    return fig, axs


h = 2.0 ** (-np.array(refinement_levels))

# Errors at the final time (last row; columns t, e_u, e_D)
final = np.array([np.loadtxt(f"output/lauter/errors_L{level}.txt", ndmin=2)[-1]
                  for level in refinement_levels])
u_errors = final[:, 1]
D_errors = final[:, 2]

fig, axs = create_convergence_plot(h, u_errors, D_errors)

os.makedirs("figures/Lauter", exist_ok=True)
fig.savefig("figures/Lauter/convergence-plot.pdf", dpi=300, bbox_inches='tight', format='pdf')
fig.savefig("figures/Lauter/convergence-plot.png", dpi=300, bbox_inches='tight')
plt.show()