"""Plots of the energy, enstrophy and divergence diagnostics (Williamson 5 or Galewsky).

Reads output/<test>/L<level>_dt<dt>_N<N>/diagnostics.txt written by williamson5.py
or galewsky.py and saves the figures to figures/<Test>/stability/.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    """Reads and normalises the diagnostics of one run."""

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def load_data(self, column):
        """Load one column (t, energy, enstrophy or div_rms) of diagnostics.txt."""
        filename = self.base_path / 'diagnostics.txt'
        with open(filename, 'r') as file:
            names = file.readline().lstrip('#').split()
        return np.loadtxt(filename, ndmin=2)[:, names.index(column)]

    def normalize_data(self, data):
        """Absolute relative change |data - data[0]| / data[0]."""
        return abs(data - data[0]) / data[0]

    def normalize_data_signed(self, data):
        """Relative change (data - data[0]) / data[0]."""
        return (data - data[0]) / data[0]

    def normalize_divl2(self, data):
        """|R_div|, from the RMS divergence ||div u|| / |Omega|^(1/2)."""
        return abs(data - data[0])

    def normalize_divl2_signed(self, data):
        """R_div, from the RMS divergence ||div u|| / |Omega|^(1/2)."""
        return data - data[0]


def create_plot(x_data, y_data, title, ylabel, labels, figsize=(12, 7), Williamson5=False, use_log=True):
    """Plot the curves y_data[i] over x_data[i], in pairs (dt, 2dt) for each refinement level."""
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=figsize)

    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # One colour per refinement level
    ref_colors = {
        5: '#0077bb',  # blue
        6: '#ee7733'   # orange
    }

    # Markers and line styles for dt (circle, solid) and 2dt (square, dotted)
    markers = ['o', 's']
    linestyles = ['-', ':']

    # Each run has its own time array
    for i in range(0, len(y_data), 2):
        ref_level = int(labels[i].split('refinement level ')[1].split(')')[0])
        plot_func = ax.semilogy if use_log else ax.plot

        plot_func(x_data[i], y_data[i],
                  label=labels[i],
                  color=ref_colors[ref_level],
                  linewidth=1.0,
                  linestyle=linestyles[0],
                  marker=markers[0],
                  markersize=3,
                  markevery=max(1, len(x_data[i]) // 60))

        plot_func(x_data[i + 1], y_data[i + 1],
                  label=labels[i + 1],
                  color=ref_colors[ref_level],
                  linewidth=1.0,
                  linestyle=linestyles[1],
                  marker=markers[1],
                  markersize=3,
                  markevery=max(1, len(x_data[i + 1]) // 60))

    ax.set_title(title, fontsize=14, pad=10)
    if Williamson5:
        ax.set_xlabel('days', fontsize=14)
    else:
        ax.set_xlabel('hours', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    ncol = 2 if len(labels) > 3 else 1
    legend = ax.legend(fontsize=12,
                       frameon=True,
                       edgecolor='gray',
                       fancybox=False,
                       borderaxespad=1,
                       ncol=ncol)
    legend.get_frame().set_linewidth(0.5)

    # Tick every day
    if Williamson5:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    ax.tick_params(axis='both', labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Energy, enstrophy and divergence plots (Williamson 5 or Galewsky)")
    parser.add_argument("--test", choices=["williamson5", "galewsky"], required=True)
    args, _ = parser.parse_known_args()
    Williamson5 = args.test == "williamson5"

    # Time steps of the runs: dt with 1 advection cycle, 2dt with 2 cycles
    if Williamson5:
        ref_level_config = {
            5: {'dt': 1200.0, 'dt2': 2400.0},
            6: {'dt': 600.0, 'dt2': 1200.0}
        }
        base_path = 'output/williamson5'
    else:
        ref_level_config = {
            5: {'dt': 550.0, 'dt2': 1100.0},
            6: {'dt': 275.0, 'dt2': 550.0}
        }
        base_path = 'output/galewsky'

    ref_levels = [5, 6]

    # Normalised diagnostics per run; keys are the columns of diagnostics.txt
    data_sets = {
        'energy': {},
        'enstrophy': {},
        'div_rms': {}
    }
    times = {}
    time_unit = 86400. if Williamson5 else 3600.  # plot in days (W5) or hours (Galewsky)

    for ref_level in ref_levels:
        dt = ref_level_config[ref_level]['dt']
        dt2 = ref_level_config[ref_level]['dt2']
        dt_loader = DataLoader(f'{base_path}/L{ref_level}_dt{dt:g}_N1')
        dt2_loader = DataLoader(f'{base_path}/L{ref_level}_dt{dt2:g}_N2')

        dt_key = f'dt_ref{ref_level}'
        dt2_key = f'2dt_ref{ref_level}'

        times[dt_key] = dt_loader.load_data('t') / time_unit
        times[dt2_key] = dt2_loader.load_data('t') / time_unit

        # Energy and enstrophy relative to the initial value, divergence as R_div;
        # absolute values for the logarithmic plots of Williamson 5
        for metric in data_sets:
            if metric == 'div_rms':
                if Williamson5:
                    data_sets[metric][dt_key] = dt_loader.normalize_divl2(
                        dt_loader.load_data(metric)
                    )
                    data_sets[metric][dt2_key] = dt2_loader.normalize_divl2(
                        dt2_loader.load_data(metric)
                    )
                else:
                    data_sets[metric][dt_key] = dt_loader.normalize_divl2_signed(
                        dt_loader.load_data(metric)
                    )
                    data_sets[metric][dt2_key] = dt2_loader.normalize_divl2_signed(
                        dt2_loader.load_data(metric)
                    )
            else:
                if Williamson5:
                    data_sets[metric][dt_key] = dt_loader.normalize_data(
                        dt_loader.load_data(metric)
                    )
                    data_sets[metric][dt2_key] = dt2_loader.normalize_data(
                        dt2_loader.load_data(metric)
                    )
                else:
                    data_sets[metric][dt_key] = dt_loader.normalize_data_signed(
                        dt_loader.load_data(metric)
                    )
                    data_sets[metric][dt2_key] = dt2_loader.normalize_data_signed(
                        dt2_loader.load_data(metric)
                    )

    plot_configs = [
        {
            'title': 'Normalised energy error over time',
            'ylabel': r'$e(t)$' if not Williamson5 else r'$\log |e(t)|$',
            'metric': 'energy'
        },
        {
            'title': 'Normalised enstrophy error over time',
            'ylabel': r'$e^{\text{enstr.}}(t)$' if not Williamson5 else r'$\log |e^{\text{enstr.}}(t)|$',
            'metric': 'enstrophy'
        },
        {
            'title': 'Relative change in velocity divergence over time',
            'ylabel': r'$R_{\text{div}}(t)$' if not Williamson5 else r'$\log|R_{\text{div}}(t)|$',
            'metric': 'div_rms'
        }
    ]

    figs = []
    for config in plot_configs:
        plot_times = []
        plot_data = []
        labels = []

        for ref_level in ref_levels:
            dt_key = f'dt_ref{ref_level}'
            dt2_key = f'2dt_ref{ref_level}'

            plot_times.extend([times[dt_key], times[dt2_key]])
            plot_data.extend([
                data_sets[config['metric']][dt_key],
                data_sets[config['metric']][dt2_key]
            ])

            dt = ref_level_config[ref_level]['dt']
            dt2 = ref_level_config[ref_level]['dt2']

            labels.extend([
                rf'$\Delta t = ${int(dt)} s (refinement level {ref_level})',
                rf'$2\Delta t = ${int(dt2)} s, 2 adv. cycles (refinement level {ref_level})'
            ])

        fig, ax = create_plot(
            plot_times,
            plot_data,
            config['title'],
            config['ylabel'],
            labels,
            Williamson5=Williamson5,
            use_log=Williamson5
        )
        figs.append(fig)
        plt.show()

    for i, fig in enumerate(figs):
        output_dir = 'Williamson5' if Williamson5 else 'Galewsky'
        os.makedirs(f"figures/{output_dir}/stability", exist_ok=True)
        fig.savefig(f"figures/{output_dir}/stability/plot_{i + 1}.pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()