"""Common layout for the plots: figures at their printed size, fonts as in the text.

The figures are made at a fraction of the text width of the document they go
into, so that they can be included unscaled (e.g. width=0.9\\textwidth) and their
fonts have the same size as the text. The defaults are for elsarticle with the
options preprint,12pt (text width 390 pt, 12 pt Computer Modern).
"""
import shutil

import matplotlib
matplotlib.use("Agg")  # save figures without opening plot windows
import matplotlib.pyplot as plt

# One colour per mesh refinement level
COLORS = {3: "#009988", 4: "#cc3311", 5: "#0077bb", 6: "#ee7733"}


def add_style_args(parser):
    """Add the layout options to an argparse parser."""
    parser.add_argument("--textwidth", type=float, default=390.,
                        help="text width of the document in pt (from \\showthe\\textwidth)")
    parser.add_argument("--fontsize", type=float, default=12., help="font size of the document in pt")
    parser.add_argument("--font", choices=["cm", "times"], default="cm",
                        help="document font: cm (Computer Modern) or times")


def setup(args):
    """Set the matplotlib fonts; return the text width in inches.

    Text is typeset with LaTeX if it is installed, otherwise with matplotlib's
    built-in Computer Modern (close, but not identical).
    """
    small = args.fontsize - 1  # ticks and legends
    style = {
        "font.family": "serif",
        "font.size": args.fontsize,
        "axes.labelsize": args.fontsize,
        "axes.titlesize": args.fontsize,
        "figure.titlesize": args.fontsize,
        "legend.fontsize": small,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "savefig.bbox": "standard",  # keep the figure at exactly its figsize
    }
    if shutil.which("latex"):
        preamble = r"\usepackage{amsmath}"
        if args.font == "times":
            preamble += r"\usepackage{newtxtext}\usepackage{newtxmath}"
        style.update({"text.usetex": True, "text.latex.preamble": preamble})
    elif args.font == "times":
        style.update({"font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix"})
    else:
        style.update({"font.serif": ["cmr10"], "mathtext.fontset": "cm",
                      "axes.formatter.use_mathtext": True})
    plt.rcParams.update(style)
    return args.textwidth/72.27
