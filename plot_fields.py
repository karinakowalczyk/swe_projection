"""Lon-lat plots of the fields: Williamson 5 at 15 days and Galewsky at 2 h, 6 h and 144 h.

Reads the lon-lat output (written with --latlon) of the level 6 runs with
2 advection subcycles and writes
    figures/Williamson5/W5_Dpb_15d.pdf, W5_vort_15d.pdf
    figures/Galewsky/Galewsky_Dpert_2.pdf, Galewsky_Dpert_6.pdf, Galewsky_vort_144.pdf
The figures are made at the full text width; include them with
\\includegraphics[width=\\textwidth]{...} so that their fonts match the text.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from swe_projection.plot_style import add_style_args, setup

parser = argparse.ArgumentParser(description="Lon-lat plots of the Williamson 5 and Galewsky fields")
parser.add_argument("--w5_run", default="output/williamson5/L6_dt1200_N2", help="Williamson 5 run directory")
parser.add_argument("--galewsky_run", default="output/galewsky/L6_dt550_N2", help="Galewsky run directory")
add_style_args(parser)
args, _ = parser.parse_known_args()
TEXTWIDTH = setup(args)


def read_latlon(run, pvd, field, t):
    """Return lon, lat (in degrees) and the values of field at time t [s] on the lon-lat grid."""
    run = Path(run)
    with open(run/"run_info.json") as f:
        dumpt = json.load(f)["options"]["dumpt"]
    index = round(t/dumpt)  # output number
    files = [d.get("file") for d in ET.parse(run/pvd).getroot().iter("DataSet")]
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(run/files[index]))
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData())
    values = vtk_to_numpy(grid.GetPointData().GetArray(field))

    # The nodes lie on a regular lon-lat grid; points shared by several MPI
    # ranks appear more than once, so keep one of each and sort into the grid
    lonlat = np.round(points[:, :2], 10)
    _, keep = np.unique(lonlat, axis=0, return_index=True)
    lonlat, values = lonlat[keep], values[keep]
    order = np.lexsort((lonlat[:, 0], lonlat[:, 1]))  # by latitude, then longitude
    nlon = len(np.unique(lonlat[:, 0]))
    shape = (-1, nlon)
    lon = np.degrees(lonlat[order, 0].reshape(shape))
    lat = np.degrees(lonlat[order, 1].reshape(shape))
    return lon, lat, values[order].reshape(shape)


def plot(lon, lat, values, label, filename, symmetric):
    """Lon-lat map with a colour bar above it; symmetric colour scale around 0 if symmetric."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.62*TEXTWIDTH), layout="constrained")
    if symmetric:
        vmax = np.abs(values).max()
        vmin = -vmax
    else:
        vmin, vmax = values.min(), values.max()
    mesh = ax.pcolormesh(lon, lat, values, cmap="coolwarm", vmin=vmin, vmax=vmax,
                         shading="gouraud", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    cbar = fig.colorbar(mesh, ax=ax, location="top", shrink=0.6, aspect=30)
    cbar.set_label(label)
    cbar.formatter.set_powerlimits((-3, 4))  # e.g. vorticity as multiples of 10^-4
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300)  # the map is a 300 dpi image, text and axes stay vector graphics
    plt.close(fig)
    print("written", filename)


day, hour = 86400., 3600.

# Williamson 5 at 15 days
lon, lat, v = read_latlon(args.w5_run, "Williamson5_ll.pvd", "D+b", 15*day)
plot(lon, lat, v, r"$D + b$ [m]", "figures/Williamson5/W5_Dpb_15d.pdf", symmetric=False)
lon, lat, v = read_latlon(args.w5_run, "Williamson5_ll.pvd", "relative vorticity", 15*day)
plot(lon, lat, v, r"relative vorticity [s$^{-1}$]", "figures/Williamson5/W5_vort_15d.pdf", symmetric=True)

# Galewsky: depth perturbation D(t) - D(0) during the fast adjustment, vorticity at the end
for h in (2, 6):
    lon, lat, v = read_latlon(args.galewsky_run, "D_pert.pvd", "D_pert", h*hour)
    plot(lon, lat, v, r"$\delta D$ [m]", f"figures/Galewsky/Galewsky_Dpert_{h}.pdf", symmetric=True)
lon, lat, v = read_latlon(args.galewsky_run, "Galewsky_lonlat.pvd", "relative vorticity", 144*hour)
plot(lon, lat, v, r"relative vorticity [s$^{-1}$]", "figures/Galewsky/Galewsky_vort_144.pdf", symmetric=True)
