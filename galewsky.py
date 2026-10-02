"""Galewsky barotropic instability test (Section 4.2 of the paper).

A balanced midlatitude jet plus a small depth perturbation h' develops a
barotropic instability: a fast gravity-wave adjustment in the first hours,
followed by a slow roll-up of the vorticity. The reference depth Dbar is the
mean depth of 10 km.

Reference:
    J. Galewsky, R. K. Scott and L. M. Polvani, An initial-value problem for
    testing numerical models of the global shallow-water equations,
    Tellus A 56 (2004) 429-440,
    https://doi.org/10.1111/j.1600-0870.2004.00071.x

Usage:
    python galewsky.py --ref_level=6 --dt=550 --n_adv_cycles=2
    python galewsky.py --help     # all options and their defaults

Output in <output>/galewsky/L<ref_level>_dt<dt>_N<n_adv_cycles>/:
Galewsky.pvd for ParaView, with --latlon also Galewsky_lonlat.pvd on a lon-lat grid, and
diagnostics.txt with the columns t, energy, enstrophy, div_rms.
"""
import argparse
from pathlib import Path

from firedrake import *

from swe_projection.common import DAY, sphere_mesh
from swe_projection.galewsky_utils import velocity_expression, depth_expression
from swe_projection.lonlat_plot_utils import get_V_ll_rect, sphere_to_latlongrect, sphere_to_latlongrect_vec
from swe_projection.output import Progress, save_run_info, start_diagnostics, write_diagnostics
from swe_projection.stepper import SWEWithProjection

parser = argparse.ArgumentParser(description="Galewsky barotropic instability test")
parser.add_argument("--ref_level", type=int, default=6, help="icosahedral mesh refinement level")
parser.add_argument("--dt", type=float, default=1100./4, help="time step in seconds")
parser.add_argument("--n_adv_cycles", type=int, default=1, help="number of advection subcycles")
parser.add_argument("--T_days", type=float, default=6., help="end time in days")
parser.add_argument("--dumpt", type=float, default=3600., help="output interval in seconds")
parser.add_argument("--latlon", action="store_true", help="also write output on a lon-lat grid")
parser.add_argument("--output", default="output", help="base directory for the output")
args, _ = parser.parse_known_args()  # unknown options (e.g. -log_view) are left for PETSc

progress = Progress(f"Galewsky, level {args.ref_level}, dt = {args.dt:g} s, N = {args.n_adv_cycles}",
                    args.T_days*DAY)

H = Constant(10000.)  # reference depth Dbar, equal to the mean depth H0 of the initial state

mesh_sphere = sphere_mesh(args.ref_level)
x = SpatialCoordinate(mesh_sphere)

u_expr = velocity_expression(*x)
D_expr, _ = depth_expression(*x)  # balanced depth plus the perturbation h'

T = args.T_days*DAY
dt = args.dt
dtc = Constant(dt)

# First-order predictor with time step dt/2: it is only used to compute the
# advecting velocity ubar at the half step for the second-order stepper
SWE_stepper_1 = SWEWithProjection(mesh_sphere, dtc/2, u_expr, D_expr, H, n_adv_cycles=1)
SWE_stepper_2 = SWEWithProjection(mesh_sphere, dtc, u_expr, D_expr, H, second_order=True,
                                  n_adv_cycles=args.n_adv_cycles)

# Output
out_dir = Path(args.output) / f"galewsky/L{args.ref_level}_dt{dt:g}_N{args.n_adv_cycles}"
out_dir.mkdir(parents=True, exist_ok=True)
save_run_info(out_dir, args, mesh_sphere.comm)
out_file = VTKFile(out_dir / "Galewsky.pvd")

if args.latlon:
    VS = FunctionSpace(mesh_sphere, "CG", degree=2)  # for |u| before the lon-lat interpolation
    V_rect = get_V_ll_rect()
    D_ll = Function(V_rect, name="D")
    D_pert_ll = Function(V_rect, name="D_pert")
    u_ll = Function(V_rect, name="u magnitude")
    q_ll = Function(V_rect, name="relative vorticity")
    pot_q_ll = Function(V_rect, name="potential vorticity")
    courant_ll = Function(V_rect, name="courant number")
    out_file_ll = VTKFile(out_dir / "Galewsky_lonlat.pvd")
    file_pert = VTKFile(out_dir / "D_pert.pvd")

diag_file = out_dir / "diagnostics.txt"
start_diagnostics(diag_file, ["t", "energy", "enstrophy", "div_rms"], mesh_sphere.comm)


def write_output(t):
    """Write fields and diagnostics at time t."""
    SWE_stepper_2.compute_Courant()
    SWE_stepper_2.compute_vorticity()
    out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.un, SWE_stepper_2.D_pert,
                   SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

    if args.latlon:
        sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
        sphere_to_latlongrect(SWE_stepper_2.D_pert, D_pert_ll)
        sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
        sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
        sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
        sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
        out_file_ll.write(D_ll, u_ll, q_ll, pot_q_ll, courant_ll)
        file_pert.write(D_pert_ll)

    write_diagnostics(diag_file, (t, *SWE_stepper_2.compute_phys_quant()), mesh_sphere.comm)


t = 0.0
tdump = 0.
write_output(t)
progress.update(t)

while t < T - 0.5*dt:
    SWE_stepper_2.step(SWE_stepper_1)

    t += dt
    tdump += dt
    progress.step(t)

    if tdump > args.dumpt - dt*0.5:
        write_output(t)
        progress.update(t)
        tdump -= args.dumpt

progress.finish()
