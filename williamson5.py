"""Williamson 5 test: zonal flow over an isolated mountain.

A geostrophically balanced zonal flow is perturbed by a conical mountain,
which generates Rossby waves. The test checks energy and enstrophy
conservation and how well the H(div)-projection controls the divergence.

Reference:
    D. L. Williamson, J. B. Drake, J. J. Hack, R. Jakob and P. N. Swarztrauber,
    A standard test set for numerical approximations to the shallow water
    equations in spherical geometry, J. Comput. Phys. 102 (1992) 211-224,
    https://doi.org/10.1016/S0021-9991(05)80016-6 (test case 5).

Usage:
    python williamson5.py --ref_level=6 --dt=1200 --n_adv_cycles=2
    python williamson5.py --help     # all options and their defaults

Output in <output>/williamson5/L<ref_level>_dt<dt>_N<n_adv_cycles>/:
Williamson5.pvd for ParaView (with --latlon also Williamson5_ll.pvd on a
lon-lat grid) and diagnostics.txt with the columns t, energy, enstrophy, div_rms.
"""
import argparse
from pathlib import Path

from firedrake import *
from firedrake.petsc import PETSc

from swe_projection.common import R0, OMEGA, G, DAY, sphere_mesh
from swe_projection.lonlat_plot_utils import get_V_ll_rect, sphere_to_latlongrect, sphere_to_latlongrect_vec
from swe_projection.output import start_diagnostics, write_diagnostics
from swe_projection.stepper import SWEWithProjection

parser = argparse.ArgumentParser(description="Williamson 5 test: zonal flow over an isolated mountain")
parser.add_argument("--ref_level", type=int, default=6, help="icosahedral mesh refinement level")
parser.add_argument("--dt", type=float, default=1200., help="time step in seconds")
parser.add_argument("--n_adv_cycles", type=int, default=2, help="number of advection subcycles")
parser.add_argument("--T_days", type=float, default=15., help="end time in days")
parser.add_argument("--dumpt", type=float, default=21600., help="output interval in seconds (default: 4 times a day)")
parser.add_argument("--latlon", action="store_true", help="also write output on a lon-lat grid")
parser.add_argument("--output", default="output", help="base directory for the output")
args, _ = parser.parse_known_args()  # unknown options (e.g. -log_view) are left for PETSc

H = Constant(5960.)  # mean height, also the reference depth Dbar
Omega = Constant(OMEGA)
g = Constant(G)

mesh_sphere = sphere_mesh(args.ref_level)
x = SpatialCoordinate(mesh_sphere)

# Initial velocity: solid-body rotation
u_0 = 20.0  # maximum zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])

# Initial total height D + b, in geostrophic balance
D_expr = H - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g

# Conical mountain
rl = pi/9.0
lambda_x = atan2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = min_value(pow(rl, 2),
                   pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)

T = args.T_days*DAY
dt = args.dt
dtc = Constant(dt)

# First-order predictor with time step dt/2: it is only used to compute the
# advecting velocity ubar at the half step for the second-order stepper
SWE_stepper_1 = SWEWithProjection(mesh_sphere, dtc/2, u_expr, D_expr, H,
                                  bexpr=bexpr, n_adv_cycles=1)
SWE_stepper_2 = SWEWithProjection(mesh_sphere, dtc, u_expr, D_expr, H,
                                  bexpr=bexpr, second_order=True, n_adv_cycles=args.n_adv_cycles)

# Output
out_dir = Path(args.output) / f"williamson5/L{args.ref_level}_dt{dt:g}_N{args.n_adv_cycles}"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = VTKFile(out_dir / "Williamson5.pvd")

if args.latlon:
    VS = FunctionSpace(mesh_sphere, "CG", degree=2)  # for |u| before the lon-lat interpolation
    V_rect = get_V_ll_rect()
    D_ll = Function(V_rect, name="D")
    Dpb_ll = Function(V_rect, name="D+b")
    u_ll = Function(V_rect, name="u magnitude")
    q_ll = Function(V_rect, name="relative vorticity")
    pot_q_ll = Function(V_rect, name="potential vorticity")
    courant_ll = Function(V_rect, name="courant number")
    out_file_ll = VTKFile(out_dir / "Williamson5_ll.pvd")

diag_file = out_dir / "diagnostics.txt"
start_diagnostics(diag_file, ["t", "energy", "enstrophy", "div_rms"], mesh_sphere.comm)


def write_output(t):
    """Write fields and diagnostics at time t."""
    SWE_stepper_2.compute_Courant()
    SWE_stepper_2.compute_vorticity()
    out_file.write(SWE_stepper_2.Dn, SWE_stepper_2.Dplusb, SWE_stepper_2.un,
                   SWE_stepper_2.Courant, SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

    if args.latlon:
        sphere_to_latlongrect(SWE_stepper_2.Dn, D_ll)
        sphere_to_latlongrect(SWE_stepper_2.Dplusb, Dpb_ll)
        sphere_to_latlongrect_vec(VS, SWE_stepper_2.un, u_ll)
        sphere_to_latlongrect(SWE_stepper_2.qn, q_ll)
        sphere_to_latlongrect(SWE_stepper_2.pot_qn, pot_q_ll)
        sphere_to_latlongrect(SWE_stepper_2.Courant, courant_ll)
        out_file_ll.write(D_ll, Dpb_ll, u_ll, q_ll, pot_q_ll, courant_ll)

    write_diagnostics(diag_file, (t, *SWE_stepper_2.compute_phys_quant()), mesh_sphere.comm)


t = 0.0
tdump = 0.
write_output(t)

while t < T - 0.5*dt:
    PETSc.Sys.Print("t = ", t)

    SWE_stepper_2.step(SWE_stepper_1)

    t += dt
    tdump += dt

    if tdump > args.dumpt - dt*0.5:
        write_output(t)
        tdump -= args.dumpt
