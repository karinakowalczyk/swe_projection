"""Convergence test: unsteady solid-body rotation with an exact solution (Section 4.3 of the paper).

A solid-body rotation about an axis tilted against the Earth's rotation axis,
over topography, for which the exact solution is known at all times. The
scheme is run on several mesh levels, and the normalised L2 errors in u and
in the total height D + b are recorded.

Reference:
    M. Läuter, D. Handorf and K. Dethloff, Unsteady analytical solutions of
    the spherical shallow water equations, J. Comput. Phys. 210 (2005)
    535-553, https://doi.org/10.1016/j.jcp.2005.04.022 (Example 3).

Usage:
    python lauter.py --ref_levels 3 4 5 6 --dts 400 200 100 50
    python lauter.py --help     # all options and their defaults

Output in <output>/lauter/: unsteady_body_rot<level>.pvd and
exact_solution<level>.pvd for ParaView, and errors_L<level>.txt with the
columns t, e_u, e_D (hourly).
"""
import argparse
import os

from firedrake import *
from firedrake.petsc import PETSc

from swe_projection.common import R0, OMEGA, G, DAY, sphere_mesh
from swe_projection.output import start_diagnostics, write_diagnostics
from swe_projection.stepper import SWEWithProjection

parser = argparse.ArgumentParser(description="Lauter unsteady solid-body rotation: convergence test")
parser.add_argument("--ref_levels", type=int, nargs="+", default=[3, 4, 5, 6], help="list of mesh refinement levels")
parser.add_argument("--dts", type=float, nargs="+", default=[400., 200., 100., 50.], help="time step for each refinement level in seconds")
parser.add_argument("--T_days", type=float, default=1., help="end time in days")
parser.add_argument("--output", default="output", help="base directory for the output")
parser.add_argument("--compute_ref", action="store_true", help="recompute the reference norms uref, dref on a level 7 mesh")
args, _ = parser.parse_known_args()  # unknown options (e.g. -log_view) are left for PETSc
assert len(args.ref_levels) == len(args.dts), "need one dt per refinement level"
out_dir = os.path.join(args.output, "lauter")
os.makedirs(out_dir, exist_ok=True)

Omega = Constant(OMEGA)
g = Constant(G)

e1 = as_vector([1., 0., 0.])
e2 = as_vector([0., 1., 0.])
e3 = as_vector([0., 0., 1.])

alpha = pi/4
c_vec = -sin(alpha)*e1 + cos(alpha)*e3  # rotation axis of the flow

u0 = (2*pi*R0/12)/DAY
H = 133681./g  # k1/g, also the reference depth Dbar


def geometry(x):
    """Return the unit normal k and height z of the point x projected onto the sphere.

    The mesh only approximates the sphere, so x itself is not exactly at radius R0.
    """
    r = (x[0]**2+x[1]**2+x[2]**2)**0.5
    k = as_vector(x)/r
    z = x[2]/r*R0
    return k, z


def exact_sol(x, t):
    """Return the exact velocity u and total height D + b at time t."""
    k, z = geometry(x)
    # Basis rotating with the Earth
    b1 = cos(Omega*t)*e1 + sin(Omega*t)*e2
    b2 = -sin(Omega*t)*e1 + cos(Omega*t)*e2
    b3 = e3
    phi_t = dot(c_vec, b1)*e1 + dot(c_vec, b2)*e2 + dot(c_vec, b3)*e3
    phi_t_dot_n = dot(phi_t, k)
    u_exact = u0 * cross(phi_t, k)
    D_exact = H - (1/(2*g))*(u0*phi_t_dot_n + z*Omega)**2 + (1/(2*g))*(z*Omega)**2
    return u_exact, D_exact


# L2 norms of the exact solution, for the normalised errors; computed on a
# level 7 mesh with --compute_ref
uref = 712008703.4132433
dref = 298004433314.0549

if args.compute_ref:
    PETSc.Sys.Print("Computing reference norms")
    u_ex_ref, D_ex_ref = exact_sol(SpatialCoordinate(sphere_mesh(7)), 0.)
    uref, dref = norm(u_ex_ref), norm(D_ex_ref)

PETSc.Sys.Print("uref, dref =", uref, dref)

for refinement_level, dt in zip(args.ref_levels, args.dts):
    PETSc.Sys.Print(f"START test for ref_level{refinement_level} with dt = {dt}.")

    mesh = sphere_mesh(refinement_level)
    x = SpatialCoordinate(mesh)
    _, z = geometry(x)

    t = 0.
    u_expr, D_expr = exact_sol(x, t)
    bexpr = (1/(2*g))*(z*Omega)**2

    T = args.T_days*DAY
    dtc = Constant(dt)

    # First-order predictor with time step dt/2: it is only used to compute the
    # advecting velocity ubar at the half step for the second-order stepper
    SWE_stepper_1 = SWEWithProjection(mesh, dtc/2, u_expr, D_expr, H,
                                      bexpr=bexpr, n_adv_cycles=1)
    SWE_stepper_2 = SWEWithProjection(mesh, dtc, u_expr, D_expr, H,
                                      bexpr=bexpr, second_order=True, n_adv_cycles=1)

    # Output of the numerical and the exact solution
    out_file = VTKFile(os.path.join(out_dir, f"unsteady_body_rot{refinement_level}.pvd"))
    file_exact = VTKFile(os.path.join(out_dir, f"exact_solution{refinement_level}.pvd"))
    V1 = FunctionSpace(mesh, FiniteElement("BDM", triangle, degree=2))
    V2 = FunctionSpace(mesh, "DG", 1)
    u_ex = Function(V1, name="u").interpolate(u_expr)
    D_ex = Function(V2, name="D").interpolate(D_expr)
    file_exact.write(u_ex, D_ex)

    SWE_stepper_2.compute_Courant()
    SWE_stepper_2.compute_vorticity()
    out_file.write(SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant,
                   SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

    # Normalised L2 errors e_u and e_D (of D + b)
    err_file = os.path.join(out_dir, f"errors_L{refinement_level}.txt")
    start_diagnostics(err_file, ["t", "e_u", "e_D"], mesh.comm)
    u_error = norm(SWE_stepper_2.un - u_expr)/uref
    D_error = norm(SWE_stepper_2.Dplusb - D_expr)/dref
    write_diagnostics(err_file, (t, u_error, D_error), mesh.comm)

    dumpt = 3600.
    tdump = 0.

    while t < T - 0.5*dt:
        SWE_stepper_2.step(SWE_stepper_1)

        t += dt
        tdump += dt

        if tdump > dumpt - dt*0.5:
            PETSc.Sys.Print("t =", t)

            SWE_stepper_2.compute_Courant()
            SWE_stepper_2.compute_vorticity()
            out_file.write(SWE_stepper_2.Dplusb, SWE_stepper_2.un, SWE_stepper_2.Courant,
                           SWE_stepper_2.qn, SWE_stepper_2.pot_qn)

            u_exact, D_exact = exact_sol(x, t)
            u_ex.project(u_exact)
            D_ex.project(D_exact)
            file_exact.write(u_ex, D_ex)

            u_error = norm(SWE_stepper_2.un - u_exact)/uref
            D_error = norm(SWE_stepper_2.Dplusb - D_exact)/dref
            write_diagnostics(err_file, (t, u_error, D_error), mesh.comm)

            tdump -= dumpt
