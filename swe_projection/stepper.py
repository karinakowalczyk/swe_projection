"""Time stepper for the rotating shallow-water equations on the sphere.

Each time step is split into
  - an explicit step (second-order scheme only): half a time step of the
    gravity, Coriolis and reference-depth divergence terms,
  - an advection step: DG-upwind advection of u and D - Dbar with SSPRK3,
    optionally subcycled N times with step dt/N,
  - a projection step: implicit solve for the gravity, Coriolis and
    reference-depth divergence terms, which projects u onto H(div).

Velocity is in BDM2 (broken BDM2 during advection) and depth in DG1.
"""
from firedrake import *
from firedrake.petsc import PETSc

from swe_projection.common import R0, OMEGA, G


class SWEWithProjection:
    """Shallow-water stepper with advection and H(div)-projection steps.

    :arg mesh: sphere mesh with initialised cell orientations.
    :arg dtc: time step, a :class:`Constant`.
    :arg u_expr: initial velocity.
    :arg D_expr: initial total height D + b.
    :arg H: reference depth Dbar used in the splitting.
    :arg bexpr: bottom topography b, or None for a flat bottom.
    :arg second_order: if True, use the second-order scheme (explicit half
        step before, implicit half step after the advection step); advance
        with :meth:`step`. If False, the first-order scheme, used as the
        predictor for the advecting velocity.
    :arg n_adv_cycles: number N of advection subcycles per time step.
    """

    def __init__(self, mesh, dtc, u_expr, D_expr, H, bexpr=None,
                 second_order=False, n_adv_cycles=1):

        self.dtc = dtc
        self.n_adv_cycles = n_adv_cycles
        self.mesh = mesh
        self.second_order = second_order
        x = SpatialCoordinate(mesh)
        n = FacetNormal(mesh)
        Omega = Constant(OMEGA)
        f = 2*Omega*x[2]/Constant(R0)  # Coriolis parameter
        g = Constant(G)

        # Function spaces: CG3 for vorticity, BDM2 (and broken BDM2) for
        # velocity, DG1 for depth
        self.V0 = FunctionSpace(mesh, "CG", degree=3)
        element = FiniteElement("BDM", triangle, degree=2)
        V1_broken = FunctionSpace(mesh, BrokenElement(element))
        V1 = FunctionSpace(mesh, element)
        V2 = FunctionSpace(mesh, "DG", 1)
        W = MixedFunctionSpace((V1, V2))

        # Advecting velocity
        self.ubar = Function(V1, name="ubar").interpolate(u_expr)

        if bexpr:
            b = Function(V2, name="Topography")
            b.interpolate(bexpr)
        else:
            b = 0
        self.b = b

        # Solution (un, Dn), and derived output fields
        self.Dn = Function(V2, name="D").interpolate(D_expr - b)
        self.D0 = Function(V2, name="D0").assign(self.Dn)
        self.Dbar = Function(V2).assign(H)
        self.Dplusb = Function(V2, name="D+b").assign(self.Dn + self.b)
        self.un = Function(V1, name="u").assign(self.ubar)
        self.D_pert = Function(V2, name='delta D').assign(self.Dn - self.D0)

        # Solution of the projection step (and of the explicit step)
        self.Unp1 = Function(W)
        self.unp1, self.Dnp1 = self.Unp1.subfunctions
        if second_order:
            self.Uhat = Function(W)
        self.Dnp1.assign(self.Dn)
        self.unp1.assign(self.un)

        outward_normals = CellNormal(mesh)

        def perp(u):
            return cross(outward_normals, u)

        # Courant number per cell K: (dt/N)/|K| times the inflow through the
        # boundary of K, i.e. the effective Courant number of the advection
        # sub-steps
        def both(u):
            return 2*avg(u)

        self.DG0 = FunctionSpace(self.mesh, "DG", 0)
        unn = 0.5 * (inner(-self.un, n) + abs(inner(-self.un, n)))  # inflow only
        v = TestFunction(self.DG0)
        phi = TrialFunction(self.DG0)
        self.courant_lhs_form = phi * v * dx
        self.courant_rhs_form = (self.dtc/self.n_adv_cycles) * (both(unn * v) * dS)
        self.Courant = Function(self.DG0, name="Courant")

        # Relative vorticity q = curl u, and potential vorticity with
        # D q_pot = q + f
        q = TrialFunction(self.V0)
        p = TestFunction(self.V0)
        self.qn = Function(self.V0, name="relative vorticity")
        self.pot_qn = Function(self.V0, name="potential vorticity")
        qparams = {'ksp_type': 'cg'}
        veqn = q*p*dx + inner(perp(grad(p)), self.un)*dx
        vprob = LinearVariationalProblem(lhs(veqn), rhs(veqn), self.qn)
        self.qsolver = LinearVariationalSolver(vprob, solver_parameters=qparams)
        pot_vort_eqn = (self.Dn*q*p*dx + inner(perp(grad(p)), self.un)*dx - f*p*dx)
        pot_vort_prob = LinearVariationalProblem(lhs(pot_vort_eqn), rhs(pot_vort_eqn), self.pot_qn)
        self.pot_vort_solver = LinearVariationalSolver(pot_vort_prob, solver_parameters=qparams)

        self.area = assemble(Constant(1.0)*dx(domain=mesh))  # |Omega|, for the RMS divergence

        # Fields advected in the advection step
        self.D = Function(V2).assign(self.Dn)
        self.u = Function(V1_broken).project(self.un)

        # ADVECTION STEP
        # Explicit SSPRK3 stages: each stage solves M dq = L(q), with the
        # inverse mass matrices M^{-1} precomputed by Slate
        dD_trial = TrialFunction(V2)
        phi = TestFunction(V2)
        a_D = phi*dD_trial*dx
        self.dmass_inv = assemble(Tensor(a_D).inv)

        du_trial = TrialFunction(V1_broken)
        w = TestFunction(V1_broken)
        a_u = inner(w, du_trial)*dx
        self.umass_inv = assemble(Tensor(a_u).inv)

        # Upwind flux u.n where positive, zero otherwise
        unn = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))

        def eq_D(ubar, Dbar):
            # Upwind DG advection of D - Dbar; the flux div(ubar Dbar) is
            # treated in the projection step
            uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
            return (-inner(grad(phi), ubar)*(self.D-Dbar)*dx
                    + jump(phi)*(uup('+')*(self.D-Dbar)('+')
                                 - uup('-')*(self.D-Dbar)('-'))*dS)

        def adv_u(ubar):
            # Upwind DG advection of the broken velocity u
            unn = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
            return (-inner(div(outer(w, ubar)), self.u)*dx
                    + dot(jump(w), (unn('+')*self.u('+') - unn('-')*self.u('-')))*dS)

        L1_D = (dtc/n_adv_cycles)*(-eq_D(self.ubar, self.Dbar))

        eq_u = adv_u(self.ubar)
        # Additional facet terms on the mesh, where the normals of
        # neighbouring cells do not cancel: n('+') + n('-') != 0
        eq_u += unn('+')*inner(w('-'), n('+')+n('-'))*inner(self.u('+'), n('+'))*dS
        eq_u += unn('-')*inner(w('+'), n('+')+n('-'))*inner(self.u('-'), n('-'))*dS

        L1_u = (dtc/n_adv_cycles)*(-eq_u)

        # Right-hand sides for the three stages, evaluated at q^n, q^(1), q^(2)
        self.D1 = Function(V2)
        self.D2 = Function(V2)
        self.L1_D = L1_D
        self.L2_D = replace(L1_D, {self.D: self.D1})
        self.L3_D = replace(L1_D, {self.D: self.D2})

        self.u1 = Function(V1_broken)
        self.u2 = Function(V1_broken)
        self.L1_u = L1_u
        self.L2_u = replace(L1_u, {self.u: self.u1})
        self.L3_u = replace(L1_u, {self.u: self.u2})

        # Increments and assembled right-hand sides of the stages
        self.dD = Function(V2)
        self.du = Function(V1_broken)
        self.rhs = Cofunction(V2.dual())
        self.rhs_u = Cofunction(V1_broken.dual())

        params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

        # PROJECTION STEP
        # Implicit in (u, D): gravity, Coriolis and div(u Dbar), with the full
        # time step (first order) or half a time step (second order)
        self.unp1, self.Dnp1 = split(self.Unp1)
        v, rho = TestFunctions(W)

        def proj_u():
            return (-div(v)*g*(self.Dnp1+b)*dx
                    + inner(v, f*perp(self.unp1))*dx)

        def proj_D(Dbar):
            return rho*div(self.unp1*Dbar)*dx

        factor = Constant(1.)
        if self.second_order:
            factor = Constant(0.5)

        a_proj_u = inner(v, self.unp1 - self.u)*dx + (dtc*factor)*proj_u()
        a_proj_D = rho*(self.Dnp1 - self.D)*dx + (dtc*factor)*proj_D(self.Dbar)
        a_proj = a_proj_u + a_proj_D

        prob = NonlinearVariationalProblem(a_proj, self.Unp1)

        # Linear problem, so a single Newton step; solved with hybridisation
        hparams = {
            "snes_lag_jacobian": -2,
            'mat_type': 'matfree',
            'ksp_type': 'gmres',
            'ksp_converged_reason': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'preonly',
                              'pc_type': 'lu'}}

        self.solver_proj = NonlinearVariationalSolver(prob, solver_parameters=hparams)

        # Projection of the solution onto broken BDM2 at the start of the
        # advection step
        self.projector_broken = Projector(self.un, self.u, solver_parameters=params)

        if second_order:
            # EXPLICIT STEP
            # Half a time step of gravity, Coriolis and div(u Dbar) at time t^n;
            # its result (uhat, Dhat) starts the advection step
            self.uhat, self.Dhat = split(self.Uhat)
            self.projector_adv = Projector(self.uhat, self.u, solver_parameters=params)

            def second_order_first_eq_u():
                return (-div(v)*g*(self.Dn+b)*dx
                        + inner(v, f*perp(self.un))*dx)

            def second_order_first_eq_D(Dbar):
                return rho*div(self.un*Dbar)*dx

            a_second_order_u = inner(v, self.uhat - self.un)*dx + (dtc/2)*second_order_first_eq_u()
            a_second_order_D = rho*(self.Dhat - self.Dn)*dx + (dtc/2)*second_order_first_eq_D(self.Dbar)
            a_second_order_1 = a_second_order_u + a_second_order_D
            prob_2_order_1 = NonlinearVariationalProblem(a_second_order_1, self.Uhat)

            # The mass matrices of u and D decouple, so the fieldsplit is additive
            params3 = {
                "snes_lag_jacobian": -2,
                'ksp_type': 'preonly',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit_0': {
                    'ksp_type': 'gmres',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu',
                },
                'fieldsplit_1': {
                    'ksp_type': 'preonly',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'
                }
            }
            self.solver_2ndorder_1st = NonlinearVariationalSolver(prob_2_order_1, solver_parameters=params3)

        self.unp1, self.Dnp1 = self.Unp1.subfunctions
        if self.second_order:
            self.uhat, self.Dhat = self.Uhat.subfunctions

    def step(self, predictor):
        """Advance (un, Dn) by one time step dt with the second-order scheme.

        :arg predictor: first-order stepper with time step dt/2, used to
            compute the advecting velocity ubar at the half step.
        """
        assert self.second_order, "step() is for the second-order stepper"

        # First order with dt/2, gives the half-step velocity in predictor.un
        predictor.un.assign(self.un)
        predictor.Dn.assign(self.Dn)
        predictor.advection_SSPRK3()
        predictor.projection_step()

        # Second-order step, advecting with the half-step velocity
        self.ubar.assign(predictor.un)
        self.second_order_1st_step()
        self.advection_SSPRK3()
        self.projection_step()

    @PETSc.Log.EventDecorator("second_order_1st_step")
    def second_order_1st_step(self):
        """Explicit step of the second-order scheme, result in (uhat, Dhat)."""
        self.uhat.assign(self.un)
        self.Dhat.assign(self.Dn)
        self.solver_2ndorder_1st.solve()

    def _mass_solve(self, L, rhs, mass_inv, result):
        """Set result = M^{-1} L, with the inverse mass matrix precomputed by Slate."""
        assemble(L, tensor=rhs)
        with result.dat.vec_wo as x_, rhs.dat.vec_ro as b_:
            mass_inv.petscmat.mult(b_, x_)

    def advection_SSPRK3_single_step(self):
        """One SSPRK3 step of size dt/N for D and u."""
        self._mass_solve(self.L1_D, self.rhs, self.dmass_inv, self.dD)
        self.D1.assign(self.D + self.dD)

        self._mass_solve(self.L2_D, self.rhs, self.dmass_inv, self.dD)
        self.D2.assign(0.75*self.D + 0.25*(self.D1 + self.dD))

        self._mass_solve(self.L3_D, self.rhs, self.dmass_inv, self.dD)
        self.D.assign((1.0/3.0)*self.D + (2.0/3.0)*(self.D2 + self.dD))

        self._mass_solve(self.L1_u, self.rhs_u, self.umass_inv, self.du)
        self.u1.assign(self.u + self.du)

        self._mass_solve(self.L2_u, self.rhs_u, self.umass_inv, self.du)
        self.u2.assign(0.75*self.u + 0.25*(self.u1 + self.du))

        self._mass_solve(self.L3_u, self.rhs_u, self.umass_inv, self.du)
        self.u.assign((1.0/3.0)*self.u + (2.0/3.0)*(self.u2 + self.du))

    @PETSc.Log.EventDecorator("advection_step")
    def advection_SSPRK3(self):
        """Advection step: N SSPRK3 sub-steps, starting from the explicit
        step (second order) or from (un, Dn) (first order)."""
        if self.second_order:
            self.projector_adv.project()
            self.D.assign(self.Dhat)
        else:
            self.projector_broken.project()
            self.D.assign(self.Dn)

        for i in range(self.n_adv_cycles):
            self.advection_SSPRK3_single_step()

    @PETSc.Log.EventDecorator("projection_step")
    def projection_step(self):
        """Projection step; updates (un, Dn) and the derived fields."""
        self.solver_proj.solve()

        self.Dn.assign(self.Dnp1)
        self.un.assign(self.unp1)
        self.D_pert.assign(self.Dn - self.D0)
        self.ubar.assign(self.un)
        self.Dplusb.assign(self.Dn + self.b)

    @PETSc.Log.EventDecorator("compute_vorticity")
    def compute_vorticity(self):
        """Update the relative and potential vorticity from (un, Dn)."""
        self.qsolver.solve()
        self.pot_vort_solver.solve()

    @PETSc.Log.EventDecorator("compute_courant")
    def compute_Courant(self):
        """Update the Courant number field from un."""
        solve(self.courant_lhs_form == self.courant_rhs_form, self.Courant)

    @PETSc.Log.EventDecorator("compute_energies")
    def compute_phys_quant(self):
        """Return energy, potential enstrophy and RMS divergence of (un, Dn).

        The energy omits the time-independent term -g/2 int b^2 dx of the
        total energy. The potential enstrophy uses the potential vorticity
        from the last call of :meth:`compute_vorticity`.
        """
        g = Constant(G)
        energy = assemble((0.5*inner(self.un, self.un)*self.Dn + 0.5 * g * (self.Dn+self.b)**2)*dx)
        enstrophy = assemble(0.5*self.pot_qn*self.pot_qn*self.Dn*dx)
        # ||div u||_L2 / |Omega|^(1/2), see R_div in the paper
        div_rms = sqrt(assemble(div(self.un)*div(self.un)*dx)/self.area)
        return energy, enstrophy, div_rms
