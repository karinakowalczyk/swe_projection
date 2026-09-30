"""Initial conditions for the Galewsky test (Galewsky et al. 2004)."""
from math import pi

import firedrake as fd
import numpy as np
from scipy import integrate
from ufl.domain import extract_unique_domain

from swe_projection.common import R0 as radius, OMEGA as omega

# Jet
umax = 80.
theta0 = pi/7.
theta1 = pi/2. - theta0
en = np.exp(-4./((theta1-theta0)**2))

# Mean depth
h0 = 10e3
H0 = fd.Constant(h0)

# Perturbation
alpha = fd.Constant(1/3.)
beta = fd.Constant(1/15.)
hhat = fd.Constant(120)
theta2 = fd.Constant(pi/4.)

gravity = 9.80616


def cart_to_sphere_coords(x, y, z):
    """Return latitude theta and longitude lambda of Cartesian coordinates."""
    r = fd.sqrt(x*x + y*y + z*z)
    zr = z/r
    zr_corr = fd.min_value(fd.max_value(zr, -1), 1)  # avoid roundoff errors at the poles
    theta = fd.asin(zr_corr)
    lambda0 = fd.atan2(y, x)
    return theta, lambda0


def sphere_to_cart_vector(x, y, z, uzonal, umerid):
    """Return the Cartesian vector with zonal and meridional components uzonal, umerid."""
    theta, lamda = cart_to_sphere_coords(x, y, z)
    cart_u_expr = -uzonal*fd.sin(lamda) - umerid*fd.sin(theta)*fd.cos(lamda)
    cart_v_expr = uzonal*fd.cos(lamda) - umerid*fd.sin(theta)*fd.sin(lamda)
    cart_w_expr = umerid*fd.cos(theta)
    return fd.as_vector((cart_u_expr, cart_v_expr, cart_w_expr))


def zonal_velocity_expression(theta, exp):
    """Zonal jet velocity for theta0 < theta < theta1; exp is fd.exp or np.exp."""
    return (umax/en)*exp(1./((theta - theta0)*(theta - theta1)))


def velocity_expression(x, y, z):
    """Initial velocity: the zonal jet, zero outside theta0 < theta < theta1."""
    theta, lamda = cart_to_sphere_coords(x, y, z)
    uzonal_expr = zonal_velocity_expression(theta, fd.exp)
    uzonal = fd.conditional(fd.ge(theta, theta0),
                            fd.conditional(fd.le(theta, theta1),
                                           uzonal_expr, 0.), 0.)
    umeridional = 0.
    return sphere_to_cart_vector(x, y, z, uzonal, umeridional)


def depth_integrand(theta):
    """Integrand of the balance equation for the depth, for theta0 <= theta <= theta1.

    Works on numpy arrays.
    """
    f = 2*omega*np.sin(theta)
    uzonal = zonal_velocity_expression(theta, np.exp)
    return uzonal*(f + np.tan(theta)*uzonal/radius)


def depth_calculation(x):
    """Balanced depth, up to a constant, at the points x (array of shape (n, 3))."""
    depth = np.zeros(len(x))
    angles = np.zeros(len(x))

    # The integral is 0 for theta <= theta0 and constant for theta >= theta1,
    # so only points in between need a quadrature. Fixed quadrature with 64
    # points gives absolute errors below 1e-13 for a quantity of order 1e-3.
    poledepth, _ = integrate.fixed_quad(depth_integrand, theta0, theta1, n=64)
    poledepth *= -radius/gravity

    angles[:] = np.arcsin(x[:, 2]/radius)

    for ii in range(len(x)):
        if angles[ii] <= theta0:
            depth[ii] = 0.0
        elif angles[ii] >= theta1:
            depth[ii] = poledepth
        else:
            v, _ = integrate.fixed_quad(depth_integrand, theta0, angles[ii], n=64)
            depth[ii] = -(radius/gravity)*v
    return depth


def depth_perturbation(theta, lamda, V):
    """The perturbation h' as a Function in V."""
    latitude_profile = fd.exp(-((theta2 - theta)/beta)**2)
    longitude_profile = fd.exp(-(lamda/alpha)**2)
    pole_scaling = fd.cos(theta)
    return fd.Function(V).interpolate(hhat*pole_scaling*longitude_profile*latitude_profile)


def depth_expression(x, y, z):
    """Return the initial depth D_0 + h' (mean depth H0) and the perturbation h', in DG1."""
    mesh = extract_unique_domain(x)
    V = fd.FunctionSpace(mesh, "DG", 1)  # depth space of the model
    W = fd.VectorFunctionSpace(mesh, V.ufl_element())

    # Balanced depth at the DG1 nodes
    coords = fd.Function(W).interpolate(mesh.coordinates)
    h = fd.Function(V)
    h.dat.data[:] = depth_calculation(coords.dat.data_ro)

    # Shift to mean depth H0 and add the perturbation
    cells = fd.Function(V).assign(fd.Constant(1))
    area = fd.assemble(cells*fd.dx)
    hmean = fd.assemble(h*fd.dx)/area
    hpert = depth_perturbation(*cart_to_sphere_coords(x, y, z), V)
    h += H0 - hmean + hpert

    return h, hpert
