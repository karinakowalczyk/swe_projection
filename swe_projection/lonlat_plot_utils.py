"""Interpolation of fields on the sphere onto a lon-lat rectangle (for --latlon output).

A rectangle mesh in (lon, lat) is mapped onto the sphere by interpolating its
coordinates. The sphere field is interpolated onto the mapped mesh, and since
both meshes share the same degree-of-freedom layout, the values are copied
directly into a function on the flat rectangle.
"""
from functools import cache

from firedrake import *

from swe_projection.common import R0


def xyz_from_lonlatr(lon, lat, r):
    """Cartesian coordinates from longitude, latitude (in radians) and radius."""
    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)
    return x, y, z


@cache
def _latlon_spaces():
    """Build the lon-lat meshes on first use; returns (V on rectangle, V on mapped sphere mesh)."""
    eps = 1e-6  # keep away from the poles
    mesh_rect = RectangleMesh(nx=200, ny=200, Lx=pi, Ly=pi/2 - eps, originX=-pi, originY=-pi/2 + eps)
    CG = VectorFunctionSpace(mesh_rect, "CG", degree=3, dim=3)
    lon, lat = SpatialCoordinate(mesh_rect)
    coords = Function(CG).interpolate(as_vector(xyz_from_lonlatr(lon, lat, R0)))
    mesh_ll = Mesh(coords)
    return FunctionSpace(mesh_rect, "CG", degree=2), FunctionSpace(mesh_ll, "CG", degree=2)


def get_V_ll_rect():
    """Function space on the flat lon-lat rectangle, for the output functions."""
    return _latlon_spaces()[0]


def sphere_to_latlongrect(f, f_ll):
    """Interpolate scalar f from the sphere into f_ll (a function in get_V_ll_rect())."""
    f_3d = Function(_latlon_spaces()[1]).interpolate(f)
    f_ll.dat.data_wo[:] = f_3d.dat.data_ro[:]


def sphere_to_latlongrect_vec(VS, f, f_ll):
    """Interpolate |f| of vector f into f_ll; VS is a scalar space on the sphere mesh."""
    f_mag = Function(VS).interpolate(sqrt(dot(f, f)))
    sphere_to_latlongrect(f_mag, f_ll)
