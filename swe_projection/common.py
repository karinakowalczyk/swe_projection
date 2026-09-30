"""Physical constants and mesh setup shared by all test cases."""
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, DistributedMeshOverlapType

R0 = 6371220.      # Earth radius [m]
OMEGA = 7.292e-5   # rotation rate [1/s]
G = 9.8            # gravitational acceleration [m/s^2]
DAY = 86400.       # [s]


def sphere_mesh(ref_level):
    """Icosahedral sphere mesh of radius R0 with cubic coordinates, cell orientations set."""
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    mesh = IcosahedralSphereMesh(radius=R0, degree=3, refinement_level=ref_level,
                                 distribution_parameters=distribution_parameters)
    mesh.init_cell_orientations(SpatialCoordinate(mesh))
    return mesh
