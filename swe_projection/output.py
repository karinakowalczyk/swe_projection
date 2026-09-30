"""Writing diagnostics text files safely in parallel.

The values passed in are global (e.g. results of ``assemble``), so they are
the same on every MPI rank; only rank 0 writes them.
"""


def start_diagnostics(filename, columns, comm):
    """Create ``filename`` with a header line naming the columns."""
    if comm.rank == 0:
        with open(filename, "w") as f:
            f.write("# " + " ".join(columns) + "\n")


def write_diagnostics(filename, values, comm):
    """Append one row of values to ``filename``."""
    if comm.rank == 0:
        with open(filename, "a") as f:
            f.write(" ".join(f"{v:.17g}" for v in values) + "\n")
