"""Writing diagnostics text files and progress messages safely in parallel.

The values passed in are global (e.g. results of ``assemble``), so they are
the same on every MPI rank; only rank 0 writes them.
"""
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from firedrake.petsc import PETSc

from swe_projection.common import DAY


def save_run_info(out_dir, args, comm):
    """Write run_info.json to out_dir: the options used, the command line,
    the number of MPI processes, the git commit and the package versions."""
    if comm.rank != 0:
        return
    repo = Path(__file__).resolve().parent.parent

    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=repo, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    modified = git("diff", "HEAD", "--name-only")
    info = {
        "options": vars(args),
        "command": " ".join(sys.argv),
        "mpi_processes": comm.size,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        # files that differ from git_commit; empty if the code is exactly that commit
        "git_modified_files": modified.split() if modified is not None else None,
        "versions": {p: version(p) for p in ("firedrake", "petsc4py")},
    }
    with open(Path(out_dir) / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


class Progress:
    """Print the test configuration at the start, then progress at each output time."""

    def __init__(self, name, T):
        self.name = name
        self.T = T
        self.start = time.perf_counter()
        self.loop_start = None
        _print(f"\n=== START {name}, until t = {T/DAY:g} d")

    def update(self, t):
        """Print the simulation time reached, wall-clock time and estimated remaining time."""
        now = time.perf_counter()
        if self.loop_start is None:  # first call at t = 0, after the setup
            self.loop_start = now
            _print(f"[{self.name}] setup done after {_fmt(now - self.start)}")
            return
        remaining = (now - self.loop_start)*(self.T - t)/t
        _print(f"[{self.name}] t = {t/DAY:.2f} d of {self.T/DAY:g} d "
               f"({100*t/self.T:.0f}%), elapsed {_fmt(now - self.start)}, "
               f"remaining ~{_fmt(remaining)}")

    def step(self, t):
        """Print the simulation time after a time step."""
        _print(f"t =  {t}")

    def finish(self):
        _print(f"=== DONE {self.name} after {_fmt(time.perf_counter() - self.start)}\n")


def _print(message):
    """Print on rank 0 and flush at once, so the line shows up in a log file immediately."""
    if PETSc.COMM_WORLD.rank == 0:
        print(message, flush=True)


def _fmt(seconds):
    """Format a duration as h:mm:ss."""
    return str(timedelta(seconds=round(seconds)))


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
