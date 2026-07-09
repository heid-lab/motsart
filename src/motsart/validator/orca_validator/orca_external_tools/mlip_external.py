#!/usr/bin/env python3
"""ORCA ``otool_external`` wrapper exposing an OMol25 MLIP as a PES engine.

This script implements ORCA's external-tool (``! ExtOpt``) interface so that
ORCA's own optimizers (``OptTS``, ``Opt``, ``IRC``, ``NumFreq``) can be driven
by a FAIRChem machine-learned interatomic potential (default:
``esen-sm-conserving-all-omol`` from the OMol25 release) instead of GFN2-xTB or
DFT.

Persistent worker (no per-call model reload)
--------------------------------------------
ORCA spawns a new process for every energy/gradient evaluation. Reloading the
model each time would dominate runtime, so this wrapper talks to a long-lived
worker that loads the model once and serves energy + gradient over a Unix
socket (see :mod:`mlip_server`). On the first call the wrapper auto-starts the
worker (guarded by a file lock so concurrent ORCA displacement jobs don't race),
and every subsequent call is a thin client. If the worker path fails for any
reason it transparently falls back to loading the model in-process, so a run
never breaks. Set ``MOTSART_MLIP_NO_SERVER=1`` (or pass ``--no-server``) to force
the in-process path. The worker exits on its own after a period of inactivity.

How ORCA calls this script
---------------------------
ORCA invokes the wrapper once per evaluation as::

    mlip_external.py  <basename>_EXT.extinp.tmp  [Ext_Params ...]

The ``*.extinp.tmp`` file contains, one value per line: xyz filename, total
charge, spin multiplicity (2S+1), number of cores, do-gradient flag (0/1), and
an optional point-charge filename. The wrapper writes ``<basename>.engrad`` next
to the input file (energy in Hartree, gradient in Eh/Bohr, ordered A1x,A1y,A1z,
A2x,...), matching the official orca-external-tools reference exactly.

Charge / spin are forwarded to FAIRChem via ``atoms.info = {"charge": q,
"spin": m}`` where ``spin`` is the multiplicity (2S+1).

``--dummy`` replaces the MLIP with a trivial analytic potential so the full
ORCA <-> wrapper round-trip can be validated without a GPU/torch/fairchem.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# CODATA / ASE-consistent conversion constants.
HARTREE_TO_EV = 27.211386245988      # eV per Hartree
BOHR_TO_ANG = 0.52917721067          # Angstrom per Bohr

# Per-process calculator cache: {(model, device, task): FAIRChemCalculator}
_CALC_CACHE: dict = {}


def resolve_device(requested: Optional[str]) -> str:
    """Pick the compute device: honor an explicit request, else CUDA if available, else CPU."""
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# ORCA external-tool I/O (mirrors orca-external-tools/src/oet/core/misc.py)
# --------------------------------------------------------------------------- #
def read_orca_extinp(inputfile: Path) -> Tuple[str, int, int, int, bool, Optional[str]]:
    """Parse the ``*.extinp.tmp`` file ORCA passes to the external tool."""
    with open(inputfile, "r") as f:
        lines = [line.split(" ")[0].strip() for line in f.readlines() if line.strip()]

    xyz_filename = lines[0]
    charge = int(lines[1])
    multiplicity = int(lines[2])
    ncores = int(lines[3])
    flag = int(lines[4])
    if flag not in (0, 1):
        raise ValueError("do_gradient flag from ORCA input must be 0 or 1.")
    do_gradient = flag == 1
    pointcharge_filename = lines[5] if len(lines) >= 6 else None
    return xyz_filename, charge, multiplicity, ncores, do_gradient, pointcharge_filename


def write_engrad(filename: Path, nat: int, etot_eh: float, grad_eh_bohr: Optional[List[float]] = None) -> None:
    """Write the ORCA ``.engrad`` file (energy in Eh, gradient in Eh/Bohr)."""
    out = "#\n# Number of atoms\n#\n"
    out += f"{nat}\n"
    out += "#\n# Total energy [Eh]\n#\n"
    out += f"{etot_eh:.12e}\n"
    if grad_eh_bohr is not None and len(grad_eh_bohr) > 0:
        out += "#\n# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...\n#\n"
        out += "\n".join(f"{g: .12e}" for g in grad_eh_bohr) + "\n"
    with open(filename, "w") as f:
        f.write(out)


def engrad_path_for(input_file: Path, xyz_filename: str) -> Path:
    """ORCA expects ``<basename>.engrad`` (basename = xyz name minus ``.xyz``)
    in the same directory as the input file."""
    basename = Path(xyz_filename).name
    if basename.endswith(".xyz"):
        basename = basename[: -len(".xyz")]
    return input_file.parent / (basename + ".engrad")


# --------------------------------------------------------------------------- #
# MLIP model + compute core
# --------------------------------------------------------------------------- #
def get_calculator(model: str, device: str, task: Optional[str]):
    """Load (and cache) a FAIRChem calculator for the requested OMol25 model.

    ``model`` may be either a path to a local checkpoint (``*.pt``) or a
    FAIRChem registry name (downloaded from HuggingFace; requires gated access).
    """
    key = (model, device, task)
    if key in _CALC_CACHE:
        return _CALC_CACHE[key]

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    if Path(model).expanduser().is_file():
        predictor = load_predict_unit(path=str(Path(model).expanduser()), device=device)
    else:
        from fairchem.core import pretrained_mlip
        predictor = pretrained_mlip.get_predict_unit(model, device=device)

    try:
        calc = FAIRChemCalculator(predictor, task_name=task) if task else FAIRChemCalculator(predictor)
    except TypeError:
        calc = FAIRChemCalculator(predictor)

    _CALC_CACHE[key] = calc
    return calc


def _build_atoms(symbols, coords, charge: int, mult: int):
    from ase import Atoms
    atoms = Atoms(symbols=list(symbols), positions=np.asarray(coords, dtype=float))
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(mult)  # FAIRChem/OMol25 'spin' == multiplicity (2S+1)
    return atoms


def _energy_grad(atoms, dograd: bool, calc):
    """Compute energy (Eh) and, if requested, gradient (Eh/Bohr) for an ASE Atoms."""
    atoms.calc = calc
    energy_eh = float(atoms.get_potential_energy()) / HARTREE_TO_EV
    grad = None
    if dograd:
        forces_ev_ang = np.asarray(atoms.get_forces(), dtype=float)  # eV/Angstrom
        grad = ((-forces_ev_ang / HARTREE_TO_EV) * BOHR_TO_ANG).reshape(-1).tolist()
    return len(atoms), energy_eh, grad


def compute_mlip_direct(xyz_file: Path, charge: int, mult: int, dograd: bool,
                        model: str, device: str, task: Optional[str], ncores: int):
    """In-process fallback: load the model in this process and compute."""
    from ase.io import read

    if device == "cpu" and ncores and ncores > 0:
        try:
            import torch
            torch.set_num_threads(int(ncores))
        except Exception:
            pass

    atoms = read(str(xyz_file))
    atoms = _build_atoms(atoms.get_chemical_symbols(), atoms.get_positions(), charge, mult)
    return _energy_grad(atoms, dograd, get_calculator(model, device, task))


def compute_dummy(xyz_file: Path, dograd: bool):
    """Trivial analytic harmonic-to-origin potential for offline plumbing tests.

    E = 0.5 * k * sum_i |r_i|^2 (r in Bohr); grad_i = k * r_i. Conservative,
    GPU-free; NOT a physical PES.
    """
    from ase.io import read

    k = 0.01
    atoms = read(str(xyz_file))
    pos_bohr = atoms.get_positions() / BOHR_TO_ANG
    energy_eh = 0.5 * k * float(np.sum(pos_bohr ** 2))
    grad = (k * pos_bohr).reshape(-1).tolist() if dograd else None
    return len(atoms), energy_eh, grad


# --------------------------------------------------------------------------- #
# Persistent-worker socket protocol (client side; server in mlip_server.py)
# --------------------------------------------------------------------------- #
def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(data)) + data)


def _recv_n(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    hdr = _recv_n(sock, 8)
    if hdr is None:
        return None
    (n,) = struct.unpack(">Q", hdr)
    body = _recv_n(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def default_socket_path(model: str, task: Optional[str], device: str) -> str:
    """Stable per-(model,task,device) Unix socket path under the temp dir."""
    h = hashlib.md5(f"{model}|{task}|{device}".encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"motsart_mlip_{h}.sock")


def can_connect(socket_path: str) -> bool:
    if not os.path.exists(socket_path):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(socket_path)
        s.close()
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _file_lock(path: str):
    import fcntl
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


def ensure_server(socket_path: str, model: str, device: str, task: Optional[str],
                  idle_timeout: float = 1800.0, wait: float = 300.0) -> None:
    """Make sure a worker is listening on ``socket_path``; auto-start one if not.

    Uses a file lock so concurrent ORCA displacement jobs start at most one
    worker. Raises if the worker does not become ready within ``wait`` seconds.
    """
    if can_connect(socket_path):
        return
    with _file_lock(socket_path + ".lock"):
        if can_connect(socket_path):
            return
        if os.path.exists(socket_path):  # stale socket from a dead worker
            try:
                os.unlink(socket_path)
            except OSError:
                pass
        server_py = str(Path(__file__).resolve().parent / "mlip_server.py")
        cmd = [sys.executable, server_py, "--model", model, "--device", device,
               "--socket", socket_path, "--idle-timeout", str(idle_timeout)]
        if task:
            cmd += ["--task", task]
        logf = open(socket_path + ".log", "a")
        subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)
        deadline = time.time() + wait
        while time.time() < deadline:
            if can_connect(socket_path):
                return
            time.sleep(0.25)
    if not can_connect(socket_path):
        raise RuntimeError(f"MLIP worker not ready within {wait}s (see {socket_path}.log)")


def compute_via_server(socket_path: str, xyz_file: Path, charge: int, mult: int, dograd: bool):
    """Send a geometry to the worker and return (natoms, energy_eh, grad)."""
    from ase.io import read
    atoms = read(str(xyz_file))
    req = {
        "atoms": atoms.get_chemical_symbols(),
        "coords": atoms.get_positions().tolist(),
        "charge": int(charge),
        "mult": int(mult),
        "dograd": bool(dograd),
    }
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1800.0)
    s.connect(socket_path)
    try:
        send_msg(s, req)
        resp = recv_msg(s)
    finally:
        s.close()
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"worker error: {None if not resp else resp.get('error')}")
    return resp["natoms"], resp["energy_eh"], resp.get("grad")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORCA ExtOpt wrapper for an OMol25 MLIP.")
    p.add_argument("inputfile", help="ORCA-generated *.extinp.tmp file")
    p.add_argument("--model", default=os.environ.get("MOTSART_MLIP_MODEL", "esen-sm-conserving-all-omol"),
                   help="Path to a local checkpoint or a FAIRChem registry name.")
    p.add_argument("--device", default=os.environ.get("MOTSART_MLIP_DEVICE") or None,
                   help="Torch device override. Default: CUDA if available, else CPU.")
    p.add_argument("--task", default=os.environ.get("MOTSART_MLIP_TASK", "omol"),
                   help="FAIRChem task name (use '' / 'none' to omit).")
    p.add_argument("--socket", default=os.environ.get("MOTSART_MLIP_SOCKET") or None,
                   help="Worker socket path (default: derived from model/task/device).")
    p.add_argument("--no-server", action="store_true",
                   help="Skip the persistent worker; load the model in-process every call.")
    p.add_argument("--dummy", action="store_true",
                   help="Use an analytic potential (no torch/fairchem) for plumbing tests.")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        sys.stderr.write(f"[mlip_external] ignoring unrecognized args: {unknown}\n")
    if args.task.lower() in ("", "none", "null"):
        args.task = None
    return args


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    input_file = Path(args.inputfile).resolve()
    xyz_filename, charge, mult, ncores, dograd, _pc = read_orca_extinp(input_file)
    xyz_file = input_file.parent / xyz_filename
    if not xyz_file.is_file():
        xyz_file = Path(xyz_filename)

    via = "dummy"
    if args.dummy:
        nat, energy_eh, grad = compute_dummy(xyz_file, dograd)
    else:
        device = resolve_device(args.device)
        use_server = not args.no_server and os.environ.get("MOTSART_MLIP_NO_SERVER", "0") != "1"
        nat = energy_eh = grad = None
        if use_server:
            sock = args.socket or default_socket_path(args.model, args.task, device)
            try:
                ensure_server(sock, args.model, device, args.task)
                nat, energy_eh, grad = compute_via_server(sock, xyz_file, charge, mult, dograd)
                via = "worker"
            except Exception as ex:
                sys.stderr.write(f"[mlip_external] worker path failed ({ex!r}); loading model in-process\n")
                nat = None
        if nat is None:
            nat, energy_eh, grad = compute_mlip_direct(
                xyz_file, charge, mult, dograd, args.model, device, args.task, ncores
            )
            via = "in-process"

    write_engrad(engrad_path_for(input_file, xyz_filename), nat, energy_eh, grad)
    print(f"[mlip_external] E={energy_eh:.8f} Eh  grad={'yes' if grad else 'no'}  "
          f"natoms={nat}  via={via}  model={'dummy' if args.dummy else args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
