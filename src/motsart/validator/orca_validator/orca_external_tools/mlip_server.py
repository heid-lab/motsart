#!/usr/bin/env python3
"""Persistent MLIP worker for the ORCA ExtOpt wrapper.

Loads the FAIRChem model once and serves energy + gradient requests over a
Unix domain socket, so ORCA's per-gradient calls (each of which spawns
``mlip_external.py``) don't reload the model every time.

It is auto-started on demand by ``mlip_external.py``. You can also start it
manually before a large sweep (e.g. on a GPU node) so the very first gradient
call is already warm:

    python -m motsart.validator.orca_validator.orca_external_tools.mlip_server \
        --model esen-sm-conserving-all-omol --socket /tmp/motsart_mlip.sock

The worker serves one request at a time (a single model instance) and exits
after ``--idle-timeout`` seconds without a new connection, so it cleans up after
a run finishes.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys

import numpy as np

from motsart.validator.orca_validator.orca_external_tools.mlip_external import (
    get_calculator,
    resolve_device,
    _build_atoms,
    _energy_grad,
    send_msg,
    recv_msg,
)


def serve(socket_path: str, model: str, device, task, idle_timeout: float) -> None:
    device = resolve_device(device)
    calc = get_calculator(model, device, task)  # load the model once

    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(128)
    srv.settimeout(idle_timeout)
    print(f"[mlip_server] ready: model={model} device={device} task={task} socket={socket_path}", flush=True)

    def _cleanup(*_):
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            print("[mlip_server] idle timeout reached, exiting", flush=True)
            _cleanup()

        with conn:
            try:
                req = recv_msg(conn)
                if req is None:
                    continue
                if req.get("cmd") == "ping":
                    send_msg(conn, {"ok": True, "pong": True})
                    continue
                atoms = _build_atoms(req["atoms"], np.asarray(req["coords"], dtype=float),
                                     req["charge"], req["mult"])
                nat, energy_eh, grad = _energy_grad(atoms, bool(req["dograd"]), calc)
                send_msg(conn, {"ok": True, "natoms": nat, "energy_eh": energy_eh, "grad": grad})
            except Exception as ex:
                try:
                    send_msg(conn, {"ok": False, "error": repr(ex)})
                except Exception:
                    pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local checkpoint path or FAIRChem registry name.")
    p.add_argument("--device", default=None, help="Torch device (default: CUDA if available, else CPU).")
    p.add_argument("--task", default="omol", help="FAIRChem task name ('' to omit).")
    p.add_argument("--socket", required=True, help="Unix socket path to listen on.")
    p.add_argument("--idle-timeout", type=float, default=1800.0,
                   help="Exit after this many seconds without a new connection.")
    a = p.parse_args()
    serve(a.socket, a.model, a.device, (a.task or None), a.idle_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
