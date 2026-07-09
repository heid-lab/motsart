"""Hydra-Zen configuration store for path-guesser parameters."""

from hydra_zen import store
from motsart.conf_default import FSMPathGuesserParams, NEBPathGuesserParams


# ---------------------------------- ML-FSM Configs ----------------------------------
fsm_store = store(group="fsm_cfg")

fsm_store(
    FSMPathGuesserParams(maxiter=4),
    name="base"
)

fsm_store(
    FSMPathGuesserParams(
        nnodes_min=10,
        ninterp=50,
        maxiter=1,
    ),
    name="test"
)

fsm_store(
    FSMPathGuesserParams(
        nnodes_min=18,
        ninterp=50,
        maxiter=2,
    ),
    name="local"
)


# ---------------------------------- CI-NEB Configs ----------------------------------
neb_store = store(group="neb_cfg")

neb_store(
    NEBPathGuesserParams(),
    name="base"
)

neb_store(
    NEBPathGuesserParams(
        n_images=7,
        steps=50,
    ),
    name="test"
)

neb_store(
    NEBPathGuesserParams(
        n_images=10,
        steps=200,
    ),
    name="local"
)
