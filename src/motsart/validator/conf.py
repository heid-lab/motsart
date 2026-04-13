"""Hydra-Zen configuration store for validator parameters."""

from hydra_zen import store, builds
from motsart.validator.orca_validator.validator import GFN2XTBValidator, DFTValidator
from motsart.conf_default import ValidatorConfig


# ---------------------------------- Validator Configs ----------------------------------
validator_cfg_store = store(group="validator_cfg")

validator_cfg_store(
    ValidatorConfig(
        SP_maxcore=2048,
        SP_nprocs=20,
        SP_MaxIter=50,
        IRC_MaxIter=70,
        skip_full_irc=True,
        path_guessers_to_validate=['racer_ts'],
    ),
    name="cluster"
)

validator_cfg_store(
    ValidatorConfig(
        SP_maxcore=2048,
        SP_nprocs=1,
        SP_MaxIter=50,
        IRC_MaxIter=70,
        skip_full_irc=True,
        path_guessers_to_validate=['learning'],
    ),
    name="cluster_goflow"
)

validator_cfg_store(
    ValidatorConfig(
        SP_maxcore=2048,
        SP_nprocs=4,
        SP_MaxIter=70,
        IRC_MaxIter=70,
        skip_full_irc=True,
        path_guessers_to_validate=['racer_ts'],
    ),
    name="local"
)

validator_cfg_store(
    ValidatorConfig(
        SP_maxcore=2048,
        SP_nprocs=12,
        SP_MaxIter=25,
        IRC_MaxIter=50,
        skip_full_irc=True,
        path_guessers_to_validate=['afir'],
    ),
    name="test"
)

# ---------------------------------- Validators ----------------------------------

validator_store = store(group="validator")

validator_store(
    builds(GFN2XTBValidator, zen_partial=True), 
    name="xtb"
)

validator_store(
    builds(DFTValidator, zen_partial=True), 
    name="dft"
)