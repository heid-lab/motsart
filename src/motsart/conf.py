"""Hydra-Zen configuration store for environment presets.

Registers named :class:`~motsart.conf_default.EnvironmentConfig` instances
(``cluster``, ``musica``, ``test``, ``local``) that specify
paths, reaction data, and solver locations for different execution contexts.
"""

from hydra_zen import store
from typing import Optional, List
from motsart.conf_default import EnvironmentConfig

env_store = store(group="env")

env_store(
    EnvironmentConfig(
        rxn_csv='data/cyclo32_small_rand.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/opt/ohpc/pub/dft/orca_6_1_0/orca',
        xtb_path='~/miniforge3/bin/xtb',
        solvent='water',
        results_folder='results_cluster',
    ),
    name="cluster"
)

env_store(
    EnvironmentConfig(
        rxn_csv='data/tetrazine/mapped_motsart.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/bin/orca',
        xtb_path='~/miniforge3/bin/xtb',
        solvent='water',
        results_folder='/results_tetrazine_cluster',
        vdw_coef=0.80,
    ),
    name="musica"
)

env_store(
    EnvironmentConfig(
        rxn_csv='data/test_rxns.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/Users/leo/Library/orca_6_1_0/orca',
        xtb_path='/Users/leo/miniconda3/envs/motsart/bin/xtb',
        solvent='water',
        results_folder='results_test',
    ),
    name="test"
)

env_store(
    EnvironmentConfig(
        rxn_csv='data/tetrazine/mapped_motsart.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/Users/leo/Library/orca_6_1_0/orca',
        xtb_path='/Users/leo/miniconda3/envs/motsart/bin/xtb',
        solvent='water',
        results_folder='results_tetrazine',
    ),
    name="local"
)

env_store(
    EnvironmentConfig(
        rxn_csv='data/tetrazine/prelim_h.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/Users/leo/Library/orca_6_1_0/orca',
        xtb_path='/Users/leo/miniconda3/envs/motsart/bin/xtb',
        solvent='water',
        results_folder='results_tetrazine',
        vdw_coef=0.80,
    ),
    name="tetrazine"
)

env_store(
    EnvironmentConfig(
        rxn_csv='data/azide/prelim_h.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/Users/leo/Library/orca_6_1_0/orca',
        xtb_path='/Users/leo/miniconda3/envs/motsart/bin/xtb',
        solvent='water',
        results_folder='results_azide',
        vdw_coef=0.80,
    ),
    name="azide"
)
