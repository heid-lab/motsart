"""Dataclass definitions for all moTSart configuration objects.

These dataclasses are registered as Hydra-Zen config stores in the
respective module ``conf.py`` files.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EnvironmentConfig:
    """Environment configuration specifying paths, reaction data, and solver locations.

    Attributes:
        rxn_csv: Path to the reaction CSV file (columns: rxn_id, rxn_smiles).
        rxn_num: Index of the reaction in the CSV file.
        rxn_id: Specific reaction ID to process (overrides ``rxn_num``).
        solvent: Solvent name for implicit solvation (e.g. ``"water"``).
        orca_path: Path to the ORCA executable.
        xtb_path: Path to the xTB executable.
        results_folder: Output directory name for results.
        vdw_coef: Van der Waals coefficient for target bond distance in AFIR.
    """
    rxn_csv: str
    rxn_num: Optional[int]
    rxn_id: Optional[int]
    solvent: Optional[str]
    orca_path: str
    xtb_path: str
    results_folder: str = "results"
    vdw_coef: float = 0.9


# --------------------------- Complex Finder ---------------------------

@dataclass
class OptimizationConfig:
    """Configuration for the evolutionary algorithm used in complex finding.

    Controls conformer generation, distance-based optimization, energy-based
    optimization, and selection parameters.

    Attributes:
        complex_method: ``"default"`` for evolutionary optimization, ``"rtsp_goflow"`` for neural network.
        n_EA_rounds: Number of evolutionary algorithm rounds.
        seed: Random seed for conformer generation.
        n_confs: Number of initial RDKit conformers to generate.
        n_confs_after_product_similarity_filter: Conformers kept after product similarity filtering.
        dist_population_size: Population size for distance optimization EA.
        dist_generations: Number of EA generations for distance optimization.
        dist_elite_num: Number of elite members preserved per generation.
        dist_rotation_sigma: Standard deviation for rotation mutations (degrees).
        dist_translation_sigma: Standard deviation for translation mutations (Angstrom).
        product_similarity_coef: Coefficient for product similarity penalty term.
        forming_bond_vdw_coef: VdW radius multiplier for target forming bond length.
    """
    complex_method: str = "default"

    # EA rounds
    n_EA_rounds: int = 4

    # Conformer generation
    seed: int = 1
    n_confs: int = 1024
    n_confs_after_product_similarity_filter: int = 128

    # Distance optimization
    dist_population_size: int = 1024
    dist_generations: int = 64
    dist_elite_num: int = 10
    dist_rotation_sigma: float = 25.0
    dist_translation_sigma: float = .5
    product_similarity_coef: float = 10.0
    forming_bond_vdw_coef: float = 1.25

    # Reactant-complex energy lowest-energy filter
    n_rcs_to_screen_for_energy: int = 2


@dataclass
class AFIRPathGuesserParams:
    """Parameters for the AFIR (Artificial Force Induced Reaction) path guesser.

    Force constants are in atomic units [Hartree/Bohr^2].

    Attributes:
        fc_lower_bound: Lower bound for force constant binary search.
        fc_upper_bound: Upper bound for force constant binary search.
        fc_init_upper: Initial upper bound for the first binary search step.
        fc_binary_search_depth: Number of binary search iterations for optimal force constant.
        num_ts_for_validation: Number of TS candidates to forward to validation.
    """
    fc_lower_bound: float = 0.0
    fc_upper_bound: float = 3.0
    fc_init_upper: float = .01
    fc_binary_search_depth: int = 10
    num_ts_for_validation: int = 4


# --------------------------- Validator ---------------------------

@dataclass
class ValidatorConfig:
    """Configuration for TS validation (saddle-point optimization + IRC).

    Attributes:
        SP_maxcore: Maximum memory per core (MB) for single-point calculations.
        SP_nprocs: Number of parallel processes.
        SP_MaxIter: Maximum geometry optimization iterations.
        IRC_MaxIter: Maximum IRC iterations.
        skip_full_irc: If ``True``, only perform heuristic IRC via graphRC.
        path_guessers_to_validate: List of path guesser method names to validate.
    """
    SP_maxcore: int
    SP_nprocs: int
    SP_MaxIter: int
    IRC_MaxIter: int
    skip_full_irc: bool
    path_guessers_to_validate: List

# --------------------------- Learning ---------------------------

@dataclass
class ALConfig:
    """Configuration for learning training and inference.

    Attributes:
        train_batch_size: Number of samples per training batch.
        train_epochs: Number of training epochs per batch.
        learning_path_guesser: Path guesser method used for generating training data.
        n_conformers: Number of RDKit conformers to generate per molecule.
        num_samples: Number of TS samples to generate per conformer during inference.
    """
    train_batch_size: int
    train_epochs: int
    learning_path_guesser: str
    n_conformers: int = 32
    num_samples: int = 3
