"""ASE CI-NEB path guesser.

Builds a Nudged Elastic Band between the reactant complex and its respective
product (IDPP-interpolated by default) and relaxes it with the climbing-image
NEB (CI-NEB) on the same OMol25/FAIRChem MLIP ("eSEN") that the MLIP validator
uses (see :mod:`motsart.validator.orca_validator.orca_external_tools.mlip_external`).
The highest-energy (climbing) image of the converged band is returned as the TS
guess.

All band images share the single cached FAIRChem calculator
(``allow_shared_calculator=True``); the MLIP is stateless per evaluation, so this
keeps a single model in memory while NEB evaluates every image.
"""

from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
from hydra_zen import zen, store

from ase.io import read as ase_read
from ase.mep import NEB
from ase.optimize import FIRE, BFGS, LBFGS

from motsart.path_guessers.base_reaction_path_guesser import BaseReactionPathGuesser
from motsart.complex_finder.utils import ReactionData, get_rxn_data, write_xyz_trajectory
from motsart.conf_default import EnvironmentConfig, NEBPathGuesserParams
from motsart.validator.orca_validator.orca_external_tools.mlip_external import (
    HARTREE_TO_EV,
    _build_atoms,
    get_calculator,
    resolve_device,
)


_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


class NebReactionPathGuesser(BaseReactionPathGuesser):
    """Climbing-image NEB TS guesser backed by an OMol25/FAIRChem MLIP."""

    def __init__(
        self,
        rxn_data: ReactionData,
        cfg: NEBPathGuesserParams,
        env: EnvironmentConfig,
        ts_method: str = 'neb',
        results_folder: str = 'results',
    ):
        super().__init__(rxn_data, ts_method, results_folder=results_folder)
        self.cfg = cfg
        self.env = env

    def _load_endpoint(self, xyz_file):
        """Read an xyz endpoint and attach charge/spin for the MLIP."""
        atoms = ase_read(str(xyz_file))
        return _build_atoms(
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            charge=self.rxn_data.charge,
            mult=self.cfg.mult,
        )

    def guess_reaction_path(self, reactive_complex_file: str, respective_product_file: str) -> Dict | None:
        """Run CI-NEB between the reactant complex and the product geometry.

        Returns:
            Dictionary containing the TS guess (``ts_atoms_N``/``ts_coords_N_3``),
            the full reactant->product band (``path_I_N_3``) and its energies in
            Hartree (``path_energies_I``) with the TS index (``ts_idx``).
            Returns ``None`` if the NEB search fails.
        """
        rc_name = Path(reactive_complex_file).name
        cfg = self.cfg
        try:
            reactant = self._load_endpoint(reactive_complex_file)
            product = self._load_endpoint(respective_product_file)
            if len(reactant) != len(product):
                print(f"Reactant/product atom counts differ for {rc_name}; skipping.")
                return None
            if cfg.n_images < 3:
                print(f"n_images must be >= 3 for an NEB band (got {cfg.n_images}); skipping {rc_name}.")
                return None

            opt_cls = _OPTIMIZERS.get(cfg.optimizer.upper())
            if opt_cls is None:
                raise ValueError(f"Unknown optimizer {cfg.optimizer!r}; expected one of {sorted(_OPTIMIZERS)}")

            device = resolve_device(self.env.mlip_device)
            calc = get_calculator(self.env.mlip_model, device, self.env.mlip_task_name)

            # Band: fixed reactant endpoint, n_images-2 movable middle images, fixed
            # product endpoint. Middle images start as reactant copies and are set by
            # interpolate(). charge/spin info and the shared calculator go on every image.
            images = [reactant] + [reactant.copy() for _ in range(cfg.n_images - 2)] + [product]
            for img in images:
                img.info["charge"] = int(self.rxn_data.charge)
                img.info["spin"] = int(cfg.mult)
                img.calc = calc

            neb = NEB(images, k=cfg.k, climb=cfg.climb, allow_shared_calculator=True)
            neb.interpolate(method=cfg.interp)

            opt = opt_cls(neb, logfile=None)
            opt.run(fmax=cfg.fmax, steps=cfg.steps)

            # FAIRChem energies are in eV -> Hartree.
            path_energies_I = np.array([img.get_potential_energy() for img in images]) / HARTREE_TO_EV
            path_coords_I_N_3 = np.asarray([img.get_positions() for img in images], dtype=float)
            ts_idx = int(np.argmax(path_energies_I))

            ts_atoms_N = list(images[ts_idx].get_chemical_symbols())
            write_xyz_trajectory(
                ts_atoms_N,
                path_coords_I_N_3,
                self.path_handler.ts_paths / rc_name,
                energies_I=path_energies_I,
            )

            return {
                'path_energies_I': path_energies_I,
                'ts_idx': ts_idx,
                'ts_atoms_N': ts_atoms_N,
                'ts_coords_N_3': path_coords_I_N_3[ts_idx],
                'path_I_N_3': path_coords_I_N_3,
            }
        except Exception as e:
            print(f"Error during CI-NEB path search for {rc_name}: {e}. Skipping this molecule.")
            return None


def neb_task(neb_cfg: NEBPathGuesserParams, env: EnvironmentConfig):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]

    rxn_data = get_rxn_data(rxn_id, rxn_smiles, solvent=env.solvent, r_or_p='r')
    path_guesser = NebReactionPathGuesser(
        rxn_data, cfg=neb_cfg, env=env, ts_method='neb', results_folder=env.results_folder
    )
    path_guesser.compute_rxn_path_and_save_data()


if __name__ == '__main__':
    import motsart.conf
    import motsart.path_guessers.conf
    store(
        neb_task,
        name="neb_root",
        hydra_defaults=[
            "_self_",
            {"neb_cfg": "test"},
            {"env": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(neb_task).hydra_main(
        config_name="neb_root",
        version_base="1.3"
    )
