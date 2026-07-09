"""ML-FSM path guesser using the Freezing String Method.

Runs the Freezing String Method (FSM) from the ``mlfsm`` package as a
double-ended TS search between the reactant complex and its respective product.
The string is grown and optimized on the same OMol25/FAIRChem MLIP ("eSEN") that
the MLIP validator uses (see :mod:`motsart.validator.orca_validator.orca_external_tools.mlip_external`),
so the path guesser and the validator agree on the underlying potential. The
highest-energy node of the converged string is returned as the TS guess.
"""

from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
from hydra_zen import zen, store

from ase.io import read as ase_read

from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer, InternalsOptimizer

from motsart.path_guessers.base_reaction_path_guesser import BaseReactionPathGuesser
from motsart.complex_finder.utils import ReactionData, get_rxn_data, write_xyz_trajectory
from motsart.conf_default import EnvironmentConfig, FSMPathGuesserParams
from motsart.validator.orca_validator.orca_external_tools.mlip_external import (
    HARTREE_TO_EV,
    _build_atoms,
    get_calculator,
    resolve_device,
)


class MlFsmReactionPathGuesser(BaseReactionPathGuesser):
    """Freezing String Method TS guesser backed by an OMol25/FAIRChem MLIP."""

    def __init__(
        self,
        rxn_data: ReactionData,
        cfg: FSMPathGuesserParams,
        env: EnvironmentConfig,
        ts_method: str = 'ml_fsm',
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

    def _make_optimizer(self, calc):
        cfg = self.cfg
        if cfg.optcoords == "cart":
            return CartesianOptimizer(calc, cfg.method, cfg.maxiter, cfg.maxls, cfg.dmax)
        if cfg.optcoords == "ric":
            return InternalsOptimizer(calc, cfg.method, cfg.maxiter, cfg.maxls, cfg.dmax)
        raise ValueError(f"Unknown optcoords {cfg.optcoords!r}; expected 'cart' or 'ric'")

    def guess_reaction_path(self, reactive_complex_file: str, respective_product_file: str) -> Dict | None:
        """Run FSM between the reactant complex and the product geometry.

        Returns:
            Dictionary containing the TS guess (``ts_atoms_N``/``ts_coords_N_3``),
            the full reactant->product string (``path_I_N_3``) and its energies in
            Hartree (``path_energies_I``) with the TS index (``ts_idx``).
            Returns ``None`` if the FSM search fails.
        """
        rc_name = Path(reactive_complex_file).name
        cfg = self.cfg
        try:
            reactant = self._load_endpoint(reactive_complex_file)
            product = self._load_endpoint(respective_product_file)
            if len(reactant) != len(product):
                print(f"Reactant/product atom counts differ for {rc_name}; skipping.")
                return None

            device = resolve_device(self.env.mlip_device)
            calc = get_calculator(self.env.mlip_model, device, self.env.mlip_task_name)
        except Exception as e:
            print(f"Error setting up ML-FSM path search for {rc_name}: {e}. Skipping this molecule.")
            return None

        # Try the configured interpolation scheme first, then fall back to
        # ``interp_fallback`` if it raises (e.g. RIC back-transformation failing
        # to converge on large reactant->product displacements, common for E2).
        interp_methods = [cfg.interp]
        if cfg.interp_fallback and cfg.interp_fallback not in interp_methods:
            interp_methods.append(cfg.interp_fallback)

        for attempt, interp_method in enumerate(interp_methods):
            try:
                result = self._run_fsm_string(reactant, product, calc, interp_method, rc_name)
            except Exception as e:
                print(f"ML-FSM path search for {rc_name} failed with interp={interp_method!r}: {e}")
                continue
            if result is None:
                continue
            if attempt > 0:
                print(f"ML-FSM recovered {rc_name} with fallback interp={interp_method!r}.")
            return result

        print(f"Error during ML-FSM path search for {rc_name}: all interpolation schemes "
              f"{interp_methods} failed. Skipping this molecule.")
        return None

    def _run_fsm_string(self, reactant, product, calc, interp_method, rc_name) -> Dict | None:
        """Build, grow and optimize one FSM string with ``interp_method``.

        Returns the TS-guess dict, or ``None`` if the string produced no evaluated
        nodes. Propagates exceptions (e.g. RIC back-transformation failures) so the
        caller can retry with the fallback interpolation scheme.
        """
        cfg = self.cfg
        string = FreezingString(
            reactant,
            product,
            nnodes_min=cfg.nnodes_min,
            interp_method=interp_method,
            ninterp=cfg.ninterp,
            stepsize=cfg.stepsize,
        )
        optimizer = self._make_optimizer(calc)

        while string.growing:
            string.grow()
            string.optimize(optimizer)

        # Combine the two ends into a single reactant->product path. ``p_string``
        # grows inward from the product end, so it is reversed here.
        all_atoms = string.r_string + string.p_string[::-1]
        all_energies = string.r_energy + string.p_energy[::-1]

        # Keep only evaluated nodes. FAIRChem energies are in eV -> Hartree.
        evaluated = [(a, e) for a, e in zip(all_atoms, all_energies) if e is not None]
        if len(evaluated) == 0:
            print(f"FSM produced no evaluated nodes for {rc_name}; skipping.")
            return None

        path_atoms = [a for a, _ in evaluated]
        path_energies_I = np.array([e for _, e in evaluated]) / HARTREE_TO_EV
        path_coords_I_N_3 = np.asarray([a.get_positions() for a in path_atoms], dtype=float)
        ts_idx = int(np.argmax(path_energies_I))

        ts_atoms_N = list(path_atoms[ts_idx].get_chemical_symbols())
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


def ml_fsm_task(fsm_cfg: FSMPathGuesserParams, env: EnvironmentConfig):
    df_smi = pd.read_csv(env.rxn_csv, sep=',', header=None)
    if env.rxn_id is not None:
        rxn_id = env.rxn_id
        matching_rows = df_smi[df_smi[0] == rxn_id]
        rxn_smiles = matching_rows[1].values[0]
    else:
        rxn_id = df_smi[0].values[env.rxn_num]
        rxn_smiles = df_smi[1].values[env.rxn_num]

    rxn_data = get_rxn_data(rxn_id, rxn_smiles, solvent=env.solvent, r_or_p='r')
    path_guesser = MlFsmReactionPathGuesser(
        rxn_data, cfg=fsm_cfg, env=env, ts_method='ml_fsm', results_folder=env.results_folder
    )
    path_guesser.compute_rxn_path_and_save_data()


if __name__ == '__main__':
    import motsart.conf
    import motsart.path_guessers.conf
    store(
        ml_fsm_task,
        name="ml_fsm_root",
        hydra_defaults=[
            "_self_",
            {"fsm_cfg": "test"},
            {"env": "test"},
        ]
    )
    store.add_to_hydra_store()
    zen(ml_fsm_task).hydra_main(
        config_name="ml_fsm_root",
        version_base="1.3"
    )
