"""PySCF-based TS validator using pysisyphus.

Performs saddle-point optimization (RSIRFOptimizer) and IRC validation
(EulerPC predictor-corrector) using PySCF as the electronic structure
backend, with optional GPU acceleration via GPU4PySCF.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import argparse
import pandas as pd
from ase.io import iread, read

from pysisyphus.calculators.PySCF import PySCF
from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.tsoptimizers.RSIRFOptimizer import RSIRFOptimizer
from pysisyphus.irc.EulerPC import EulerPC

from motsart.validator.base_validator import BaseValidator
from motsart.complex_finder.utils import (
    rdkit_mols_equal, 
    standardized_rdkit_mol_from_smiles, 
    get_rdkit_mol_from_xyz
)

HARTREE_TO_KCAL = 627.509

@dataclass
class PySCFValidatorParams:
    use_gpu: bool = False
    
    basis: str = "def2-SVP"
    functional: str = "b3lyp"
    
    SP_nprocs: int = 8
    SP_MaxIter: int = 20
    
    IRC_MaxIter: int = 60
    IRC_step_length: float = 0.1
    IRC_rms_grad_thresh: float = 0.001


class PySCFCalcWithSolvent(PySCF):
    """Pysisyphus PySCF calculator with solvent support."""
    
    def __init__(self, solvent=None, solvent_model="smd", **kwargs):
        super().__init__(**kwargs)
        self.solvent = solvent
        self.solvent_model = solvent_model.lower()
    
    def prepare_mf(self, mf):
        """Override to add solvent before GPU conversion."""
        if self.solvent:
            if self.solvent_model == "ddcosmo":
                mf = mf.DDCOSMO()
                mf.with_solvent.eps = self._get_dielectric(self.solvent)
            elif self.solvent_model == "smd":
                mf = mf.SMD()
                mf.with_solvent.solvent = self.solvent
            elif self.solvent_model == "pcm":
                mf = mf.PCM()
                mf.with_solvent.eps = self._get_dielectric(self.solvent)

        if self.use_gpu:
            mf = mf.to_gpu()
        
        return mf
    
    def _get_dielectric(self, solvent):
        dielectrics = {
            "water": 78.3553,
            "dmso": 46.826,
            "acetonitrile": 35.688,
            "methanol": 32.613,
            "ethanol": 24.852,
            "acetone": 20.493,
            "thf": 7.4257,
            "chloroform": 4.7113,
            "toluene": 2.3741,
        }
        return dielectrics.get(solvent.lower(), 78.3553)


class PySCFValidator(BaseValidator):
    """
    Validator using pysisyphus for TS optimization and IRC with PySCF.
    """
    def validate_single_ts(self, ts_guess_file: Path) -> Tuple[Dict, bool]:
        """
        Complete workflow for validating a single TS guess using pysisyphus.

        Args:
            ts_guess_file: Path to TS guess XYZ file

        Returns:
            tuple: (ts_results, irc_results) or (ts_results, None) if IRC not run
        """
        if not hasattr(self, 'path_handler') or self.path_handler is None:
            raise RuntimeError("path_handler not initialized. Call validate() instead of validate_single_ts() directly.")
        # Define output files
        ts_opt_dir = self.path_handler.ts_sp_opt_orca / ts_guess_file.stem
        ts_opt_dir.mkdir(parents=True, exist_ok=True)
        
        optimized_ts_xyz = ts_opt_dir / f"{ts_guess_file.stem}_ts_opt.xyz"
        irc_dir = self.path_handler.irc_orca / ts_guess_file.stem
        irc_dir.mkdir(parents=True, exist_ok=True)
        
        # TS optimization + frequency
        print(f"Running TS optimization on {ts_guess_file.name}...")
        ts_results, ts_is_sp = self.perform_ts_sp_opt(
            ts_guess_file, optimized_ts_xyz, ts_opt_dir
        )
        
        if not ts_is_sp:
            return ts_results, False
        
        # IRC validation
        print(f"Running IRC validation...")
        irc_results = self.perform_irc(optimized_ts_xyz, irc_dir)
        
        return ts_results, irc_results
    
    
    def create_pyscf_calculator(self):
        return PySCFCalcWithSolvent(
            basis=self.params.basis,
            xc=self.params.functional,
            charge=self.rxn_data.charge,
            mult=1,
            pal=self.params.SP_nprocs,
            use_gpu=self.params.use_gpu,
        )
    
    
    def read_xyz_to_geometry(self, xyz_file: Path):
        """Read XYZ file and create pysisyphus Geometry object."""
        atoms_obj = read(str(xyz_file))
        
        atoms = [atom.symbol for atom in atoms_obj]
        coords = atoms_obj.get_positions().flatten() * 1.88973  # Angstrom to Bohr
        
        calc = self.create_pyscf_calculator()
        geom = Geometry(atoms, coords, coord_type="cart")
        geom.set_calculator(calc)
        
        return geom
    
    
    def perform_ts_sp_opt(self, ts_guess_file: Path, output_xyz: Path, work_dir: Path) -> Tuple[Dict, bool]:
        """
        Perform TS optimization and frequency calculation using pysisyphus.

        Args:
            ts_guess_file: Input TS guess XYZ file.
            output_xyz: Output optimized TS XYZ file.
            work_dir: Working directory for calculation.

        Returns:
            Tuple of (results_dict, is_valid_ts).
        """
        geom = self.read_xyz_to_geometry(ts_guess_file)
        
        # Set up TS optimizer (RSIRFO - Restricted Step Image Function Optimizer)
        ts_opt = RSIRFOptimizer(
            geom,
            thresh="gau",  # Gaussian convergence criteria
            max_cycles=self.cfg.SP_MaxIter,
            hessian_recalc=5,
            trust_radius=0.3,
            trust_max=0.5,
            dump=True,
            out_dir=str(work_dir)
        )            
        ts_opt.run()
        
        # Get optimized geometry
        opt_geom = ts_opt.geometry
        converged = ts_opt.is_converged
        
        # Calculate Hessian at optimized geometry
        print("Calculating Hessian at optimized TS...")
        hessian = opt_geom.cart_hessian
        
        # Mass-weight the Hessian and diagonalize
        mass_weighted_hess = opt_geom.mass_weigh_hessian(hessian)
        eigvals, eigvecs = np.linalg.eigh(mass_weighted_hess)
        
        # Convert eigenvalues to frequencies (cm^-1)
        # omega = sqrt(k/m) but eigenvalues are already mass-weighted
        # Convert from atomic units to cm^-1
        AU_TO_INVCM = 5140.48714
        frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * AU_TO_INVCM
        
        # Count negative (imaginary) frequencies (consistent with ORCA validator)
        neg_imag_cnt = np.sum(frequencies < 0.0)
        
        # Save optimized geometry
        opt_coords_ang = opt_geom.coords.reshape(-1, 3) / 1.88973  # Bohr to Angstrom
        self.write_xyz(output_xyz, opt_geom.atoms, opt_coords_ang)
        
        results = {
            'converged': converged,
            'neg_imag_cnt': neg_imag_cnt,
            'energy_kcal': opt_geom.energy * HARTREE_TO_KCAL,
            'frequencies': frequencies,
            'optimized_geometry': opt_coords_ang,
        }
        
        is_valid_ts = (neg_imag_cnt == 1) # and converged. TODO: adapt so that max(|step|) and rms(step) also are in converged regime. Check why they currently aren't!
        
        if not is_valid_ts:
            print(f"Warning: Found {neg_imag_cnt} imaginary frequencies, converged: {converged}")
        
        return results, is_valid_ts

    
    def perform_irc(self, optimized_ts_xyz: Path, irc_dir: Path) -> bool:
        """
        Perform IRC validation using pysisyphus.

        Args:
            optimized_ts_xyz: Path to optimized TS XYZ file.
            irc_dir: Directory for IRC calculation.

        Returns:
            Whether reactant and product were recovered.
        """
        # Read TS geometry
        geom = self.read_xyz_to_geometry(optimized_ts_xyz)
        
        # Set up IRC calculator (EulerPC - Euler Predictor-Corrector)
        irc = EulerPC(
            geom,
            step_length=self.cfg.IRC_step_length,
            max_cycles=self.cfg.IRC_MaxIter,
            rms_grad_thresh=self.cfg.IRC_rms_grad_thresh,
            forward=True,
            backward=True,
            hessian_init='calc',
            out_dir=str(irc_dir),
            #dump=True,
        )            
        irc.run()
        
        # Get IRC trajectories
        forward_coords = irc.forward_coords_list if hasattr(irc, 'forward_coords_list') else []
        backward_coords = irc.backward_coords_list if hasattr(irc, 'backward_coords_list') else []
        
        # Save full IRC trajectory
        irc_traj_file = irc_dir / "irc_full_trj.xyz"
        self.save_irc_trajectory(irc_traj_file, geom.atoms, forward_coords, backward_coords)
        
        return self.parse_irc_results(irc_traj_file)
    
    
    def save_irc_trajectory(self, output_file: Path, atoms: List[str], 
                           forward_coords: List, backward_coords: List):
        """Save IRC trajectory to XYZ file."""
        with open(output_file, 'w') as f:
            # Backward trajectory (reversed)
            for coords in reversed(backward_coords):
                coords_ang = coords.reshape(-1, 3) / 1.88973  # Bohr to Angstrom
                f.write(f"{len(atoms)}\n")
                f.write("IRC backward\n")
                for atom, coord in zip(atoms, coords_ang):
                    f.write(f"{atom:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")
            
            # Forward trajectory
            for coords in forward_coords:
                coords_ang = coords.reshape(-1, 3) / 1.88973  # Bohr to Angstrom
                f.write(f"{len(atoms)}\n")
                f.write("IRC forward\n")
                for atom, coord in zip(atoms, coords_ang):
                    f.write(f"{atom:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")
    
    
    def write_xyz(self, output_file: Path, atoms: List[str], coords_ang: np.ndarray):
        """Write XYZ file."""
        with open(output_file, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write("Optimized structure\n")
            for atom, coord in zip(atoms, coords_ang):
                f.write(f"{atom:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TS validation with pysisyphus')
    parser.add_argument('--rxn_csv', type=str, required=True, help='Path to CSV file with rxn data')
    parser.add_argument('--rxn_id', type=int, required=True, help='Reaction index to process')
    parser.add_argument('--solvent', type=str, default=None, help='Solvent')
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    
    df_smi = pd.read_csv(args.rxn_csv, sep=',', header=None)
    rxn_id = df_smi[0].values[args.rxn_id]
    rxn_smiles = df_smi[1].values[args.rxn_id]
    
    params = PySCFValidatorParams(
        use_gpu = not args.local,
    )
    
    mode = "CPU" if args.local else "GPU"
    print(f"Running GPU4PySCF validation on rxn {rxn_id} with B3LYP/def2-SVP, on {mode}")
    
    validator = PySCFValidator(rxn_id, rxn_smiles, args.solvent, params)
    validator.validate()
