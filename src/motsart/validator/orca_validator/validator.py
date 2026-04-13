"""ORCA-based TS validators (GFN2-xTB and DFT).

Implements saddle-point optimization, frequency analysis, and IRC
validation through ORCA. Includes a fast heuristic IRC via
graphRC vibrational analysis.
"""

from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import pandas as pd
from ase.io import read
import os
from dataclasses import dataclass
from abc import abstractmethod
from graphrc import run_vib_analysis

from .utils import OrcaParser, run_orca, SOLVENT_BLOCK, DFT_INP_TEMPLATE_IRC, DFT_INP_TEMPLATE_TS_FREQ, XTB_INP_TEMPLATE_TS_FREQ, XTB_INP_TEMPLATE_IRC
from motsart.validator.base_validator import BaseValidator
from motsart.complex_finder.utils import rdkit_mols_equal, standardized_rdkit_mol_from_smiles, get_rdkit_mol_from_xyz
from motsart.common import PathHandler
from motsart.conf_default import ValidatorConfig, EnvironmentConfig

# File extensions to keep after ORCA calculations
ORCA_EXTENSIONS_TO_KEEP = {'.xyz', '.out', '.inp'}


class ORCAValidator(BaseValidator):
    def cleanup_orca_temp_files(self, inp_files: List[Path]):
        """Remove temporary ORCA files created from the given input files.

        Keeps .xyz, .out, and .inp files, deletes all other ORCA-generated files
        (e.g., .gbw, .hess, .prop, .densities, .opt, .cpcm, .engrad, .tmp, etc.).
        """
        for inp_file in inp_files:
            directory = inp_file.parent
            stem = inp_file.stem

            for file in directory.glob(f"{stem}*"):
                if file.is_file() and file.suffix not in ORCA_EXTENSIONS_TO_KEEP:
                    try:
                        file.unlink()
                    except OSError:
                        pass

    def validate_single_ts(self, ts_guess_file: Path) -> Tuple[Dict, bool]:
        """
        Complete workflow for validating a single TS guess.
        
        Args:
            ts_guess_file: Path to TS guess XYZ file
        
        Returns:
            tuple: (ts_results, irc_results) or (ts_results, None) if IRC not run
        """
        # --------------- Define filenames ---------------
        # Saddle-point optimization input
        sp_opt_inp_file = self.path_handler.ts_sp_opt_orca / ts_guess_file.with_suffix('.ts_freq.inp').name
        sp_opt_out_file = self.path_handler.ts_sp_opt_orca / ts_guess_file.with_suffix('.out').name
        # IRC validation input
        optimized_ts_xyz_file = sp_opt_inp_file.with_suffix('.xyz')
        hess_file = sp_opt_inp_file.with_suffix('.hess')
        irc_inp_file = self.path_handler.irc_orca / ts_guess_file.with_suffix('.irc.inp').name
        irc_out_file = self.path_handler.irc_orca / ts_guess_file.with_suffix('.out').name
        
        # --------------- TS optimization + frequency ---------------
        ts_results, ts_is_sp = self.perform_ts_sp_opt(ts_guess_file, sp_opt_inp_file, sp_opt_out_file)
        if not ts_is_sp:
            self.cleanup_orca_temp_files([sp_opt_inp_file])
            return ts_results, False
        
        # Load optimized xyz to numpy array
        opt_xyz_N_3 = read(optimized_ts_xyz_file).get_positions()
        assert list(opt_xyz_N_3.shape) == [len(self.rxn_data.atoms_N), 3]
        ts_results['opt_xyz_N_3'] = opt_xyz_N_3
        
        # --------------- IRC validation (only if TS is valid) ---------------
        irc_results = self.perform_irc(irc_inp_file, irc_out_file, sp_opt_out_file, optimized_ts_xyz_file, hess_file)

        # --------------- Cleanup temporary ORCA files ---------------
        self.cleanup_orca_temp_files([sp_opt_inp_file, irc_inp_file])

        return ts_results, irc_results

    
    @abstractmethod
    def get_orca_input_irc(self, ts_xyzfile: str, hess_file: str):
        """Generate ORCA input for IRC Validation."""
        raise NotImplementedError("IRC input template not implemented")
    
    @abstractmethod
    def get_orca_input_ts_freq(self, ts_xyzfile: str):
        """Generate ORCA input for TS optimization and frequency calculation."""
        raise NotImplementedError("TS Opt input template not implemented")

    def are_bond_changes_equal(self, irc_bond_changes_B, rxn_bond_changes_B):
        def canon_bond(b, order_mn):
            i, j = b
            if order_mn:
                i = self.rxn_data.mn_order[i]
                j = self.rxn_data.mn_order[j]
            return (i, j) if i <= j else (j, i)
        def canon_set(bonds, order_mn):
            return {canon_bond(b, order_mn) for b in bonds}

        irc_set = canon_set(irc_bond_changes_B, order_mn=False)
        rxn_set = canon_set(rxn_bond_changes_B, order_mn=True)

        missing = rxn_set - irc_set   # expected from reaction, not found by graphRC
        extra = irc_set - rxn_set     # found by graphRC, not expected from reaction
        
        matches = (len(missing) == 0)# and len(extra) == 0)
        if len(extra) > 0:
            print(f'Found by graphRC, not expected from reaction: {extra}')

        return matches, missing, extra

    def run_graph_rc_irc_heuristic(self, sp_opt_out_file):
        try:
            res = run_vib_analysis(
                input_file=sp_opt_out_file,
                charge=self.rxn_data.charge,
            )
            if res is None or 'vibrational' not in res:
                print(f"graphRC returned invalid result for {sp_opt_out_file}")
                return False
            vib = res['vibrational']
            if 'bond_changes' not in vib:
                print(f"graphRC result missing 'bond_changes' for {sp_opt_out_file}")
                return False
            irc_bond_changes = set(vib['bond_changes'].keys())
        except Exception as e:
            print(f"graphRC failed for {sp_opt_out_file}: {e}")
            return False

        rxn_bond_changes = self.rxn_data.formed_bonds_mn_Bf | self.rxn_data.broken_bonds_mn_Bf

        matches, missing, extra = self.are_bond_changes_equal(irc_bond_changes, rxn_bond_changes)
        if not matches:
            print(f"IRC: graphRC vs rxn mismatch; missing={missing}, extra={extra}")
        return matches
    
    def perform_irc(self, irc_inp_file: Path, irc_out_file: Path, sp_opt_out_file: Path, optimized_ts_xyz_file: Path, hess_file: Path) -> bool:
        if self.cfg.skip_full_irc:
            irc_results = self.run_graph_rc_irc_heuristic(sp_opt_out_file)
        else:
            print(f"Running IRC validation...")
            # Get optimized TS geometry (ORCA writes it to .xyz file)
            irc_input = self.get_orca_input_irc(str(optimized_ts_xyz_file), str(hess_file))
            irc_inp_file.write_text(irc_input)
            
            # Copy Hessian file for IRC (ORCA creates .hess file)
            run_orca(irc_inp_file, irc_out_file, self.env.orca_path)

            # Step 4: Parse IRC results
            irc_traj_file = irc_inp_file.parent / f'{irc_inp_file.stem}_IRC_Full_trj.xyz' # orca-dft
            if not irc_traj_file.exists():
                irc_traj_file = irc_out_file.with_suffix('.irc_trj.xyz') # orca-xtb
            irc_results = self.parse_irc_results(irc_traj_file)
        
        return irc_results

    
    def perform_ts_sp_opt(self, ts_guess_file: Path, sp_opt_inp_file: Path, sp_opt_out_file: Path):
        # Create orca input and run orca
        orca_input_txt = self.get_orca_input_ts_freq(str(ts_guess_file))
        sp_opt_inp_file.write_text(orca_input_txt)

        run_orca(sp_opt_inp_file, sp_opt_out_file, self.env.orca_path)
        
        # Parse results
        hess_file = sp_opt_inp_file.with_suffix('.hess')
        orca_parser = OrcaParser(sp_opt_out_file, hess_file)
        ts_is_sp = (orca_parser.results['neg_imag_cnt'] == 1) and orca_parser.results['converged']
        
        if not ts_is_sp:
            print(f"Warning: Found {orca_parser.results['neg_imag_cnt']} imag freqs, SP opt converged: {orca_parser.results['converged']} skipping IRC")
        
        return orca_parser.results, ts_is_sp
    

class DFTValidator(ORCAValidator):
    def get_orca_input_irc(self, ts_xyzfile: str, hess_file: str):
        solvent_block = SOLVENT_BLOCK.format(SOLVENT=self.rxn_data.solvent) if self.rxn_data.solvent else ""
        return DFT_INP_TEMPLATE_IRC.format(
            NPROCS=self.cfg.SP_nprocs,
            MAXCORE=self.cfg.SP_maxcore,
            MAXITER=self.cfg.IRC_MaxIter,
            CHARGE=self.rxn_data.charge,
            MULT=1,
            ts_XYZFILE=ts_xyzfile,
            HESS_FILE=hess_file,
            SOLVENT_BLOCK=solvent_block
        )

    def get_orca_input_ts_freq(self, ts_xyzfile: str):
        solvent_block = SOLVENT_BLOCK.format(SOLVENT=self.rxn_data.solvent) if self.rxn_data.solvent else ""
        return DFT_INP_TEMPLATE_TS_FREQ.format(
            NPROCS=self.cfg.SP_nprocs,
            MAXCORE=self.cfg.SP_maxcore,
            MAXITER=self.cfg.SP_MaxIter,
            CHARGE=self.rxn_data.charge,
            MULT=1,
            ts_XYZFILE=ts_xyzfile,
            SOLVENT_BLOCK=solvent_block
        )


class GFN2XTBValidator(ORCAValidator):
    def __init__(self, rxn_id: str, rxn_smiles: str, cfg: ValidatorConfig, env: EnvironmentConfig):
        super().__init__(rxn_id, rxn_smiles, cfg, env)
        os.environ['XTBEXE'] = env.xtb_path

    def get_orca_input_irc(self, ts_xyzfile: str, hess_file: str):
        solvent_block = f'ALPB({self.rxn_data.solvent}) ' if self.rxn_data.solvent else ''
        return XTB_INP_TEMPLATE_IRC.format(
            NPROCS=self.cfg.SP_nprocs,
            MAXCORE=self.cfg.SP_maxcore,
            MAXITER=self.cfg.IRC_MaxIter,
            CHARGE=self.rxn_data.charge,
            MULT=1,
            SOLVENT_BLOCK=solvent_block,
            ts_XYZFILE=ts_xyzfile,
            HESS_FILE=hess_file,
            XTB_PATH=self.env.xtb_path,
            PROJ_ROOT_DIR=self.path_handler.project_root_dir,
        )

    def get_orca_input_ts_freq(self, ts_xyzfile):
        solvent_block = f'ALPB({self.rxn_data.solvent}) ' if self.rxn_data.solvent else ''
        return XTB_INP_TEMPLATE_TS_FREQ.format(
            NPROCS=self.cfg.SP_nprocs,
            MAXCORE=self.cfg.SP_maxcore,
            MAXITER=self.cfg.SP_MaxIter,
            CHARGE=self.rxn_data.charge,
            MULT=1,
            SOLVENT_BLOCK=solvent_block,
            ts_XYZFILE=ts_xyzfile,
            XTB_PATH=self.env.xtb_path,
            PROJ_ROOT_DIR=self.path_handler.project_root_dir,
        )


class GXTBValidator(ORCAValidator):
    pass
