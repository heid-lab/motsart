"""ORCA input templates and output parsers for TS validation.

Contains input file templates for GFN2-xTB and DFT saddle-point
optimization and IRC calculations, plus :func:`run_orca` and
:class:`OrcaParser` for executing and parsing results.
"""

from typing import Dict
from pathlib import Path
import re
import subprocess
import numpy as np
import os


GXTB_INP_TEMPLATE_TS_FREQ = r"""! ExtOpt NUMFREQ

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

%method
  ProgExt "{PROJ_ROOT_DIR}/src/motsart/validator/orca_validator/orca_external_tools/gxtb.py"
  Ext_Params "-x {XTB_PATH} --gfn 2 {SOLVENT_BLOCK} -c 0 --grad"
end

%geom
  MaxIter {MAXITER}
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""

GXTB_INP_TEMPLATE_IRC = r"""! ExtOpt IRC

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

%method
  ProgExt "{PROJ_ROOT_DIR}/src/motsart/validator/orca_validator/orca_external_tools/gxtb.py"
  Ext_Params "-x {XTB_PATH} --gfn 2 {SOLVENT_BLOCK}-c 0 --grad"
end

%irc
  MaxIter {MAXITER}
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""



XTB_INP_TEMPLATE_TS_FREQ = r"""! GFN2-XTB {SOLVENT_BLOCK}OptTS Freq

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

%geom
  MaxIter {MAXITER}
  Recalc_Hess 5
  Calc_Hess true
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""

XTB_INP_TEMPLATE_IRC = r"""! GFN2-XTB {SOLVENT_BLOCK}IRC

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

%irc
  MaxIter {MAXITER}
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""


# Template for saddle point optimization + frequency check
DFT_INP_TEMPLATE_TS_FREQ = r"""! OptTS Freq
! B3LYP D4 def2-TZVP def2/J RIJCOSX

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

{SOLVENT_BLOCK}
%geom
    Calc_Hess true
    Recalc_Hess 10
    MaxIter {MAXITER}
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""


# Template for IRC validation (separate job)
DFT_INP_TEMPLATE_IRC = r"""! IRC
! B3LYP D4 def2-TZVP def2/J RIJCOSX

%maxcore {MAXCORE}

%pal nprocs {NPROCS} end

{SOLVENT_BLOCK}

%geom
    inhess read
    inhessname "{HESS_FILE}"
end
    
%irc
    MaxIter {MAXITER}
end

* xyzfile {CHARGE} {MULT} {ts_XYZFILE}
"""


SOLVENT_BLOCK = r"""%cpcm
    smd true
    smdsolvent "{SOLVENT}"
end

"""

PROG_EXT_BLOCK = r"""
%method
  ProgExt {METHOD}
  Ext_Params "--gfn 2 --alpb {SOLVENT}"
end


"""


def run_orca(input_file: Path, output_file: Path, orca_path: str = "orca") -> Path:
    """
    Run ORCA calculation.
    
    Args:
        input_file: Path to .inp file
        orca_path: Path to ORCA executable (use full path for parallel runs)
    
    Returns:
        Path to output file
    """
    original_dir = Path.cwd()
    try:
        # Change to input file's directory (ORCA wants this)
        input_dir = input_file.parent
        os.chdir(input_dir)
        
        with open(output_file, 'w') as out_f:
            subprocess.run(
                [orca_path, str(input_file.name)],
                stdout=out_f,
                stderr=subprocess.STDOUT,
                check=True
            )
    finally:
        # Always change back to original directory
        os.chdir(original_dir)
    
    return output_file


class OrcaParser():
    """
    Parses given files upon initialization.
    Reults are saved in self.results
    """
    def __init__(self, out_file: Path, hess_file):
        self.out_file = out_file
        self.hess_file = hess_file
        self.sp_traj_file = self.hess_file.parent / f'{self.hess_file.stem}_trj.xyz'
        self.freq_re = re.compile(r'(-?\d+(?:\.\d+)?)\s*(i?)\s*cm\*\*-1', re.IGNORECASE)

        self._parse_files()


    def _parse_files(self):
        # parse files
        out_results = self._parse_ts_optimization_results(self.out_file)
        if not out_results['converged']:
            self.results = out_results
            return
        
        if self.hess_file.exists():
            hess_results = self._parse_orca_hess_file(self.hess_file)
            self.results = {**out_results, **hess_results}
            
            # assertion
            neg_freq_count_hess = len([f for f in hess_results['frequencies'] if f < 0])
            assert out_results['neg_imag_cnt'] == neg_freq_count_hess
        else:
            print(f"Warning: Hessian file {self.hess_file} does not exist.")
            self.results = out_results

    
    def _parse_ts_optimization_results(self, out_file: Path):
        """
        Parse an ORCA output to extract TS optimization results.

        Returns a dict with:
        - converged (bool)
        - cycle_cnt (int)
        - energy_kcal (float|None): last single-point energy in kcal/mol
        - neg_imag_freqs_cm1 (list[float])      # subset that ORCA printed as negative numbers
        - neg_imag_cnt (int)                     # how many were printed as negatives
        """
        converged = False
        cycle_count = 0
        energy_hartree = 0

        in_freq = False
        saw_freq_line = False

        # Keep ONLY the last frequency block encountered
        neg_imag_freqs = []

        with out_file.open("r", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")

                if "GEOMETRY OPTIMIZATION CYCLE" in line:
                    cycle_count += 1

                if ("THE OPTIMIZATION HAS CONVERGED" in line) or ("FINAL SINGLE POINT ENERGY" in line): # first part: for DFT runs, second part: for XTB runs
                    converged = True

                if "ERROR !!!" in line:
                    converged = False
                
                if "FINAL SINGLE POINT ENERGY" in line:
                    parts = line.split()
                    try:
                        energy_hartree = float(parts[-1])
                    except ValueError:
                        pass

                if "VIBRATIONAL FREQUENCIES" in line:
                    in_freq = True
                    saw_freq_line = False
                    neg_imag_freqs.clear()
                    continue

                if in_freq:
                    # End of freq section: blank line after at least one frequency line
                    if line.strip() == "" and saw_freq_line:
                        in_freq = False
                        continue

                    matches = self.freq_re.findall(line)
                    if matches:
                        saw_freq_line = True
                        for val_str, _ in matches:
                            val = float(val_str)
                            is_imag = val < 0.0
                            if is_imag:
                                neg_imag_freqs.append(val)

        return {
            "converged": converged,
            "cycle_cnt": cycle_count,
            "energy_kcal":  energy_hartree * 627.509,
            "neg_imag_freqs_cm1": neg_imag_freqs,
            "neg_imag_cnt": len(neg_imag_freqs),
        }

    
    def _parse_orca_hess_file(self, hess_file: Path) -> Dict:
        """
        Parse ORCA .hess file to extract frequencies and normal mode eigenvectors.
        
        Returns:
            dict with:
                - 'frequencies': list of frequencies in cm^-1
                - 'normal_modes': (n_modes, n_atoms, 3) array of displacements
                - 'atoms': list of atom symbols
                - 'coords': (n_atoms, 3) array of coordinates in Angstrom
                - 'n_atoms': number of atoms
        """
        
        frequencies = []
        atoms = []
        coords = []
        normal_modes = None
        
        with hess_file.open('r') as f:
            content = f.read()
        
        # Split into sections
        sections = content.split('$')
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            header = lines[0].strip().lower()
            
            # Parse vibrational frequencies
            if header == 'vibrational_frequencies':
                n_freqs = int(lines[1].strip())
                for i in range(2, 2 + n_freqs):
                    parts = lines[i].strip().split()
                    freq = float(parts[1])  # Second column is frequency
                    frequencies.append(freq)
            
            # Parse normal modes (block format)
            elif header == 'normal_modes':
                dimension = int(lines[1].strip().split()[0])
                normal_modes = np.zeros((dimension, dimension))
                
                line_idx = 2
                end = False
                
                while not end and line_idx < len(lines):
                    # Read column indices for this block
                    col_indices = [int(j) for j in lines[line_idx].strip().split()]
                    line_idx += 1
                    
                    # Read dimension rows of data for this block
                    for i in range(dimension):
                        if line_idx >= len(lines):
                            break
                        parts = lines[line_idx].strip().split()
                        # First element is row index, rest are values
                        values = [float(v) for v in parts[1:]]
                        
                        # Assign values to the appropriate columns
                        for j_idx, col in enumerate(col_indices):
                            if j_idx < len(values):
                                normal_modes[i, col] = values[j_idx]
                        
                        line_idx += 1
                    
                    # Check if we've read all columns
                    if col_indices[-1] == dimension - 1:
                        end = True
            
            # Parse atoms and coordinates
            elif header == 'atoms':
                n_atoms = int(lines[1].strip())
                for i in range(2, 2 + n_atoms):
                    parts = lines[i].strip().split()
                    
                    # Handle different possible formats
                    # Common formats: 
                    # Symbol X Y Z (4 fields)
                    # Symbol Mass X Y Z (5 fields)
                    # Symbol AtomicNum Mass X Y Z (6 fields)
                    
                    atoms.append(parts[0])  # First is always symbol
                    
                    # Find X, Y, Z coordinates (last 3 fields)
                    if len(parts) >= 4:
                        coords.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
                    else:
                        # Fallback: skip coords if not enough fields
                        coords.append([0.0, 0.0, 0.0])
        
        coords = np.array(coords) if coords else None
        
        # Reshape normal_modes from (3*n_atoms, n_modes) to (n_modes, n_atoms, 3)
        if normal_modes is not None and len(atoms) > 0:
            n_atoms = len(atoms)
            n_modes = len(frequencies)
            reshaped_modes = np.zeros((n_modes, n_atoms, 3))
            
            for mode_idx in range(n_modes):
                for atom_idx in range(n_atoms):
                    reshaped_modes[mode_idx, atom_idx, 0] = normal_modes[atom_idx * 3 + 0, mode_idx]
                    reshaped_modes[mode_idx, atom_idx, 1] = normal_modes[atom_idx * 3 + 1, mode_idx]
                    reshaped_modes[mode_idx, atom_idx, 2] = normal_modes[atom_idx * 3 + 2, mode_idx]
            
            normal_modes = reshaped_modes
        
        return {
            'frequencies': frequencies,
            'normal_modes': normal_modes,
            'atoms': atoms,
            'coords': coords,
            'n_atoms': len(atoms)
        }