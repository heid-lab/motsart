"""
Tool to add explicit hydrogen atom mappings to reaction SMILES strings.

This script takes a reaction SMILES with atom mappings on heavy atoms only
and adds explicit hydrogens with their corresponding atom mappings to both
reactants and products.
"""

from rdkit import Chem
from typing import Dict
import pandas as pd


REMOVE_NON_ORGANICS = True
MAX_ATOMS_PER_RXN = 200
CYCLO_RXN_CSV_FILE = 'azide/prelim.csv'
NEW_CYCLO_RXN_CSV_FILE = 'azide/prelim_h.csv'
RXN_SMILES_COL = 'mapped'
SHUFFLE = False


def add_hydrogen_atom_mappings(reaction_smiles: str, verbose: bool = True) -> str:
    """
    Add explicit hydrogens with atom mappings to a reaction SMILES.

    This function:
    1. Parses the reaction SMILES (reactants >> products)
    2. Adds explicit hydrogens to all molecules
    3. Assigns atom map numbers to all hydrogens based on their parent atoms
    4. Returns the complete reaction SMILES with all atoms mapped

    Args:
        reaction_smiles: Reaction SMILES with atom mappings on heavy atoms
        verbose: If True, print detailed information about the mapping

    Returns:
        Reaction SMILES with explicit hydrogens and complete atom mappings

    Example:
        >>> rxn = "[CH2:1]=[CH:2][CH3:3].[CH:4]#[N+:5][C-:6]([CH3:7])[C:8]#[N:9]>>[CH2:1]1[CH:2]([CH3:3])[CH:4]=[N:5][C:6]1([CH3:7])[C:8]#[N:9]"
        >>> mapped_rxn = add_hydrogen_atom_mappings(rxn)
    """
    # Split reaction into reactants and products
    parts = reaction_smiles.split('>>')
    if len(parts) != 2:
        raise ValueError("Invalid reaction SMILES format. Expected format: reactants>>products")

    reactants_smiles = parts[0]
    products_smiles = parts[1]

    # Parse reactants and products
    reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split('.')]
    products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split('.')]

    # Check parsing was successful
    if None in reactants or None in products:
        raise ValueError("Failed to parse SMILES. Check for syntax errors.")

    # Find the maximum atom map number currently used
    max_map_num = 0
    for mol in reactants + products:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > max_map_num:
                max_map_num = map_num

    if verbose:
        print(f"Maximum heavy atom map number: {max_map_num}")
        print(f"Starting hydrogen mapping from: {max_map_num + 1}")

    # Add explicit hydrogens and map them
    next_map_num = max_map_num + 1

    # Process reactants
    reactants_mapped = []
    reactant_h_mapping = {}  # Maps (parent_atom_map, h_index) -> new_h_map_num

    for mol_idx, mol in enumerate(reactants):
        mol_with_h = Chem.AddHs(mol)

        # Group hydrogens by their parent atom
        parent_h_dict = {}

        for atom in mol_with_h.GetAtoms():
            if atom.GetSymbol() == 'H':
                # Find the parent heavy atom
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    parent_atom = neighbors[0]
                    parent_map = parent_atom.GetAtomMapNum()

                    if parent_map > 0:
                        if parent_map not in parent_h_dict:
                            parent_h_dict[parent_map] = []
                        parent_h_dict[parent_map].append(atom)

        # Assign map numbers to hydrogens
        for parent_map, h_atoms in parent_h_dict.items():
            for h_idx, h_atom in enumerate(h_atoms):
                h_atom.SetAtomMapNum(next_map_num)
                reactant_h_mapping[(parent_map, h_idx)] = next_map_num
                if verbose:
                    print(f"  Reactant: H on atom {parent_map} (#{h_idx}) -> map {next_map_num}")
                next_map_num += 1

        reactants_mapped.append(mol_with_h)

    if verbose:
        print(f"\nTotal hydrogens in reactants: {len(reactant_h_mapping)}")

    # Process products - map hydrogens based on their parent atoms
    products_mapped = []
    new_h_count = 0

    for mol_idx, mol in enumerate(products):
        mol_with_h = Chem.AddHs(mol)

        # Group hydrogens by their parent atom
        parent_h_dict = {}

        for atom in mol_with_h.GetAtoms():
            if atom.GetSymbol() == 'H':
                # Find the parent heavy atom
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    parent_atom = neighbors[0]
                    parent_map = parent_atom.GetAtomMapNum()

                    if parent_map > 0:
                        if parent_map not in parent_h_dict:
                            parent_h_dict[parent_map] = []
                        parent_h_dict[parent_map].append(atom)

        # Assign map numbers to hydrogens
        for parent_map, h_atoms in parent_h_dict.items():
            for h_idx, h_atom in enumerate(h_atoms):
                key = (parent_map, h_idx)
                if key in reactant_h_mapping:
                    # Hydrogen existed in reactant
                    h_atom.SetAtomMapNum(reactant_h_mapping[key])
                    if verbose:
                        print(f"  Product: H on atom {parent_map} (#{h_idx}) -> map {reactant_h_mapping[key]} (from reactant)")
                else:
                    # New hydrogen formed in reaction
                    h_atom.SetAtomMapNum(next_map_num)
                    if verbose:
                        print(f"  Product: H on atom {parent_map} (#{h_idx}) -> map {next_map_num} (NEW)")
                    next_map_num += 1
                    new_h_count += 1

        products_mapped.append(mol_with_h)

    if verbose:
        print(f"\nNew hydrogens formed in products: {new_h_count}")
        print(f"Total atom map numbers used: {next_map_num - 1}")

    # Convert back to SMILES with atom maps
    reactants_smiles_mapped = '.'.join([Chem.MolToSmiles(mol) for mol in reactants_mapped])
    products_smiles_mapped = '.'.join([Chem.MolToSmiles(mol) for mol in products_mapped])

    return f"{reactants_smiles_mapped}>>{products_smiles_mapped}"


def compare_hydrogen_counts(reaction_smiles: str) -> Dict[str, int]:
    """
    Compare hydrogen counts between reactants and products.

    Args:
        reaction_smiles: Reaction SMILES (with or without explicit H)

    Returns:
        Dictionary with hydrogen statistics
    """
    parts = reaction_smiles.split('>>')
    if len(parts) != 2:
        raise ValueError("Invalid reaction SMILES format")

    reactants = [Chem.MolFromSmiles(smi) for smi in parts[0].split('.')]
    products = [Chem.MolFromSmiles(smi) for smi in parts[1].split('.')]

    # Count hydrogens
    reactant_h = sum([sum(1 for atom in Chem.AddHs(mol).GetAtoms() if atom.GetSymbol() == 'H') 
                      for mol in reactants])
    product_h = sum([sum(1 for atom in Chem.AddHs(mol).GetAtoms() if atom.GetSymbol() == 'H') 
                     for mol in products])

    return {
        'reactant_hydrogens': reactant_h,
        'product_hydrogens': product_h,
        'balanced': reactant_h == product_h
    }

def example_rxn():
    # Example: [3+2] cycloaddition reaction
    test_reaction = "[CH2:1]=[CH:2][CH3:3].[CH:4]#[N+:5][C-:6]([CH3:7])[C:8]#[N:9]>>[CH2:1]1[CH:2]([CH3:3])[CH:4]=[N:5][C:6]1([CH3:7])[C:8]#[N:9]"

    print("="*80)
    print("HYDROGEN ATOM MAPPING TOOL FOR REACTION SMILES")
    print("="*80)
    print("\nOriginal reaction:")
    print(test_reaction)
    print("\n" + "-"*80 + "\n")

    # Check hydrogen balance
    print("Hydrogen balance check:")
    h_stats = compare_hydrogen_counts(test_reaction)
    print(f"  Reactants: {h_stats['reactant_hydrogens']} H atoms")
    print(f"  Products: {h_stats['product_hydrogens']} H atoms")
    print(f"  Balanced: {h_stats['balanced']}")
    print("\n" + "-"*80 + "\n")

    # Add hydrogen mappings
    print("Adding explicit hydrogen atom mappings...")
    print()
    mapped_reaction = add_hydrogen_atom_mappings(test_reaction, verbose=True)

    print("\n" + "="*80)
    print("\nCOMPLETE REACTION WITH HYDROGEN MAPPINGS:")
    print(mapped_reaction)
    print("\n" + "="*80)

    # You can also use it programmatically
    print("\n\nUsage in your code:")
    print("from hydrogen_mapper import add_hydrogen_atom_mappings")
    print("mapped_rxn = add_hydrogen_atom_mappings(your_reaction_smiles)")


def recompute_all_smiles_cyclo32_dataset():
    df = pd.read_csv(CYCLO_RXN_CSV_FILE, index_col=0)
    
    # Filter reactions based on atom count
    def count_atoms(rxn_smiles):
        parts = rxn_smiles.split('>>')
        reactants = [Chem.MolFromSmiles(smi) for smi in parts[0].split('.')]
        total_atoms = sum(mol.GetNumAtoms() for mol in reactants if mol is not None)
        return total_atoms
    
    # Check if reaction contains only organic molecules
    def is_organic_reaction(rxn_smiles):
        organic_atoms = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl'}
        parts = rxn_smiles.split('>>')
        all_molecules = parts[0].split('.') + parts[1].split('.')
        
        for smi in all_molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in organic_atoms:
                    return False
        return True
    
    if REMOVE_NON_ORGANICS:
        df = df[df[RXN_SMILES_COL].apply(is_organic_reaction)]

    df['atom_count'] = df[RXN_SMILES_COL].apply(count_atoms)
    df = df[df['atom_count'] <= MAX_ATOMS_PER_RXN]
    df = df.drop(columns=['atom_count'])

    df[RXN_SMILES_COL] = df[RXN_SMILES_COL].apply(lambda x: add_hydrogen_atom_mappings(x, verbose=False))
    if SHUFFLE:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(NEW_CYCLO_RXN_CSV_FILE, index=False)


if __name__ == '__main__':
    recompute_all_smiles_cyclo32_dataset()
