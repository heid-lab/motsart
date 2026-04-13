import sys
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def read_xyz_and_perceive_bonds(xyz_file, charge):
    """
    Reads an XYZ file and perceives bonds allowing for a specific molecular charge.
    """
    try:
        raw_mol = Chem.MolFromXYZFile(xyz_file)
        if raw_mol is None:
            raise ValueError(f"Could not parse XYZ file: {xyz_file}")
        
        mol = Chem.Mol(raw_mol)
        
        # 1. Determine Connectivity (Distance-based)
        rdDetermineBonds.DetermineConnectivity(mol)
        
        # 2. Determine Bond Orders (Valence-based)
        # Explicitly pass the charge so RDKit allows ions (e.g., O-)
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
        
        # 3. Handle Hydrogens (Make them explicit atoms)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                atom.SetNoImplicit(True)
                
        return mol
    except Exception as e:
        print(f"Error processing {xyz_file} with charge {charge}:")
        print(f"Details: {e}")
        sys.exit(1)

def generate_reaction_smiles(reactant_file, product_file, charge):
    # 1. Load Molecules with the specified charge
    print(f"Processing reactant: {reactant_file}...")
    reactant = read_xyz_and_perceive_bonds(reactant_file, charge)    
    print(f"Processing product: {product_file}...")
    product = read_xyz_and_perceive_bonds(product_file, charge)

    # 2. Check Atom Consistency
    if reactant.GetNumAtoms() != product.GetNumAtoms():
        print("Error: Reactant and Product have different number of atoms.")
        sys.exit(1)

    # 3. Assign Atom Mapping
    num_atoms = reactant.GetNumAtoms()
    for i in range(num_atoms):
        map_num = i + 1
        reactant.GetAtomWithIdx(i).SetAtomMapNum(map_num)
        product.GetAtomWithIdx(i).SetAtomMapNum(map_num)

    # 4. Generate SMILES
    r_smiles = Chem.MolToSmiles(reactant, allHsExplicit=True, canonical=True)
    p_smiles = Chem.MolToSmiles(product, allHsExplicit=True, canonical=True)

    return f"{r_smiles}>>{p_smiles}"


if __name__ == "__main__":
    SYSTEM_CHARGE = 0
    # Replace these with your actual filenames
    r_file = "leon-rxn/r-2-1.xyz"
    p_file = "leon-rxn/p-2-1.xyz"
    
    # Simple file existence check or argument parsing could be added here
    try:
        rxn_smiles = generate_reaction_smiles(r_file, p_file, SYSTEM_CHARGE)
        print("Atom-Mapped Reaction SMILES:")
        print(rxn_smiles)
    except Exception as e:
        print(f"An error occurred: {e}")