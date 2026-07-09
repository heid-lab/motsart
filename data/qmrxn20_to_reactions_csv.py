#!/usr/bin/env python3
"""Build atom-mapped reaction SMILES from the local QMrxn20 geometries.

The QMrxn20 paper defines labels as R1_R2_R3_R4_X_Y:

    Rk: A=H, B=NO2, C=CN, D=CH3, E=NH2
    X:  A=F, B=Cl, C=Br
    Y:  A=H, B=F, C=Cl, D=Br

C1 is the ethane carbon bearing R1, R2, and X; C2 bears R3, R4,
and the beta H abstracted in E2.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem


SUBSTITUENTS = {
    "A": "H",
    "B": "NO2",
    "C": "CN",
    "D": "CH3",
    "E": "NH2",
}

LEAVING_GROUPS = {
    "A": "F",
    "B": "Cl",
    "C": "Br",
}

NUCLEOPHILES = {
    "A": "H",
    "B": "F",
    "C": "Cl",
    "D": "Br",
}


@dataclass(frozen=True)
class LabelParts:
    r1: str
    r2: str
    r3: str
    r4: str
    x: str
    y: str

    @classmethod
    def parse(cls, label: str) -> "LabelParts":
        parts = label.split("_")
        if len(parts) != 6:
            raise ValueError(f"Expected six-part QMrxn20 label, got {label!r}")
        r1, r2, r3, r4, x, y = parts
        for code in (r1, r2, r3, r4):
            if code not in SUBSTITUENTS:
                raise ValueError(f"Unknown substituent code {code!r} in {label!r}")
        if x not in LEAVING_GROUPS:
            raise ValueError(f"Unknown leaving-group code {x!r} in {label!r}")
        if y not in NUCLEOPHILES:
            raise ValueError(f"Unknown nucleophile/base code {y!r} in {label!r}")
        return cls(r1, r2, r3, r4, x, y)

    @property
    def reactant_conformer_label(self) -> str:
        return "_".join([self.r1, self.r2, self.r3, self.r4, self.x, "0"])

    @property
    def sn2_product_label(self) -> str:
        return "_".join([self.r1, self.r2, self.r3, self.r4, "0", self.y])

    @property
    def e2_product_label(self) -> str:
        return "_".join([self.r1, self.r2, self.r3, self.r4, "0", "0"])


@dataclass(frozen=True)
class AtomMaps:
    c1: int
    c2: int
    r1: tuple[int, ...]
    r2: tuple[int, ...]
    r3: tuple[int, ...]
    r4: tuple[int, ...]
    x: int
    y: int
    beta_h: int


def substituent_atom_count(code: str) -> int:
    group = SUBSTITUENTS[code]
    return {
        "H": 1,
        "NO2": 3,
        "CN": 2,
        "CH3": 4,
        "NH2": 3,
    }[group]


def allocate_maps(parts: LabelParts) -> AtomMaps:
    next_map = 1
    c1, c2 = next_map, next_map + 1
    next_map += 2

    substituent_maps = []
    for code in (parts.r1, parts.r2, parts.r3, parts.r4):
        count = substituent_atom_count(code)
        maps = tuple(range(next_map, next_map + count))
        substituent_maps.append(maps)
        next_map += count

    x = next_map
    y = next_map + 1
    beta_h = next_map + 2
    return AtomMaps(c1, c2, *substituent_maps, x, y, beta_h)


class MolBuilder:
    def __init__(self) -> None:
        self.rw_mol = Chem.RWMol()
        self.map_to_idx: dict[int, int] = {}

    def atom(self, symbol: str, atom_map: int, formal_charge: int = 0) -> int:
        atom = Chem.Atom(symbol)
        atom.SetAtomMapNum(atom_map)
        atom.SetFormalCharge(formal_charge)
        atom.SetNoImplicit(True)
        idx = self.rw_mol.AddAtom(atom)
        self.map_to_idx[atom_map] = idx
        return idx

    def bond(self, map_1: int, map_2: int, order: Chem.BondType = Chem.BondType.SINGLE) -> None:
        self.rw_mol.AddBond(self.map_to_idx[map_1], self.map_to_idx[map_2], order)

    def build(self) -> Chem.Mol:
        mol = self.rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol


def add_substituent(builder: MolBuilder, parent_map: int, code: str, maps: tuple[int, ...]) -> None:
    group = SUBSTITUENTS[code]

    if group == "H":
        (h_map,) = maps
        builder.atom("H", h_map)
        builder.bond(parent_map, h_map)
        return

    if group == "NO2":
        n_map, o_dbl_map, o_minus_map = maps
        builder.atom("N", n_map, formal_charge=1)
        builder.atom("O", o_dbl_map)
        builder.atom("O", o_minus_map, formal_charge=-1)
        builder.bond(parent_map, n_map)
        builder.bond(n_map, o_dbl_map, Chem.BondType.DOUBLE)
        builder.bond(n_map, o_minus_map)
        return

    if group == "CN":
        c_map, n_map = maps
        builder.atom("C", c_map)
        builder.atom("N", n_map)
        builder.bond(parent_map, c_map)
        builder.bond(c_map, n_map, Chem.BondType.TRIPLE)
        return

    if group == "CH3":
        c_map, h1_map, h2_map, h3_map = maps
        builder.atom("C", c_map)
        builder.bond(parent_map, c_map)
        for h_map in (h1_map, h2_map, h3_map):
            builder.atom("H", h_map)
            builder.bond(c_map, h_map)
        return

    if group == "NH2":
        n_map, h1_map, h2_map = maps
        builder.atom("N", n_map)
        builder.bond(parent_map, n_map)
        for h_map in (h1_map, h2_map):
            builder.atom("H", h_map)
            builder.bond(n_map, h_map)
        return

    raise ValueError(f"Unhandled substituent group {group!r}")


def add_scaffold(builder: MolBuilder, parts: LabelParts, maps: AtomMaps, c1_c2_order: Chem.BondType) -> None:
    builder.atom("C", maps.c1)
    builder.atom("C", maps.c2)
    builder.bond(maps.c1, maps.c2, c1_c2_order)

    for parent_map, code, group_maps in (
        (maps.c1, parts.r1, maps.r1),
        (maps.c1, parts.r2, maps.r2),
        (maps.c2, parts.r3, maps.r3),
        (maps.c2, parts.r4, maps.r4),
    ):
        add_substituent(builder, parent_map, code, group_maps)


def build_reactant(parts: LabelParts) -> Chem.Mol:
    maps = allocate_maps(parts)
    builder = MolBuilder()
    add_scaffold(builder, parts, maps, Chem.BondType.SINGLE)

    builder.atom(LEAVING_GROUPS[parts.x], maps.x)
    builder.bond(maps.c1, maps.x)
    builder.atom("H", maps.beta_h)
    builder.bond(maps.c2, maps.beta_h)

    builder.atom(NUCLEOPHILES[parts.y], maps.y, formal_charge=-1)
    return builder.build()


def build_sn2_product(parts: LabelParts) -> Chem.Mol:
    maps = allocate_maps(parts)
    builder = MolBuilder()
    add_scaffold(builder, parts, maps, Chem.BondType.SINGLE)

    builder.atom(LEAVING_GROUPS[parts.x], maps.x, formal_charge=-1)
    builder.atom("H", maps.beta_h)
    builder.bond(maps.c2, maps.beta_h)
    builder.atom(NUCLEOPHILES[parts.y], maps.y)
    builder.bond(maps.c1, maps.y)
    return builder.build()


def build_e2_product(parts: LabelParts) -> Chem.Mol:
    maps = allocate_maps(parts)
    builder = MolBuilder()
    add_scaffold(builder, parts, maps, Chem.BondType.DOUBLE)

    builder.atom(LEAVING_GROUPS[parts.x], maps.x, formal_charge=-1)
    builder.atom(NUCLEOPHILES[parts.y], maps.y)
    builder.atom("H", maps.beta_h)
    builder.bond(maps.y, maps.beta_h)
    return builder.build()


def mapped_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)


def atom_map_set(mol: Chem.Mol) -> set[int]:
    return {atom.GetAtomMapNum() for atom in mol.GetAtoms()}


def element_count_by_map(mol: Chem.Mol) -> dict[int, str]:
    return {atom.GetAtomMapNum(): atom.GetSymbol() for atom in mol.GetAtoms()}


def element_counts(mol: Chem.Mol) -> Counter[str]:
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def xyz_element_counts(path: Path) -> Counter[str]:
    lines = path.read_text().splitlines()
    return Counter(line.split()[0] for line in lines[2:] if line.split())


def validate_reaction(rxn_smiles: str) -> None:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    r_smiles, p_smiles = rxn_smiles.split(">>")
    r_mol = Chem.MolFromSmiles(r_smiles, parser)
    p_mol = Chem.MolFromSmiles(p_smiles, parser)
    if r_mol is None or p_mol is None:
        raise ValueError(f"RDKit could not parse generated reaction: {rxn_smiles}")
    if atom_map_set(r_mol) != atom_map_set(p_mol):
        raise ValueError(f"Reactant/product atom-map sets differ: {rxn_smiles}")
    if element_count_by_map(r_mol) != element_count_by_map(p_mol):
        raise ValueError(f"Reactant/product atom identities differ: {rxn_smiles}")
    if Chem.GetFormalCharge(r_mol) != Chem.GetFormalCharge(p_mol):
        raise ValueError(f"Reactant/product charges differ: {rxn_smiles}")


def expected_geometry_paths(root: Path, reaction: str, parts: LabelParts, label: str) -> tuple[Path, Path, Path]:
    ts_path = root / "transition-states" / reaction / f"{label}.xyz"
    reactant_path = root / "reactant-conformers" / parts.reactant_conformer_label
    if reaction == "sn2":
        product_path = root / "product-conformers" / reaction / parts.sn2_product_label
    elif reaction == "e2":
        product_path = root / "product-conformers" / reaction / parts.e2_product_label
    else:
        raise ValueError(f"Unknown reaction type {reaction!r}")
    return ts_path, reactant_path, product_path


def is_curated_reaction(reaction: str, parts: LabelParts) -> bool:
    """
    Paper dataset curation: QMrxn20's Y=H cases are formal hydride
    reactions (SN2 gives C-H substitution; E2 gives H2 formation), which are
    charge-balanced but not representative halide SN2/E2 elementary steps.
    Degenerate SN2 identity exchanges such as R-Cl + Cl- -> R-Cl + Cl- are
    also removed because only the atom maps change while the unmapped reaction
    graph is identical, making them less useful as distinct validation cases.
    """
    if NUCLEOPHILES[parts.y] == "H":
        return False
    if reaction == "sn2" and LEAVING_GROUPS[parts.x] == NUCLEOPHILES[parts.y]:
        return False
    return True


def iter_reaction_rows(root: Path, include: set[str], include_edge_cases: bool = False) -> Iterable[tuple[int, str, str, str]]:
    row_id = 0
    for reaction in ("sn2", "e2"):
        if reaction not in include:
            continue
        ts_dir = root / "transition-states" / reaction
        for ts_file in sorted(ts_dir.glob("*.xyz")):
            label = ts_file.stem
            parts = LabelParts.parse(label)
            if not include_edge_cases and not is_curated_reaction(reaction, parts):
                continue
            ts_path, reactant_path, product_path = expected_geometry_paths(root, reaction, parts, label)
            if not ts_path.is_file():
                raise FileNotFoundError(ts_path)
            if not reactant_path.is_dir():
                raise FileNotFoundError(reactant_path)
            if not product_path.is_dir():
                raise FileNotFoundError(product_path)

            reactant = build_reactant(parts)
            product = build_sn2_product(parts) if reaction == "sn2" else build_e2_product(parts)
            ts_formula = xyz_element_counts(ts_path)
            if element_counts(reactant) != ts_formula or element_counts(product) != ts_formula:
                raise ValueError(f"Generated atom formula does not match TS XYZ formula for {reaction}/{label}")
            rxn_smiles = f"{mapped_smiles(reactant)}>>{mapped_smiles(product)}"
            validate_reaction(rxn_smiles)
            yield row_id, rxn_smiles, reaction, label
            row_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/sn2_e2_geometries"),
        help="Local QMrxn20 geometries root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reactions_sn2_e2.csv"),
        help="Output CSV matching reactions_am.csv: id,reaction_smiles with no header.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("data/reactions_sn2_e2_metadata.csv"),
        help="Optional metadata CSV linking row ids to QMrxn20 labels.",
    )
    parser.add_argument(
        "--include",
        choices=("both", "sn2", "e2"),
        default="both",
        help="Which QMrxn20 reaction channel(s) to export.",
    )
    parser.add_argument(
        "--include-edge-cases",
        action="store_true",
        help="Also export formal hydride and degenerate SN2 identity-exchange reactions.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for deterministic row shuffling before CSV writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include = {"sn2", "e2"} if args.include == "both" else {args.include}
    rows = list(iter_reaction_rows(args.root, include, include_edge_cases=args.include_edge_cases))
    random.Random(args.shuffle_seed).shuffle(rows)
    rows = [(row_id, rxn_smiles, reaction, label) for row_id, (_old_id, rxn_smiles, reaction, label) in enumerate(rows)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row_id, rxn_smiles, _reaction, _label in rows:
            writer.writerow([row_id, rxn_smiles])

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_out.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "reaction", "label"])
            for row_id, _rxn_smiles, reaction, label in rows:
                writer.writerow([row_id, reaction, label])

    print(f"Wrote {len(rows)} reactions to {args.out}")
    print(f"Shuffled rows with seed {args.shuffle_seed}")
    if args.metadata_out:
        print(f"Wrote metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
