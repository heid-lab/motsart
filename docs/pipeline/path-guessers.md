# Step 2: Path Guessers

Path guessers take reactant complexes from Step 1 and generate transition state (TS) guess geometries.

## Execution order

The methods are implemented, but they have a data dependency in the default workflow:

1. Run **RMSD-PP** first to create initial TS guesses.
2. Run **RacerTS** second to find lower-energy conformers from the initial RMSD-PP TS guesses.

## Available methods

### RMSD-PP

Uses the xTB RMSD-PP algorithm to interpolate a reaction path between reactant and product geometries, then extracts the highest-energy point as the TS guess.

```bash
python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser env=test env.rxn_num=0
```

**Reference:** [xTB RMSD-PP documentation](https://xtb-docs.readthedocs.io/en/latest/path.html)

### RacerTS

Conformer sampling approach to find lower-energy conformers from the RMSD-PP TS guesses.

```bash
python -m motsart.path_guessers.ts_conf_sampler env=test env.rxn_num=0
```

### ML-FSM (Freezing String Method)

Double-ended TS search via the [Freezing String Method](https://github.com/thegomeslab/ML-FSM) (`mlfsm`). A string is grown and optimized between the reactant complex and its respective product on the same OMol25/FAIRChem MLIP ("eSEN", `esen-sm-conserving-all-omol`) used by the MLIP validator; the highest-energy node of the converged string is the TS guess.

```bash
python -m motsart.path_guessers.ml_fsm.ml_fsm_reaction_path_guesser env=test fsm_cfg=test env.rxn_num=0
```

The MLIP is configured through the `env` preset (`mlip_model`, `mlip_task_name`, `mlip_device`); FSM parameters (interpolation/optimization coordinates, node count, optimizer settings) are set via the `fsm_cfg` group (`base` / `test` / `local`, see `FSMPathGuesserParams`).

### ASE CI-NEB

Climbing-image Nudged Elastic Band via [ASE](https://wiki.fysik.dtu.dk/ase/). A band is IDPP-interpolated between the reactant complex and its respective product and relaxed on the same OMol25/FAIRChem MLIP ("eSEN", `esen-sm-conserving-all-omol`) used by the MLIP validator; the highest-energy (climbing) image of the converged band is the TS guess.

```bash
python -m motsart.path_guessers.neb.neb_reaction_path_guesser env=test neb_cfg=test env.rxn_num=0
```

The MLIP is configured through the `env` preset (`mlip_model`, `mlip_task_name`, `mlip_device`); NEB parameters (image count, interpolation, optimizer, force tolerance, spring constant) are set via the `neb_cfg` group (`base` / `test` / `local`, see `NEBPathGuesserParams`).

### Learning / GoFlow

Neural network-based TS guessing using trained GoFlow models. See [Learning](learning.md).

## Adding a new path guesser

To implement a new path-guessing algorithm to use desired methods such as NEB, FSM, etc.:

1. Create a new module under `src/motsart/path_guessers/`.
2. Inherit from `BaseReactionPathGuesser`.
3. Implement `guess_reaction_path()`.

```python
from motsart.path_guessers.base_reaction_path_guesser import BaseReactionPathGuesser

class MyPathGuesser(BaseReactionPathGuesser):
    def guess_reaction_path(self):
        # Your implementation here
        ...
```

## Output

TS guesses are saved to:

- `results*/R{rxn_id}/ts/{method}/ts_to_validate/*.xyz`

TSs in those `ts_to_validate` folders are saddle point optimized and IRC validated when the validator module is called.