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