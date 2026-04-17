# Quick Start

This guide walks you through running moTSart on a single reaction using the built-in `env=test` configuration.

## Input Format

moTSart reads CSV files with **no header row** and two columns:

1. `rxn_id`
2. `rxn_smiles`

Example:

```csv
0,[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>[CH2:5]1[CH2:6][CH2:4][CH:3]=[CH:2][CH2:1]1
```

Test data is provided in `data/test_rxns.csv`.

## Running Individual Steps

### Step 1: Complex Finder

```bash
python -m motsart.complex_finder.complex_finder env=test env.rxn_num=0
```

!!! tip
    For fast runs, keep `env=test` and explicit test configs:
    `afir_cfg=test optim_cfg=test`.

### Step 2: Path Guessers

Run path guessers in this order:

1. RMSD-PP to generate initial TS guesses.
2. RacerTS to refine those RMSD-PP guesses.

=== "RMSD-PP (required first)"

    ```bash
    python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser env=test env.rxn_num=0
    ```

=== "RacerTS (refines RMSD-PP output)"

    ```bash
    python -m motsart.path_guessers.ts_conf_sampler env=test env.rxn_num=0
    ```

### Step 3: Validator

Validate TS guesses with xTB:

```bash
python -m motsart.validator.base_validator env=test validator_cfg=test validator=xtb env.rxn_num=0
```

### Optional: Reproduction Workflow

For model-generated TS evaluation and paper-style comparison workflow, see [Paper Reproduction Workflow](../pipeline/paper-reproduction.md).

## Running the Full Pipeline

```bash
bash complex_and_ts_search_local.sh
```

`complex_and_ts_search_local.sh` is a local template script. Adjust `RXN_NUM`, `env`, and config names for your run.

## Parallel Execution

Use Hydra joblib launcher:

```bash
python -m motsart.complex_finder.complex_finder -m \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=2 \
    env=test \
    "env.rxn_num=range(0,2)"
```

## Results

Results are saved under the configured results directory (for `env=test`, default is `results_test/`):

```
results_test/
└── R{rxn_id}/
    ├── r/                       # Reactant complexes
    ├── p/                       # Product complexes
    ├── ts/{method}/ts_to_validate/
    └── validation/{method}/
```
