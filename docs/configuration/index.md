# Configuration

moTSart uses [Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for managing configurations. This page describes the environment configuration class. To see examplex on how to use other configuration classes please refer to the pipeline documentation and the API reference.

## Environment Config

The `env` group sets paths, reaction data, and solver locations.

```python
@dataclass
class EnvironmentConfig:
    rxn_csv: str
    rxn_num: Optional[int]
    rxn_id: Optional[int]
    solvent: Optional[str]
    orca_path: str
    xtb_path: str
    results_folder: str = "results"
    vdw_coef: float = 0.9
```

### Predefined Environments
Examples:

=== "local"

    Local machine setup:
    ```bash
    python -m motsart.complex_finder.complex_finder env=local env.rxn_num=0
    ```

=== "cluster"

    Cluster/SLURM setup:
    ```bash
    python -m motsart.complex_finder.complex_finder env=cluster env.rxn_num=0
    ```

## Overriding Values

Hydra lets you override any field from CLI.

```bash
# Override results folder
python -m motsart.complex_finder.complex_finder env=test env.results_folder=my_results

# Override multiple values
python -m motsart.complex_finder.complex_finder env=test env.rxn_num=1 optim_cfg.n_confs=256

# Sweep multiple reactions
python -m motsart.complex_finder.complex_finder -m env=test "env.rxn_num=range(0,2)"
```

## Module-Specific Group Usage

Different modules require different config groups. Common examples:

```bash
# Complex finder (optim_cfg + afir_cfg + env)
python -m motsart.complex_finder.complex_finder env=test optim_cfg=test afir_cfg=test env.rxn_num=0

# Validator (validator_cfg + validator + env)
python -m motsart.validator.base_validator env=test validator_cfg=test validator=xtb env.rxn_num=0
```

## Adding a New Environment

Add your preset in `src/motsart/conf.py`.

```python
env_store(
    EnvironmentConfig(
        rxn_csv='data/my_reactions.csv',
        rxn_num=0,
        rxn_id=None,
        orca_path='/path/to/orca',
        xtb_path='/path/to/xtb',
        solvent='water',
        results_folder='results_custom',
    ),
    name="my_env"
)
```

Then use it:

```bash
python -m motsart.complex_finder.complex_finder env=my_env
```

## Dataclass Definitions

Dataclasses are defined in `src/motsart/conf_default.py`. See the [API Reference](../reference/) for generated module docs.
