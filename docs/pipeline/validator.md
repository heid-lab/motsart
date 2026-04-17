# Step 3: Validator

The Validator takes TS guess geometries from path guessers and validates them with quantum chemistry calculations. A TS is accepted if it has one imaginary frequency and the IRC check matches reaction connectivity.

## Validation process

```mermaid
flowchart LR
    TS[TS Guess] --> OPT[Geometry<br/>Optimization]
    OPT --> FREQ[Frequency<br/>Calculation]
    FREQ --> CHECK{Single imaginary<br/>frequency?}
    CHECK -->|Yes| IRC[IRC Calculation]
    CHECK -->|No| REJECT[Rejected]
    IRC --> VERIFY{Connects R & P?}
    VERIFY -->|Yes| VALID[Validated TS]
    VERIFY -->|No| REJECT
```

## Available validators

### GFN2-xTB validator

Fast semi-empirical validation.

```bash
python -m motsart.validator.base_validator env=test validator_cfg=test validator=xtb env.rxn_num=0
```

### DFT validator (ORCA)

More accurate but slower.

```bash
python -m motsart.validator.base_validator env=local validator_cfg=local validator=dft env.rxn_num=0
```

!!! note
    ORCA must be installed and configured in `env.orca_path`.

## Configuration

### Validator config (`validator_cfg`)

| Parameter | Description |
|-----------|-------------|
| `SP_maxcore` | Maximum memory per core (MB) for single-point calculations |
| `SP_nprocs` | Number of parallel processes |
| `SP_MaxIter` | Maximum optimization iterations |
| `IRC_MaxIter` | Maximum IRC iterations |
| `skip_full_irc` | If `true`, use heuristic IRC via graphRC only |
| `path_guessers_to_validate` | List of TS methods to validate |

## Computing statistics

When you ran the pipeline, e.g. in parallel on a cluster for hundreds of reactions, you want to check statistics on success metrics of SP optimization or IRC validation.
The custom script below was used to compare two different runs: guesses generated with GoFlow (TsOptNet in the paper) and those from another path guesser, such as RMSD-PP.
Please adjust it to your liking.

```bash
python -m motsart.validator.compute_stats \
  --cluster-folder /data/results \
  --learning-folder /home/lgalustian/projects/motsart/results_goflow \
  --validator DFTValidator \
  --output-csv /home/lgalustian/projects/motsart/results_goflow/stats_al.csv \
  --cluster-ts-method racer_ts \
  --al-ts-method learning \
  --mode both
```

## Output

Results are saved to `results*/R{rxn_id}/validation/{method}/`:

- optimized TS geometries
- frequency outputs
- IRC trajectories
- per-guess validation status CSV
