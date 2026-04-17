# Pipeline Overview

Multi-stage pipeline of moTSart for transition-state discovery. Each stage consumes artifacts from the previous stage.

## Architecture

```mermaid
flowchart TD
    INPUT["Input Reactions (CSV)<br/><code>rxn_id, rxn_smiles</code>"]
    CF["<b>Step 1: Complex Finder</b><br/>Evolut. algorithm + AFIR<br/><code>motsart.complex_finder</code>"]
    PG["<b>Step 2: Path Guessers</b><br/>RMSD-PP then RacerTS<br/><code>motsart.path_guessers</code>"]
    VAL["<b>Step 3: Validator</b><br/>xTB or DFT + IRC<br/><code>motsart.validator</code>"]
    AL["<b>Optional: Learning</b><br/>Data prep + model TS evaluation<br/><code>motsart.learning</code>"]

    INPUT --> CF --> PG --> VAL --> AL
    AL -.->|improved TS guesses| PG
```

## Main abstractions

### PathHandler

`PathHandler` (`motsart.common`) is the central path utility used across all stages.

### Configuration

All runtime entrypoints use Hydra-Zen-backed config stores. See [Configuration](../configuration/index.md).

## Module summary

| Module | Entry Point | Purpose |
|--------|------------|---------|
| `complex_finder` | `python -m motsart.complex_finder.complex_finder` | Find reactant complexes |
| `path_guessers.rmsd_pp` | `python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser` | Generate initial TS guesses |
| `path_guessers.ts_conf_sampler` | `python -m motsart.path_guessers.ts_conf_sampler` | Refine RMSD-PP guesses via RacerTS |
| `validator` | `python -m motsart.validator.base_validator` | Validate TS guesses + IRC |
| `learning.results_to_data_pkl` | `python -m motsart.learning.results_to_data_pkl` | Build AL training/eval data |
| `learning.rtsp_guesser` | `python -m motsart.learning.rtsp_guesser` | Multihead RTSP guess generation |

## Results directory structure

Each reaction `R{rxn_id}` has its own subtree:

```
results*/
└── R{rxn_id}/
    ├── r/
    │   ├── temp/
    │   ├── struct_xyzs/
    │   └── final_complexes/
    ├── p/
    ├── ts/
    │   ├── rmsd_pp/
    │   ├── racer_ts/
    │   └── learning/
    └── validation/
        ├── rmsd_pp/
        ├── racer_ts/
        └── learning/
```
