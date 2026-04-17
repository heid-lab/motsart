# moTSart

![motsart-overview](assets/motsart-overview.png)

## Pipeline at a Glance

```mermaid
graph LR
    A[Input Reactions<br/>CSV: rxn_id, rxn_smiles] --> B[Complex Finder]
    B --> C[Path Guessers]
    C --> D[Validator]
    D --> E[Learning]
    E -.-> C

    style A fill:#e1bee7,stroke:#6a1b9a
    style B fill:#ce93d8,stroke:#6a1b9a
    style C fill:#ba68c8,stroke:#6a1b9a,color:#fff
    style D fill:#ab47bc,stroke:#6a1b9a,color:#fff
    style E fill:#9c27b0,stroke:#6a1b9a,color:#fff
```

| Step | Module | Description |
|------|--------|-------------|
| 1 | [Complex Finder](pipeline/complex-finder.md) | Evolutionary algorithm + AFIR to find reactant complexes |
| 2 | [Path Guessers](pipeline/path-guessers.md) | RMSD-PP + RacerTS TS guess generation |
| 3 | [Validator](pipeline/validator.md) | xTB or DFT validation with IRC pathway confirmation |
| 4 | [Learning](pipeline/learning.md) | Data preparation and evaluatio workflow for TsOptNet |

## Quick Links

- [Installation](getting-started/installation.md) - Set up your environment
- [Quick Start](getting-started/quickstart.md) - Run your first reaction
- [Configuration](configuration/index.md) - Hydra-Zen config system
- [Cluster & HPC](cluster/index.md) - Running on SLURM clusters
- [Paper Reproduction Workflow](pipeline/paper-reproduction.md)
- [API Reference](reference/) - Auto-generated module documentation
