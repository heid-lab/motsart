<img width="2079" height="955" alt="motsart-overview" src="https://github.com/user-attachments/assets/d08eb38e-7b2c-4ba2-848d-34c5ea3681db" />

# Installation

moTSart uses conda (xTB is distributed via conda-forge). Installing Miniforge is the recommended setup.

0. Install Miniforge on Linux:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash
exec bash
```

0. Install Miniforge on macOS (Apple Silicon):
```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
exec zsh
```

1. Clone the moTSart repository:
  ```bash
  git clone https://github.com/heid-lab/motsart.git
  cd motsart
  ```

2. Create environment, activate it, and install moTSart (plus the vendored
   `ML-FSM` and `goflow` packages) in editable mode:
```bash
conda env create -f environment.yml
conda activate motsart
pip install -e ./ML-FSM -e ./goflow -e .
```
`goflow` provides the generative flow-matching model used by the learning
pipeline. To import `goflow` and run the learning/generative pipeline you also
need PyTorch + PyG (see [PyTorch & PyG](#pytorch--pyg-required-for-learning)
below) — the conda environment does not install them. The training/sampling
scripts under `goflow/scripts/` are the authors' original cluster scripts and
contain machine-specific paths; adapt them before use.

Optional dependencies:
```bash
# Reaction path optimization
pip install pysisyphus

# GPU DFT validation (Linux, optional)
pip install pyscf gpu4pyscf-cuda12x

# MLIP validator engine (validator=mlip): OMol25 model via ORCA ExtOpt
pip install fairchem-core   # + torch; see experiments/README.md for model access
```

The saddle-point optimization engine is selectable via `validator=xtb|dft|mlip`.
The `mlip` engine drives ORCA's optimizer with a FAIRChem OMol25 potential
(default `eSEN-sm-conserving`) through the `otool_external` interface; see
[`experiments/`](experiments/) for engine-comparison and analysis scripts.

## PyTorch & PyG (required for learning)

Required to import `goflow` and run the learning/generative pipeline. Install the build matching your platform. If you also use the `mlip` validator, install `fairchem-core` before this step (it can pull a newer torch and break the pinned versions below).

```bash
# Linux (CUDA 12.4)
pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0'
pip install -f https://data.pyg.org/whl/torch-2.6.0+cu124.html pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# macOS (CPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

## Pretrained models

The pretrained goflow (TsOptNet) checkpoint is available on Zenodo:
[zenodo.org/records/19554844](https://zenodo.org/records/19554844).

# Usage
Please check out the documentation for a comprehensive user guide: [heid-lab.github.io/motsart](https://heid-lab.github.io/motsart/)

## Configure

We use hydra-zen for managing configurations directly in python configuration files. Before running the pipeline, make sure you set all the paths in the environment configuration file under `src/motsart/conf.py`. This includes the paths to software such as xTB or ORCA, as well as the output directory where results will be written. These configuration files for the modules `complex_finder`, `path_guessers`, `validator`, and `learning` are found in their respective module folders and are named `conf.py`

## Run

Run the pipeline stages locally with the `env=local` config (replace `0` with the reaction index you want to process):

```bash
python -m motsart.complex_finder.complex_finder env=local env.rxn_num=0
python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser env=local env.rxn_num=0
python -m motsart.path_guessers.ts_conf_sampler env=local env.rxn_num=0
python -m motsart.validator.base_validator env=local validator=xtb env.rxn_num=0
```

To run across reactions on a SLURM cluster, use the batch template: `sbatch complex_and_ts_search_cpu.sh`.

# Analysis

Pipeline artifacts (geometries, paths, validation outputs) are written per reaction under the configured results directory.