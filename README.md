<img width="2079" height="955" alt="motsart-overview" src="https://github.com/user-attachments/assets/d08eb38e-7b2c-4ba2-848d-34c5ea3681db" />

# Installation

moTSart uses conda (xTB is distributed via conda-forge). Installing Miniforge is the recommended setup.

Install Miniforge on Linux:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash
exec bash
```

Install Miniforge on macOS (Apple Silicon):
```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
exec zsh
```

Create environment, activate it, and install moTSart in editable mode:
```bash
conda env create -f environment.yml
conda activate motsart
pip install -e .
```

Optional dependencies:
```bash
# Reaction path optimization
pip install pysisyphus

# GPU DFT validation (Linux, optional)
pip install pyscf gpu4pyscf-cuda12x
```

PyTorch + PyG for learning:
```bash
# Linux (CUDA 12.4)
pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0'
pip install -f https://data.pyg.org/whl/torch-2.6.0+cu124.html pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# macOS (CPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

# Usage

## Add modules

To add a reaction path guesser, add a module under `src/motsart/path_guessers` that implements it.
See `src/motsart/path_guessers/rmsd_pp/rmsd_pp_reaction_path_guesser.py` for an example implementation.

## Run

- Run the full pipeline locally: `bash complex_and_ts_search_local.sh`
- Run the pipeline on SLURM CPU nodes: `sbatch complex_and_ts_search_cpu.sh`

# Analysis

Pipeline artifacts (geometries, paths, validation outputs) are written per reaction under the configured results directory.

Compute stats (example):
```bash
python -m motsart.validator.compute_stats \
  --cluster-folder results \
  --learning-folder results_goflow \
  --validator DFTValidator \
  --output-csv stats.csv \
  --cluster-ts-method racer_ts \
  --al-ts-method learning \
  --mode both
```

# Documentation
Check out the documentation for guides and the paper reproduction workflow.
