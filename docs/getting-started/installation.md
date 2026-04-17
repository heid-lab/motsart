# Installation

## Prerequisites

moTSart requires **conda** (or mamba) because xTB is distributed via conda-forge.

We recommend installing [Miniforge](https://github.com/conda-forge/miniforge/releases/) (includes mamba):

=== "Linux"

    ```bash
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh
    ~/miniforge3/bin/conda init bash
    exec bash
    ```

=== "macOS (Apple Silicon)"

    ```bash
    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    chmod +x Miniforge3-MacOSX-arm64.sh
    bash Miniforge3-MacOSX-arm64.sh
    exec zsh
    ```

## Install moTSart

Create the environment, activate it, and install moTSart in editable mode:

```bash
conda env create -f environment.yml
conda activate motsart
pip install -e .
```

## Optional Dependencies

### Pysisyphus (reaction path optimization)

```bash
pip install pysisyphus
```

### GPU4PySCF (Linux only, GPU-accelerated DFT)

```bash
pip install pyscf gpu4pyscf-cuda12x
```

### PyTorch & PyTorch Geometric

Required for learning. Install the version matching your platform:

=== "Linux (CUDA 12.4)"

    ```bash
    pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0'
    pip install -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
        pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
    ```

=== "macOS (CPU)"

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \
        -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    ```

## External Software

### xTB

Installed via the conda environment. Verify:

```bash
xtb --version
```

### ORCA 6.1 (optional, for DFT validation)

[Download ORCA](https://orcaforum.kofo.mpg.de/) and configure its path in your environment config (see [Configuration](../configuration/index.md)).
