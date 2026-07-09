# Experiments & analysis

Scripts for running and analyzing the **PES-engine comparison** for saddle-point
optimization. The moTSart validator can drive ORCA's TS optimizer with three
interchangeable engines, selected via the Hydra `validator` group:

| `validator=` | Class | Level of theory |
|--------------|-------|-----------------|
| `xtb`  | `GFN2XTBValidator` | ORCA native GFN2-xTB |
| `dft`  | `DFTValidator`     | B3LYP-D4/def2-TZVP |
| `mlip` | `MLIPValidator`    | OMol25 MLIP (default eSEN-sm-conserving) via ORCA `ExtOpt` |

The `mlip` engine works through ORCA's `otool_external` interface: ORCA's own
`OptTS`/`IRC`/`NumFreq` drive the geometry while every energy+gradient comes from
a FAIRChem model, implemented in
[`mlip_external.py`](../src/motsart/validator/orca_validator/orca_external_tools/mlip_external.py).
Switching engines changes nothing else in the pipeline.

## Prerequisites

- `xtb` / `dft`: nothing beyond the base install + ORCA.
- `mlip`: `fairchem-core` + `torch` in the `motsart` env, and access to the eSEN
  checkpoint. By default the model is the registry name `esen-sm-conserving-all-omol`
  (HuggingFace-gated); or point at a local file:
  `export MOTSART_MLIP_MODEL=/abs/path/to/esen_sm_conserving_all.pt`.
  The device is auto-selected (CUDA if available, else CPU).

## 0. Validate the ORCA ⇄ wrapper plumbing (no GPU/torch needed)

```bash
ORCA_PATH=/path/to/orca bash experiments/test_orca_mlip_plumbing.sh
```

Runs a single `! ExtOpt EnGrad` job with the wrapper in `--dummy` mode (a trivial
analytic potential), confirming ORCA can call the wrapper and read back a valid
`.engrad`. Passing this means only the model needs swapping in for real runs.

## 1. Run validation with a chosen engine

The validator optimizes **TS guesses that already exist** in the results tree
(from an earlier ComplexFinder → PathGuesser run), for a given `TS_METHOD`
(`racer_ts`, `rmsd_pp`, `learning`, ...), and writes per-reaction
`validation_<Validator>.csv` files.

Local (loops over all `R*` folders):
```bash
ENGINE=mlip RESULTS_FOLDER=results_test CSV_FILE=data/test_rxns.csv \
TS_METHOD=racer_ts ENV_NAME=test VALIDATOR_CFG=local \
bash experiments/run_validation_engine_local.sh
```

SLURM array (one reaction per task; uncomment the GPU lines for `mlip`):
```bash
ENGINE=mlip RESULTS_FOLDER=results_goflow/results_goflow TS_METHOD=racer_ts \
sbatch experiments/run_validation_engine.sh
```

### Full pipeline on GPU nodes (DataLab)

`run_rxns_datalab_gpu.sh` runs the **whole pipeline** (ComplexFinder → RMSD-PP →
racerTS → MLIP validator) on the `GPU-a100s` partition, one A100 per array task.
Edit the variables directly at the top of the script: set `--array=0-(NODES-1)` to
the number of GPU nodes you want in parallel and `N_REACTIONS` to how many
reactions to run (`all` = every row in the CSV). The reactions `0..N-1` are split
**evenly across the array tasks** (task `i` does `i, i+A, i+2A, ...`, so each task
runs ~N/A reactions). The persistent MLIP worker auto-starts once per node and
stays warm across that node's reactions.

```bash
# edit the CONFIG block + #SBATCH --array at the top of the script, then:
mkdir -p slurm
sbatch experiments/run_rxns_datalab_gpu.sh
```

DataLab-specific bits to set in the script: the `#SBATCH --array` size (number of
GPU nodes), `ORCA_PATH`, any `module load`, and the `ENV`/`*_CFG`/`CSV_FILE`/
`RESULTS_FOLDER` for your dataset. The CPU stages (xTB) run on the node's CPUs
while the GPU serves the validator.

## 2. Summarize optimization cost

```bash
python experiments/summarize_cycles.py \
    --results-folder results_goflow/results_goflow \
    --validator MLIPValidator --ts-method racer_ts
```
Reports convergence rates and the optimization-cycle distribution (mean / median
/ MAD) over IRC-validated TSs - the metric, for any engine/guess source.

## 3. Compare engines / guess sources (with reductions)

```bash
# Does TsOptNet still cut optimization cycles under the MLIP engine?
python experiments/compare_engines.py \
    --paired \
    --series baseline:results_goflow/results_goflow:MLIPValidator:racer_ts \
    --series tsoptnet:results_goflow/results_goflow:MLIPValidator:learning \
    --out-csv results_goflow/compare_mlip.csv
```

Each `--series` is `LABEL:RESULTS_FOLDER:VALIDATOR:TS_METHOD`. With `--paired`,
all series are restricted to the molecules IRC-valid in every series, so the
cycle-count comparison is paired. Reductions are reported relative to the first
series. Use the same pattern to compare engines on identical guesses
(`DFTValidator:racer_ts` vs `MLIPValidator:racer_ts`).

## 4. Pipeline timings & per-stage failure rates

The pipeline is instrumented (`motsart.telemetry`) to record, per reaction, both
**timings** of the expensive components and **failure tallies** at each stage.
Records are appended to `results*/R{rxn_id}/telemetry/metrics.jsonl` automatically
whenever you run ComplexFinder / path guessers / the validator — no extra flags.
Set `MOTSART_TELEMETRY=0` to disable.

What is tracked:

| Stage | Timed components | Failure tallies (total vs failed) |
|-------|------------------|-----------------------------------|
| `complex_finder` | `complex_finder_total`, `rdkit_conformers`, `dist_optimization`, `xtb_relax`, `afir_total`, `afir_product_search` | `ea_round`, `reaction_processed`, `afir_product_search` |
| `rmsd_pp` | `path_search_total`, `path_search` | `path_search` (RMSD-PP path search) |
| `racer_ts` | `racer_ts_total`, `racer_ts_sampling` | `racer_ts_sampling` |
| `validator` | `sp_opt`, `irc` | `sp_opt` (converged single-imaginary saddle?), `irc` (connects R/P?) |

Summarize across all reactions:

```bash
python experiments/summarize_telemetry.py \
    --results-folder results_goflow/results_goflow \
    --out-prefix results_goflow/telemetry
```

This reports per-component durations, **non-overlapping per-stage wall-clock**
(the `*_total` wrappers, so nested leaves are not double-counted), the
**end-to-end wall-clock per reaction**, the **validator seconds per IRC-validated
TS** (answers the total-cost question directly), and **per-stage pass/failure
rates**. CSVs (`*_timing_by_component.csv`, `*_failures.csv`,
`*_failures_per_reaction.csv`) are written for the manuscript.
