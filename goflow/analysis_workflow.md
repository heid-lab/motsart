# Results analysis plots
Run `fetch_reaction_analysis.sh`.
Finetune folders with different dataset sizes must have their number as _NUM_ in their filename. Scripts will recognize those automatically.
## Create geometric analysis plots
Run `finetune_dp_plot_paper.py` in the `reaction_analysis` folder. 
## Create force analysis plots
Run `bash scripts/cyclo/test_samples_analysis_forces.sh`. Set the desired files first. Set `--evaluate-force-norm-guess`and `--evaluate-force-norm-ground-truth`
Run `python finetune_dp_force_plot.py` in the `reaction_analysis/cyclo` folder.