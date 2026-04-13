RXN_NUM=1

# Complex finder ------------
python -m motsart.complex_finder.complex_finder env=tetrazine env.rxn_num=$RXN_NUM afir_cfg=local optim_cfg=tetrazine

# Path guesser ------------
python -m motsart.path_guessers.rmsd_pp.rmsd_pp_reaction_path_guesser -m env=tetrazine env.rxn_num=$RXN_NUM

#python -m motsart.learning.inference -m env=azide env.rxn_num=$RXN_NUM flow_module=mhfm_default

# Racer TS ------------
python -m motsart.path_guessers.ts_conf_sampler -m env=azide env.rxn_num=$RXN_NUM

# Validator ------------
python -m motsart.validator.base_validator -m env=azide validator_cfg=local validator=dft env.rxn_num=$RXN_NUM
