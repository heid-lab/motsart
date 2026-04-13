"""Hydra-Zen configuration store for complex finder parameters."""

from hydra_zen import store
from motsart.conf_default import OptimizationConfig, AFIRPathGuesserParams


# ---------------------------------- Optimization Configs ----------------------------------
# Create builder that automatically populates defaults from dataclasses
optim_store = store(group="optim_cfg")

optim_store(
    OptimizationConfig, 
    name="base"
)

optim_store(
    OptimizationConfig(
        n_EA_rounds=1,
        n_confs=128,
        n_confs_after_product_similarity_filter=16,
        dist_population_size=512,
        dist_generations=32,
        n_rcs_to_screen_for_energy=2,
    ),
    name="test"
)

optim_store(
    OptimizationConfig(
        complex_method="rtsp_goflow",
    ),
    name="rtsp"
)

optim_store(
    OptimizationConfig(
        n_EA_rounds=4,
        n_confs=512,
        n_confs_after_product_similarity_filter=32,
        dist_population_size=1024,
        dist_generations=128,
        n_rcs_to_screen_for_energy=10,
    ),
    name="local"
)

optim_store(
    OptimizationConfig(
        n_EA_rounds=4,
        n_confs=512,
        n_confs_after_product_similarity_filter=32,
        dist_population_size=1024,
        dist_generations=128,
        n_rcs_to_screen_for_energy=8,
    ),
    name="cyclo32"
)

optim_store(
    OptimizationConfig(
        n_EA_rounds=6,
        n_confs=128,
        n_confs_after_product_similarity_filter=128,
        dist_population_size=1024,
        dist_generations=256,
        dist_rotation_sigma=35.0,
        dist_translation_sigma=4.5,
        n_rcs_to_screen_for_energy=6,
    ),
    name="leon"
)

optim_store(
    OptimizationConfig(
        n_EA_rounds=12,
        n_confs=64,
        n_confs_after_product_similarity_filter=128,
        dist_population_size=1024,
        dist_generations=128,
        dist_rotation_sigma=35.0,
        dist_translation_sigma=4.5,
        n_rcs_to_screen_for_energy=4,
        product_similarity_coef=0.0,
        forming_bond_vdw_coef=2.15,
    ),
    name="tetrazine"
)



# ---------------------------------- AFIR Configs ----------------------------------
afir_store = store(group="afir_cfg")

afir_store(
    AFIRPathGuesserParams(
        fc_init_upper=0.01,
        fc_binary_search_depth=5,
        num_ts_for_validation=3,
    ),
    name="base"
)

afir_store(
    AFIRPathGuesserParams(
        fc_init_upper=0.01,
        fc_binary_search_depth=3,
        num_ts_for_validation=1,
    ),
    name="test"
)

afir_store(
    AFIRPathGuesserParams(
        fc_init_upper=0.01,
        fc_binary_search_depth=8,
        num_ts_for_validation=4,
    ),
    name="local"
)

afir_store(
    AFIRPathGuesserParams(
        fc_init_upper=0.1,
        fc_binary_search_depth=4,
        num_ts_for_validation=12,
    ),
    name="leon"
)

afir_store(
    AFIRPathGuesserParams(
        fc_init_upper=0.1,
        fc_binary_search_depth=4,
        num_ts_for_validation=12,
    ),
    name="tetrazine"
)