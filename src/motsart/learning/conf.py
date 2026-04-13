"""Hydra-Zen configuration store for learning and GoFlow models.

Registers configs for the learning loop (:class:`ALConfig`),
the single-head GoFlow :class:`FlowModule`, and the multi-head
:class:`MultiHeadFlowModule` used by the RTSP guesser.
"""

from hydra_zen import make_custom_builds_fn, store
from goflow.gotennet.models.components.ops import CosineCutoff
from goflow.gotennet.models.representation.gotennet import GotenNet
from goflow.gotennet.models.representation.multihead_gotennet import MultiHeadGotenNet
from goflow.flow_matching.flow_module import FlowModule
from goflow.flow_matching.multihead_flow_module import MultiHeadFlowModule
from motsart.conf_default import ALConfig

# Initialize the store
fbuilds = make_custom_builds_fn(populate_full_signature=True)

# -------------------------------------- Learning --------------------------------------

al_cfg_store = store(group="al_cfg")

al_cfg_store(
    ALConfig(
        train_batch_size=64,
        train_epochs=4,
        learning_path_guesser='rmsd_pp',
    ),
    name="default"
)

al_cfg_store(
    ALConfig(
        train_batch_size=64,
        train_epochs=4,
        learning_path_guesser='afir',
    ),
    name="test"
)

# -------------------------------------- GoFlow --------------------------------------

# Define Cutoff Config
CutoffConfig = fbuilds(
    CosineCutoff,
    cutoff=5.0
)

# Define Representation (GotenNet) Config
GotenNetConfig = fbuilds(
    GotenNet,
    cutoff_fn=CutoffConfig,
    n_atom_basis=256,
    n_atom_rdkit_feats=36,
    n_interactions=3,
    n_rbf=20,
    radial_basis='expnorm',
    activation="swish",
    max_z=100,
    weight_init="xavier_uniform",
    bias_init="zeros",
    int_layer_norm="",
    int_vector_norm="",
    num_heads=8,
    attn_dropout=0.1,
    edge_updates='norej',
    aggr="add",
    edge_ln='',
    sep_int_vec=True,
    lmax=2
)

# Define FlowModule Config
FlowModuleConfig = fbuilds(
    FlowModule,
    representation=GotenNetConfig,
    lr=0.0005,
    lr_decay=0.8,
    lr_patience=5,
    lr_monitor="validation/ema_loss",
    ema_decay=0.9,
    weight_decay=0.01,
    num_steps=25,
    num_samples=1,
    seed=1,
    sample_method="pos_guess",
    output={'n_hidden': 64}, 
)

# Register the config to the store
store(FlowModuleConfig, group="flow_module", name="fm_default")

# -------------------------------------- MultiHead GoFlow --------------------------------------

# Define MultiHead Representation (MultiHeadGotenNet) Config
MultiHeadGotenNetConfig = fbuilds(
    MultiHeadGotenNet,
    cutoff_fn=CutoffConfig,
    n_atom_basis=256,
    n_atom_rdkit_feats=36,
    n_shared_layers=3,
    n_head_layers=2,
    n_rbf=20,
    radial_basis='expnorm',
    activation="swish",
    max_z=100,
    weight_init="xavier_uniform",
    bias_init="zeros",
    int_layer_norm="layer",
    int_vector_norm="max_min",
    num_heads=8,
    attn_dropout=0.1,
    edge_updates='norej',
    aggr="add",
    edge_ln='',
    sep_int_vec=True,
    lmax=2,
    active_heads=["TS"],
    use_cross_attention=False,
    predict_activation_energy=True,
    use_init_cond=True,
)

# Define MultiHeadFlowModule Config
MultiHeadFlowModuleConfig = fbuilds(
    MultiHeadFlowModule,
    representation=MultiHeadGotenNetConfig,
    lr=0.0005,
    lr_decay=0.8,
    lr_patience=5,
    lr_monitor="validation/val_loss",
    weight_decay=0.01,
    num_steps=25,
    num_samples=25,
    seed=1,
    active_heads=["TS"],
    loss_weights={"R": 1.0, "TS": 1.0, "P": 1.0},
    prior_modes={"R": "rdkit", "TS": "pos_guess", "P": "rdkit"},
    noise_levels={"R": 0.4, "TS": 0.4, "P": 0.4},
    output={'n_hidden': 64},
)

# Register multihead config to the store
store(MultiHeadFlowModuleConfig, group="flow_module", name="mhfm_default")
store(MultiHeadFlowModuleConfig, group="multihead_module", name="mhfm_default")
