"""
Multi-head GotenNet for simultaneous R, TS, P geometry prediction.

Architecture:
- Shared embedding layers (same as GotenNet)
- Shared backbone: N GATA+EQFF layer pairs
- Per-head layers: M GATA+EQFF layer pairs for each active head
- Optional cross-attention for TS head to attend to R and P features
"""

from functools import partial
from typing import Callable, Optional, Tuple, Mapping, Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import scatter
import math

from goflow.gotennet.models.components.ops import (
    Dense, str2basis, get_weight_init_by_string, str2act, Distance, CosineCutoff
)
from goflow.gotennet.models.components.ops import (
    TensorInit, NodeInit, EdgeInit, AtomCGREmbedding, EdgeCGREmbedding
)
from goflow.gotennet.utils import RankedLogger, _extend_condensed_graph_edge
from .gotennet import GATA, EQFF, TimestepEmbedding

log = RankedLogger(__name__, rank_zero_only=True)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention for TS head to attend to R and/or P features.

    Supports:
    - "r_to_ts": TS attends only to R features
    - "p_to_ts": TS attends only to P features
    - "bidirectional": TS attends to both R and P features

    during training to prevent over-reliance on cross-attention features.
    """

    def __init__(
        self,
        n_atom_basis: int,
        num_heads: int = 8,
        cross_type: str = "bidirectional",
        dropout: float = 0.0,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads
        self.n_atom_basis = n_atom_basis
        self.head_dim = n_atom_basis // num_heads

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        # Query from TS
        self.q_proj = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Keys and values from R
        if cross_type in ["r_to_ts", "bidirectional"]:
            self.k_proj_r = InitDense(n_atom_basis, n_atom_basis, activation=None)
            self.v_proj_r = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Keys and values from P
        if cross_type in ["p_to_ts", "bidirectional"]:
            self.k_proj_p = InitDense(n_atom_basis, n_atom_basis, activation=None)
            self.v_proj_p = InitDense(n_atom_basis, n_atom_basis, activation=None)

        self.out_proj = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.dropout = nn.Dropout(dropout)

        # Gate for vector features (equivariant gating)
        self.vec_gate = InitDense(n_atom_basis, n_atom_basis, activation=None)

        self.layer_norm = nn.LayerNorm(n_atom_basis)

    def forward(
        self,
        h_ts: Tensor,      # (N, 1, Z) TS scalar features
        mu_ts: Tensor,     # (N, L2, Z) TS vector features
        h_r: Optional[Tensor] = None,   # (N, 1, Z) R scalar features
        h_p: Optional[Tensor] = None,   # (N, 1, Z) P scalar features
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply cross-attention from TS to R and/or P features.

        Returns:
            Updated (h_ts, mu_ts) with cross-attended information.
        """
        # Remove the extra dimension for attention computation
        h_ts_2d = h_ts.squeeze(1)  # (N, Z)

        N, Z = h_ts_2d.shape

        # Query from TS
        q = self.q_proj(h_ts_2d).view(N, self.num_heads, self.head_dim)  # (N, H, D)

        attended_values = []

        # Attention to R
        if self.cross_type in ["r_to_ts", "bidirectional"] and h_r is not None:
            h_r_2d = h_r.squeeze(1)  # (N, Z)
            k_r = self.k_proj_r(h_r_2d).view(N, self.num_heads, self.head_dim)
            v_r = self.v_proj_r(h_r_2d).view(N, self.num_heads, self.head_dim)

            # Scaled dot-product attention (per-atom, same atom attends to itself)
            attn_scores_r = (q * k_r).sum(-1, keepdim=True) / math.sqrt(self.head_dim)
            attn_weights_r = torch.sigmoid(attn_scores_r)  # Use sigmoid for self-attention gating
            attended_r = (attn_weights_r * v_r).view(N, Z)
            attended_values.append(attended_r)

        # Attention to P
        if self.cross_type in ["p_to_ts", "bidirectional"] and h_p is not None:
            h_p_2d = h_p.squeeze(1)  # (N, Z)
            k_p = self.k_proj_p(h_p_2d).view(N, self.num_heads, self.head_dim)
            v_p = self.v_proj_p(h_p_2d).view(N, self.num_heads, self.head_dim)

            attn_scores_p = (q * k_p).sum(-1, keepdim=True) / math.sqrt(self.head_dim)
            attn_weights_p = torch.sigmoid(attn_scores_p)
            attended_p = (attn_weights_p * v_p).view(N, Z)
            attended_values.append(attended_p)

        if attended_values:
            # Average attended values if multiple sources
            attended = sum(attended_values) / len(attended_values)

            # Residual connection with layer norm
            h_out_2d = h_ts_2d + self.dropout(self.out_proj(attended))
            h_out_2d = self.layer_norm(h_out_2d)

            # Gate for vector features based on updated scalar features
            gate = torch.sigmoid(self.vec_gate(h_out_2d)).unsqueeze(1)  # (N, 1, Z)
            mu_out = mu_ts * gate

            h_out = h_out_2d.unsqueeze(1)  # (N, 1, Z)
        else:
            h_out, mu_out = h_ts, mu_ts

        return h_out, mu_out


class InitCondConditioning(nn.Module):
    """
    Additive conditioning based on displacement from initial condition (t=0 prior).

    For L0 (scalar features): adds distance embedding
    For L1 (vector features): adds direction * distance-dependent weight
    """

    def __init__(
        self,
        n_atom_basis: int,
        activation: Optional[Callable] = F.silu,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        # L0 (scalar) embedding: distance -> hidden
        self.emb_s = nn.Sequential(
            InitDense(1, n_atom_basis // 2, norm='layer', activation=activation),
            InitDense(n_atom_basis // 2, n_atom_basis, activation=None)
        )

        # L1 (vector) embedding: distance -> per-channel weight for direction
        self.emb_v = nn.Sequential(
            InitDense(1, n_atom_basis // 2, norm='layer', activation=activation),
            InitDense(n_atom_basis // 2, n_atom_basis, activation=None)
        )

    def reset_parameters(self):
        for layer in self.emb_s:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.emb_v:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
        self,
        h_N_1_Z: Tensor,
        mu_N_L2_Z: Tensor,
        x_t_N_3: Tensor,
        init_cond_N_3: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply additive init_cond conditioning.

        Args:
            h_N_1_Z: Scalar features (N, 1, Z)
            mu_N_L2_Z: Vector features (N, L2, Z)
            x_t_N_3: Current positions (N, 3)
            init_cond_N_3: Initial condition positions (N, 3) - t=0 prior

        Returns:
            Conditioned (h_N_1_Z, mu_N_L2_Z)
        """
        # Compute displacement from init_cond to current position
        displacement_N_3 = x_t_N_3 - init_cond_N_3  # (N, 3)
        d0_N_1 = torch.linalg.norm(displacement_N_3, dim=-1, keepdim=True)  # (N, 1)
        u0_N_3 = displacement_N_3 / (d0_N_1 + self.eps)  # (N, 3) unit direction

        # L0: add distance embedding to scalar features
        h_N_1_Z = h_N_1_Z + self.emb_s(d0_N_1).unsqueeze(1)  # (N, 1, Z)

        # L1: add direction * distance-dependent weight to L1 vector features
        dir_weight_N_1_Z = self.emb_v(d0_N_1).unsqueeze(1)  # (N, 1, Z)
        mu_l1 = mu_N_L2_Z[:, :3, :] + u0_N_3.unsqueeze(-1) * dir_weight_N_1_Z  # (N, 3, Z)
        mu_N_L2_Z = torch.cat([mu_l1, mu_N_L2_Z[:, 3:, :]], dim=1)

        return h_N_1_Z, mu_N_L2_Z


class MultiHeadGotenNet(nn.Module):
    """
    Multi-head GotenNet with shared backbone and per-head specialized layers.

    Architecture:
    - Shared embeddings (node, edge, time, CGR)
    - Shared backbone: n_shared_layers GATA+EQFF pairs
    - Per-head layers: n_head_layers GATA+EQFF pairs for each head (R, TS, P)
    - Optional cross-attention for TS head
    """

    def __init__(
            self,
            n_atom_basis: int = 128,
            n_atom_rdkit_feats: int = 28,
            n_shared_layers: int = 3,
            n_head_layers: int = 2,
            numerical_size_scale: float = 0.25,
            radial_basis: Union[Callable, str] = 'BesselBasis',
            n_rbf: int = 20,
            cutoff_fn: Optional[Union[Callable, str]] = None,
            edge_order: int = 4,
            activation: Optional[Union[Callable, str]] = F.silu,
            max_z: int = 100,
            epsilon: float = 1e-8,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            max_num_neighbors: int = 32,
            int_layer_norm="",
            int_vector_norm="",
            num_heads: int = 8,
            attn_dropout: float = 0.0,
            edge_updates=True,
            scale_edge=True,
            lmax: int = 2,
            aggr: str = "add",
            edge_ln: str = '',
            evec_dim=None,
            emlp_dim=None,
            sep_int_vec: bool = True,
            # Multi-head specific params
            active_heads: List[str] = None,
            use_cross_attention: bool = False,
            cross_attn_type: str = "bidirectional",
            # Initial condition conditioning
            use_init_cond: bool = False,
            # Backbone skip connections
            use_backbone_skip: bool = False,
    ):
        """
        Args:
            n_shared_layers: Number of shared backbone GATA+EQFF layers
            n_head_layers: Number of per-head GATA+EQFF layers
            active_heads: List of active heads ["R", "TS", "P"]
            use_cross_attention: Whether to use cross-attention for TS head
            cross_attn_type: Type of cross-attention ("r_to_ts", "p_to_ts", "bidirectional")
            use_init_cond: Whether to use initial condition (t=0 prior) conditioning via FiLM in GATA layers
            use_backbone_skip: Whether to add skip connections around GATA+EQFF blocks in backbone/head layers
        """
        super().__init__()

        if active_heads is None:
            active_heads = ["R", "TS", "P"]

        self.active_heads = active_heads
        self.n_shared_layers = n_shared_layers
        self.n_head_layers = n_head_layers
        self.use_cross_attention = use_cross_attention
        self.cross_attn_type = cross_attn_type

        # Initial condition conditioning flag
        self.use_init_cond = use_init_cond

        # Backbone skip connection flag
        self.use_backbone_skip = use_backbone_skip

        # Process initialization params
        self.scale_edge = scale_edge
        if isinstance(weight_init, str):
            log.info(f'Using {weight_init} weight initialization')
            weight_init = get_weight_init_by_string(weight_init)
        if isinstance(bias_init, str):
            bias_init = get_weight_init_by_string(bias_init)
        if isinstance(activation, str):
            activation = str2act(activation)

        self.numerical_size_scale = numerical_size_scale
        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.edge_order = edge_order
        self.lmax = lmax

        # Store layer creation params
        self._layer_params = {
            'n_atom_basis': n_atom_basis,
            'activation': activation,
            'weight_init': weight_init,
            'bias_init': bias_init,
            'aggr': aggr,
            'int_layer_norm': int_layer_norm,
            'int_vector_norm': int_vector_norm,
            'cutoff': self.cutoff,
            'epsilon': epsilon,
            'num_heads': num_heads,
            'attn_dropout': attn_dropout,
            'edge_updates': edge_updates,
            'scale_edge': scale_edge,
            'edge_ln': edge_ln,
            'evec_dim': evec_dim,
            'emlp_dim': emlp_dim,
            'sep_int_vec': sep_int_vec,
            'lmax': lmax,
        }

        # ===== Shared Components =====
        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)

        self.neighbor_embedding = NodeInit(
            [self.hidden_dim // 2, self.hidden_dim], n_atom_rdkit_feats, n_rbf,
            self.cutoff, max_z=max_z,
            weight_init=weight_init, bias_init=bias_init, concat=False,
            proj_ln='layer', activation=activation
        )
        self.edge_embedding = EdgeInit(
            n_rbf, [self.hidden_dim // 2, self.hidden_dim],
            weight_init=weight_init, bias_init=bias_init, proj_ln=''
        )

        self.time_embedding = TimestepEmbedding(128, 128, self.hidden_dim)

        # Index 0 = no recycling / cycle 0, 1..max = refinement cycles
        max_recycle = 8  # Support up to 8 recycle steps
        self.recycle_embedding = nn.Embedding(max_recycle + 1, self.hidden_dim)
        nn.init.zeros_(self.recycle_embedding.weight)  # Zero-init so recycling=0 is a no-op

        # Task embedding for explicit task token conditioning (Option 1)
        # "R": 0, "TS": 1, "P": 2
        self.task_map = {"R": 0, "TS": 1, "P": 2}
        self.task_embedding = nn.Embedding(len(self.task_map), self.hidden_dim)
        # Initialize with small magnitude to avoid shocking the network initially
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.1)

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, n_rbf=n_rbf)

        self.atom_cgr_embedding = AtomCGREmbedding(n_atom_rdkit_feats, n_atom_basis)
        self.edge_cgr_embedding = EdgeCGREmbedding(self.hidden_dim)

        self.tensor_init = TensorInit(l=lmax)

        # ===== Shared Backbone Layers =====
        self.shared_gata = nn.ModuleList([
            GATA(
                n_atom_basis=n_atom_basis, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=int_layer_norm, vector_norm=int_vector_norm,
                cutoff=self.cutoff, epsilon=epsilon,
                num_heads=num_heads, dropout=attn_dropout,
                edge_updates=edge_updates,
                last_layer=False,  # Never last layer in shared backbone
                scale_edge=scale_edge, edge_ln=edge_ln,
                evec_dim=evec_dim, emlp_dim=emlp_dim,
                sep_vecj=sep_int_vec, lmax=lmax,
            ) for _ in range(n_shared_layers)
        ])

        self.shared_eqff = nn.ModuleList([
            EQFF(
                n_atom_basis=n_atom_basis, activation=activation, epsilon=epsilon,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=int_layer_norm, vector_norm=int_vector_norm
            ) for _ in range(n_shared_layers)
        ])

        # Edge LayerNorm for numerical stability
        # Prevents edge embedding explosion through layers
        self.shared_edge_ln = nn.ModuleList([
            nn.LayerNorm(n_atom_basis) for _ in range(n_shared_layers)
        ])

        # --------- Per-Head Layers ---------
        self.head_gata = nn.ModuleDict()
        self.head_eqff = nn.ModuleDict()

        for head_name in ["R", "TS", "P"]:
            if head_name in active_heads:
                self.head_gata[head_name] = nn.ModuleList([
                    GATA(
                        n_atom_basis=n_atom_basis, activation=activation, aggr=aggr,
                        weight_init=weight_init, bias_init=bias_init,
                        layer_norm=int_layer_norm, vector_norm=int_vector_norm,
                        cutoff=self.cutoff, epsilon=epsilon,
                        num_heads=num_heads, dropout=attn_dropout,
                        edge_updates=edge_updates,
                        last_layer=(i == n_head_layers - 1),  # Last layer only for final
                        scale_edge=scale_edge, edge_ln=edge_ln,
                        evec_dim=evec_dim, emlp_dim=emlp_dim,
                        sep_vecj=sep_int_vec, lmax=lmax,
                    ) for i in range(n_head_layers)
                ])

                self.head_eqff[head_name] = nn.ModuleList([
                    EQFF(
                        n_atom_basis=n_atom_basis, activation=activation, epsilon=epsilon,
                        weight_init=weight_init, bias_init=bias_init,
                        layer_norm=int_layer_norm, vector_norm=int_vector_norm
                    ) for _ in range(n_head_layers)
                ])

        # Per-head edge LayerNorm
        self.head_edge_ln = nn.ModuleDict()
        for head_name in ["R", "TS", "P"]:
            if head_name in active_heads:
                self.head_edge_ln[head_name] = nn.ModuleList([
                    nn.LayerNorm(n_atom_basis) for _ in range(n_head_layers)
                ])

        # --------- Cross-Attention (optional) ---------
        if use_cross_attention and "TS" in active_heads:
            self.cross_attention = CrossAttentionBlock(
                n_atom_basis=n_atom_basis,
                num_heads=num_heads,
                cross_type=cross_attn_type,
                dropout=attn_dropout,
                weight_init=weight_init,
                bias_init=bias_init,
            )

        # --------- Initial Condition Conditioning ---------
        # Additive conditioning based on displacement from t=0 prior
        if use_init_cond:
            self.init_cond_conditioning = InitCondConditioning(
                n_atom_basis=n_atom_basis,
                activation=activation,
                weight_init=weight_init,
                bias_init=bias_init,
            )
            log.info("Initial condition conditioning: Enabled (additive embedding based on displacement from t=0 prior)")

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all parameters."""
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()

        for layer in self.shared_gata:
            layer.reset_parameters()
        for layer in self.shared_eqff:
            layer.reset_parameters()

        for head_name in self.active_heads:
            for layer in self.head_gata[head_name]:
                layer.reset_parameters()
            for layer in self.head_eqff[head_name]:
                layer.reset_parameters()

        # Reset edge LayerNorm modules
        for layer in self.shared_edge_ln:
            layer.reset_parameters()
        for head_name in self.active_heads:
            for layer in self.head_edge_ln[head_name]:
                layer.reset_parameters()

        # Reset init_cond conditioning module
        if self.use_init_cond:
            self.init_cond_conditioning.reset_parameters()

    def _compute_edge_context(
        self,
        x_t_N_3: Tensor,
        t_G: Tensor,
        inputs: Mapping[str, torch.Tensor],
        head_name: Optional[str] = None,
        recycle_idx: int = 0,
    ) -> dict:
        """
        Compute edge-related quantities and initial embeddings.
        This is shared computation that depends on positions.

        Args:
            x_t_N_3: Current positions
            t_G: Time per graph
            inputs: PyTorch Geometric batch data
            head_name: Which head ("R", "P", or "TS") - used for chirality features
        """
        edge_index, edge_type, batch_N = inputs.edge_index, inputs.edge_type, inputs.batch
        r_feat, p_feat, atom_type = inputs.r_feat, inputs.p_feat, inputs.atom_type

        if batch_N is None:
            batch_N = torch.zeros(len(x_t_N_3), dtype=torch.long, device=x_t_N_3.device)

        # Extend graph with radius edges
        edge_index, _, edge_type_r, edge_type_p = _extend_condensed_graph_edge(
            x_t_N_3, edge_index, edge_type, batch_N,
            cutoff=self.cutoff, edge_order=self.edge_order
        )

        # Edge features
        edge_vec_E_3 = x_t_N_3[edge_index[0]] - x_t_N_3[edge_index[1]]
        edge_weight_E = torch.norm(edge_vec_E_3, dim=-1)

        edge_attr_E_Rbf = self.radial_basis(edge_weight_E)

        N = x_t_N_3.size(0)
        Z = self.n_atom_basis

        # Node embeddings
        h_N_Z = self.atom_cgr_embedding(atom_type, r_feat, p_feat)
        h_N_Z = self.neighbor_embedding(
            atom_type, r_feat, p_feat, h_N_Z,
            edge_index, edge_weight_E, edge_attr_E_Rbf,
            edge_type_r, edge_type_p
        )

        # Time embedding
        t_emb_G_Z = self.time_embedding(t_G)
        t_emb_N_Z = t_emb_G_Z[batch_N]
        h_N_Z = h_N_Z + t_emb_N_Z

        # Recycle step embedding (zero-init: no effect when recycle_idx=0)
        recycle_idx_clamped = min(recycle_idx, self.recycle_embedding.num_embeddings - 1)
        recycle_emb_Z = self.recycle_embedding(
            torch.tensor([recycle_idx_clamped], device=h_N_Z.device)
        )  # (1, Z)
        h_N_Z = h_N_Z + recycle_emb_Z  # Broadcast to (N, Z)
        
        # Task Token Embedding
        # Inject explicit information about which geometry (R, TS, or P) we are processing
        if head_name in self.task_map:
            task_idx = torch.tensor([self.task_map[head_name]], device=h_N_Z.device)
            task_emb_Z = self.task_embedding(task_idx)  # (1, Z)
            h_N_Z = h_N_Z + task_emb_Z  # Broadcast addition to (N, Z)
            

        # Edge embeddings
        edge_emb_E_Z = self.edge_embedding(edge_index, edge_attr_E_Rbf, edge_type_r, edge_type_p)
        edge_emb_E_Z = edge_emb_E_Z + t_emb_N_Z[edge_index[0]]
        edge_emb_E_Z = edge_emb_E_Z + recycle_emb_Z  # Broadcast (1, Z) to (E, Z)

        # Normalize edge vectors to unit vectors
        mask_Eu = edge_index[0] != edge_index[1]
        dist_Eu = edge_weight_E[mask_Eu].unsqueeze(1)
        eps = 1e-6
        edge_vec_E_3[mask_Eu] = edge_vec_E_3[mask_Eu] / (dist_Eu + eps)

        edge_vec_E_3 = self.tensor_init(edge_vec_E_3)

        # Edge counts
        num_edges = scatter(torch.ones_like(edge_weight_E), edge_index[0], dim=0, reduce="sum")
        num_edges_expanded_E = num_edges[edge_index[0]]

        L2 = ((self.tensor_init.l + 1) ** 2) - 1

        return {
            'h_N_Z': h_N_Z,
            'edge_index': edge_index,
            'edge_vec_E_3': edge_vec_E_3,
            'edge_emb_E_Z': edge_emb_E_Z,
            'edge_weight_E': edge_weight_E,
            'num_edges_expanded_E': num_edges_expanded_E,
            'N': N,
            'Z': Z,
            'L2': L2,
            'x_t_N_3': x_t_N_3,  # Store current positions for init_cond conditioning
        }

    def _run_shared_backbone(
        self,
        h_N_Z: Tensor,
        edge_ctx: dict,
        init_cond_N_3: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run shared backbone layers.

        Args:
            h_N_Z: Initial node embeddings
            edge_ctx: Edge context dict from _compute_edge_context
            init_cond_N_3: Initial condition positions (t=0 prior) for additive conditioning

        Returns:
            h_N_1_Z: Scalar features (N, 1, Z)
            mu_N_L2_Z: Vector features (N, L2, Z)
            edge_emb_E_Z: Updated edge embeddings
        """
        N, Z, L2 = edge_ctx['N'], edge_ctx['Z'], edge_ctx['L2']
        edge_index = edge_ctx['edge_index']
        edge_vec_E_3 = edge_ctx['edge_vec_E_3']
        edge_emb_E_Z = edge_ctx['edge_emb_E_Z']
        edge_weight_E = edge_ctx['edge_weight_E']
        num_edges_expanded_E = edge_ctx['num_edges_expanded_E']
        x_t_N_3 = edge_ctx['x_t_N_3']

        mu_N_L2_Z = torch.zeros((N, L2, Z), device=h_N_Z.device)
        h_N_1_Z = h_N_Z.unsqueeze(1)

        # Apply init_cond conditioning
        if self.use_init_cond and init_cond_N_3 is not None:
            h_N_1_Z, mu_N_L2_Z = self.init_cond_conditioning(
                h_N_1_Z, mu_N_L2_Z, x_t_N_3, init_cond_N_3
            )

        for gata, eqff, edge_ln in zip(self.shared_gata, self.shared_eqff, self.shared_edge_ln):
            # Store input for backbone-level skip connection
            if self.use_backbone_skip:
                h_in, mu_in = h_N_1_Z, mu_N_L2_Z

            h_N_1_Z, mu_N_L2_Z, edge_emb_E_Z = gata(
                edge_index,
                h_N_1_Z,
                mu_N_L2_Z,
                edge_vec_E_3=edge_vec_E_3,
                edge_emb_E_Z=edge_emb_E_Z,
                edge_weight_E=edge_weight_E,
                num_edges_expanded_E=num_edges_expanded_E,
            )

            # Apply LayerNorm to edge embeddings to prevent explosion
            edge_emb_E_Z = edge_ln(edge_emb_E_Z)

            h_N_1_Z, mu_N_L2_Z = eqff(h_N_1_Z, mu_N_L2_Z)

            # Backbone-level skip connection (in addition to internal GATA/EQFF skips)
            if self.use_backbone_skip:
                h_N_1_Z = h_in + h_N_1_Z
                mu_N_L2_Z = mu_in + mu_N_L2_Z

            h_N_1_Z = h_N_1_Z * self.numerical_size_scale
            mu_N_L2_Z = mu_N_L2_Z * self.numerical_size_scale
            edge_emb_E_Z = edge_emb_E_Z * self.numerical_size_scale

        return h_N_1_Z, mu_N_L2_Z, edge_emb_E_Z

    def _run_head_layers(
        self,
        head_name: str,
        h_N_1_Z: Tensor,
        mu_N_L2_Z: Tensor,
        edge_ctx: dict,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run per-head layers.

        Args:
            head_name: Name of the head ("R", "TS", or "P")
            h_N_1_Z: Scalar features from shared backbone
            mu_N_L2_Z: Vector features from shared backbone
            edge_ctx: Edge context dict

        Returns:
            h_N_Z: Final scalar features (N, Z)
            mu_N_L2_Z: Final vector features (N, L2, Z)
        """
        edge_index = edge_ctx['edge_index']
        edge_vec_E_3 = edge_ctx['edge_vec_E_3']
        edge_emb_E_Z = edge_ctx['edge_emb_E_Z']
        edge_weight_E = edge_ctx['edge_weight_E']
        num_edges_expanded_E = edge_ctx['num_edges_expanded_E']

        gata_layers = self.head_gata[head_name]
        eqff_layers = self.head_eqff[head_name]
        edge_ln_layers = self.head_edge_ln[head_name]

        for gata, eqff, edge_ln in zip(gata_layers, eqff_layers, edge_ln_layers):
            # Store input for backbone-level skip connection
            if self.use_backbone_skip:
                h_in, mu_in = h_N_1_Z, mu_N_L2_Z

            h_N_1_Z, mu_N_L2_Z, edge_emb_E_Z = gata(
                edge_index,
                h_N_1_Z,
                mu_N_L2_Z,
                edge_vec_E_3=edge_vec_E_3,
                edge_emb_E_Z=edge_emb_E_Z,
                edge_weight_E=edge_weight_E,
                num_edges_expanded_E=num_edges_expanded_E,
            )

            # Apply LayerNorm to edge embeddings to prevent explosion
            edge_emb_E_Z = edge_ln(edge_emb_E_Z)

            h_N_1_Z, mu_N_L2_Z = eqff(h_N_1_Z, mu_N_L2_Z)

            # Backbone-level skip connection (in addition to internal GATA/EQFF skips)
            if self.use_backbone_skip:
                h_N_1_Z = h_in + h_N_1_Z
                mu_N_L2_Z = mu_in + mu_N_L2_Z

            h_N_1_Z = h_N_1_Z * self.numerical_size_scale
            mu_N_L2_Z = mu_N_L2_Z * self.numerical_size_scale
            edge_emb_E_Z = edge_emb_E_Z * self.numerical_size_scale  # Prevent edge explosion

        h_N_Z = h_N_1_Z.squeeze(1)
        return h_N_Z, mu_N_L2_Z

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Handle loading checkpoints from before recycle_embedding was added.

        Missing keys for zero-initialized parameters (recycle_embedding) are
        silently skipped since the default zeros already act as a no-op.
        """
        # Keys that are zero-initialized and safe to skip if missing
        zero_init_prefixes = ("recycle_embedding.",)
        missing = [k for k in self.state_dict() if k not in state_dict]
        safe_missing = [k for k in missing if any(k.startswith(p) for p in zero_init_prefixes)]

        if safe_missing:
            print(f"[MultiHeadGotenNet] Checkpoint missing {safe_missing} — using zero-init defaults")
            strict = False

        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(
        self,
        x_t_dict: Dict[str, Tensor],
        t_G: Tensor,
        inputs: Mapping[str, torch.Tensor],
        init_cond_dict: Optional[Dict[str, Tensor]] = None,
        recycle_idx: int = 0,
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Forward pass for all active heads.

        Each head processes its own x_t through the network because edge indices
        and distances depend on the current positions.

        Args:
            x_t_dict: Dict mapping head names to position tensors {"R": x_t_R, "TS": x_t_TS, "P": x_t_P}
            t_G: Time per graph (shared across heads)
            inputs: PyTorch Geometric batch data
            init_cond_dict: Optional dict mapping head names to initial condition positions (t=0 prior, no noise).
                           Used for FiLM conditioning when use_init_cond=True.

        Returns:
            Dict mapping head names to (h_N_Z, mu_N_L2_Z) tuples
        """
        outputs = {}
        shared_features = {}
        edge_contexts = {}

        # Only process heads that are present in x_t_dict
        heads_to_process = [h for h in self.active_heads if h in x_t_dict]

        # Run shared backbone for each head's geometry
        for head_name in heads_to_process:
            x_t = x_t_dict[head_name]

            # Get init_cond for this head if provided
            init_cond_N_3 = None
            if init_cond_dict is not None and head_name in init_cond_dict:
                init_cond_N_3 = init_cond_dict[head_name]

            # Compute edge context (depends on positions)
            edge_ctx = self._compute_edge_context(x_t, t_G, inputs, head_name, recycle_idx=recycle_idx)
            edge_contexts[head_name] = edge_ctx

            # Run shared backbone
            h_shared, mu_shared, edge_emb_updated = self._run_shared_backbone(
                edge_ctx['h_N_Z'], edge_ctx, init_cond_N_3=init_cond_N_3
            )

            # Store for potential cross-attention
            # Update edge context with updated edge embeddings
            edge_ctx['edge_emb_E_Z'] = edge_emb_updated
            shared_features[head_name] = (h_shared, mu_shared)

        # Optional cross-attention for TS head
        if self.use_cross_attention and "TS" in heads_to_process:
            h_ts, mu_ts = shared_features["TS"]

            h_r, _ = shared_features.get("R", (None, None))
            h_p, _ = shared_features.get("P", (None, None))

            h_ts, mu_ts = self.cross_attention(
                h_ts, mu_ts, h_r, h_p
            )
            shared_features["TS"] = (h_ts, mu_ts)

        # Run per-head layers
        for head_name in heads_to_process:
            h_shared, mu_shared = shared_features[head_name]
            edge_ctx = edge_contexts[head_name]

            h_out, mu_out = self._run_head_layers(
                head_name, h_shared, mu_shared, edge_ctx
            )

            outputs[head_name] = (h_out, mu_out)

        return outputs

