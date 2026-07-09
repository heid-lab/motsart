from typing import Union, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.utils import scatter

from goflow.gotennet.models.components.ops import Dense, shifted_softplus, str2act, VecLayerNorm
from goflow.gotennet.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SNNDense(nn.Linear):
    """Fully connected linear layer with activation function."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Union[Callable, nn.Module] = None,
            weight_init: Callable = xavier_uniform_,
            bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features
    for tensorial properties.
    """

    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation=F.silu, sactivation=None):
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = SNNDense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def reset_parameters(self):
        """Reset all parameters."""
        self.mix_vectors.reset_parameters()
        for layer in self.scalar_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, scalars, vectors):
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class Atomwise3DOut(nn.Module):
    def __init__(
            self,
            n_in,
            n_hidden=None,
            activation=shifted_softplus
    ):
        super().__init__()

        if type(activation) is str:
            activation = str2act(activation)

        self.out_net = nn.ModuleList(
            [
                GatedEquivariantBlock(n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden, n_hidden=n_hidden,
                                      activation=activation,
                                      sactivation=activation),
                GatedEquivariantBlock(n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                                      n_hidden=n_hidden, activation=activation)
            ])

        self.vec_norm = VecLayerNorm(n_in, trainable=False, norm_type="rms")

    def forward(self, l0, l1):
        l1 = self.vec_norm(l1)

        for eqiv_layer in self.out_net:
            l0, l1 = eqiv_layer(l0, l1)

        return l1.squeeze()


# =============================================================================
# Activation Energy Prediction Heads
# =============================================================================

class ActivationEnergyHeadMLP(nn.Module):
    """
    Simple MLP-based activation energy prediction head.

    Takes L0 (scalar) feature differences between TS and R/P,
    produces per-atom energy contributions, and sums over atoms.

    Architecture:
        diff_L0_N_Z → MLP → per-atom energy (N,) → scatter_sum → (G,) activation energy
    """

    def __init__(
            self,
            n_in: int,
            n_hidden: int = 128,
            n_layers: int = 2,
            activation: Union[Callable, str] = F.silu,
    ):
        """
        Args:
            n_in: Input feature dimension (hidden_dim of representation)
            n_hidden: Hidden layer dimension
            n_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        if isinstance(activation, str):
            activation = str2act(activation)

        layers = []
        in_dim = n_in
        for _ in range(n_layers):
            layers.append(Dense(in_dim, n_hidden, activation=activation))
            in_dim = n_hidden
        # Final layer outputs scalar (no activation)
        layers.append(Dense(n_hidden, 1, activation=None))

        self.mlp = nn.Sequential(*layers)

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
            self,
            diff_l0_N_Z: Tensor,
            batch: Tensor,
    ) -> Tensor:
        """
        Compute activation energy from L0 feature differences.

        Args:
            diff_l0_N_Z: L0 feature difference (TS - R or TS - P), shape (N, Z)
            batch: Batch indices for each atom, shape (N,)

        Returns:
            Activation energy per graph, shape (G,)
        """
        # Per-atom energy contribution
        atom_energy_N_1 = self.mlp(diff_l0_N_Z)  # (N, 1)
        atom_energy_N = atom_energy_N_1.squeeze(-1)  # (N,)

        # Sum over atoms per graph
        energy_G = scatter(atom_energy_N, batch, dim=0, reduce='sum')  # (G,)

        return energy_G


class ActivationEnergyHeadEquivariant(nn.Module):
    """
    Equivariant activation energy prediction head using GatedEquivariantBlocks.

    Takes L0 (scalar) and L1 (vector) feature differences between TS and R/P,
    processes them through equivariant blocks, then outputs per-atom energies
    that are summed to get activation energy.

    Architecture:
        diff_L0_N_Z, diff_L1_N_3_Z → GatedEquivariantBlock(s) → L0 → Dense →
        per-atom energy (N,) → scatter_sum → (G,) activation energy
    """

    def __init__(
            self,
            n_in: int,
            n_hidden: int = 128,
            n_layers: int = 1,
            activation: Union[Callable, str] = F.silu,
    ):
        """
        Args:
            n_in: Input feature dimension (hidden_dim of representation)
            n_hidden: Hidden layer dimension
            n_layers: Number of GatedEquivariantBlock layers
            activation: Activation function
        """
        super().__init__()

        if isinstance(activation, str):
            activation = str2act(activation)

        self.vec_norm = VecLayerNorm(n_in, trainable=False, norm_type="rms")

        # Build equivariant layers
        self.eq_layers = nn.ModuleList()
        in_dim = n_in
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            out_dim = n_hidden if not is_last else n_hidden
            self.eq_layers.append(
                GatedEquivariantBlock(
                    n_sin=in_dim,
                    n_vin=in_dim,
                    n_sout=out_dim,
                    n_vout=out_dim,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation if not is_last else None,
                )
            )
            in_dim = out_dim

        # Final MLP to get per-atom energy from scalar features
        self.energy_mlp = nn.Sequential(
            Dense(n_hidden, n_hidden, activation=activation),
            Dense(n_hidden, 1, activation=None),
        )

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.eq_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.energy_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
            self,
            diff_l0_N_Z: Tensor,
            diff_l1_N_3_Z: Tensor,
            batch: Tensor,
    ) -> Tensor:
        """
        Compute activation energy from L0 and L1 feature differences.

        Args:
            diff_l0_N_Z: L0 (scalar) feature difference, shape (N, Z)
            diff_l1_N_3_Z: L1 (vector) feature difference, shape (N, 3, Z)
            batch: Batch indices for each atom, shape (N,)

        Returns:
            Activation energy per graph, shape (G,)
        """
        l0 = diff_l0_N_Z
        l1 = self.vec_norm(diff_l1_N_3_Z)

        # Process through equivariant layers
        for eq_layer in self.eq_layers:
            l0, l1 = eq_layer(l0, l1)

        # Get per-atom energy from scalar features
        atom_energy_N_1 = self.energy_mlp(l0)  # (N, 1)
        atom_energy_N = atom_energy_N_1.squeeze(-1)  # (N,)

        # Sum over atoms per graph
        energy_G = scatter(atom_energy_N, batch, dim=0, reduce='sum')  # (G,)

        return energy_G
