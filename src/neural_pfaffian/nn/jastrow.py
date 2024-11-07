from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from neural_pfaffian.nn.module import ReparamModule
from neural_pfaffian.nn.utils import MLP, ActivationOrName
from neural_pfaffian.nn.wave_function import ElecEmbedding, JastrowP
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules


class MLPJastrow(ReparamModule, JastrowP):
    hidden_dims: Sequence[int]
    activation: ActivationOrName

    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: ElecEmbedding):
        out = MLP((*self.hidden_dims, 1), self.activation)(elec_embeddings)
        out = jnp.squeeze(out, axis=-1)
        out = jax.ops.segment_sum(out, systems.electron_molecule_mask, systems.n_mols)
        out *= self.param('weight', jax.nn.initializers.zeros, (), jnp.float32)
        return jnp.ones_like(out), out


class CuspJastrow(ReparamModule, JastrowP):
    @nn.compact
    def __call__(self, systems: Systems, elec_embeddings: ElecEmbedding):
        w_par, w_anti = self.param('weight', jax.nn.initializers.zeros, (2,), jnp.float32)
        a_par, a_anti = self.param('alpha', jax.nn.initializers.ones, (2,), jnp.float32)

        dists = systems.elec_elec_dists[..., -1]
        dists_same, dists_diff = (
            dists[: systems.n_elec_pair_same],
            dists[systems.n_elec_pair_same :],
        )
        segments = systems.elec_elec_idx[-1]
        seg_same, seg_diff = (
            segments[: systems.n_elec_pair_same],
            segments[systems.n_elec_pair_same :],
        )

        result = jnp.zeros((), elec_embeddings.dtype)
        result += w_par * jax.ops.segment_sum(
            -(1 / 4) * a_par**2 / (a_par + dists_same), seg_same, systems.n_mols
        )
        result += w_anti * jax.ops.segment_sum(
            -(1 / 2) * a_anti**2 / (a_anti + dists_diff), seg_diff, systems.n_mols
        )
        return jnp.zeros_like(result), result


JASTROWS = Modules[JastrowP](
    {
        cls.__name__.lower().replace('jastrow', ''): cls
        for cls in [MLPJastrow, CuspJastrow]
    }
)
