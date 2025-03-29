from typing import Protocol

import jax.numpy as jnp
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float

from neural_pfaffian.utils import Modules
from neural_pfaffian.utils.jax_utils import pgather_if_pmap, pmean_if_pmap, vmap

LocalEnergies = Float[Array, ' batch_size n_mols']
LocalEnergiesPerMol = Float[Array, ' batch_size n_mols']


class Clipping(Protocol):
    def __call__(self, local_energies: LocalEnergiesPerMol) -> LocalEnergiesPerMol: ...


class NoneClipping(Clipping, PyTreeNode):
    def __call__(self, local_energies: LocalEnergiesPerMol) -> LocalEnergiesPerMol:
        return local_energies


class MeanClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)

    @vmap(in_axes=-1, out_axes=-1)
    def __call__(self, local_energies: LocalEnergies) -> LocalEnergies:
        center = pmean_if_pmap(jnp.mean(local_energies))
        dev = pmean_if_pmap(jnp.abs(local_energies - center).mean())
        max_dev = self.max_deviation * dev
        return jnp.clip(local_energies, center - max_dev, center + max_dev)


class MedianClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)

    @vmap(in_axes=-1, out_axes=-1)
    def __call__(self, local_energies: LocalEnergies) -> LocalEnergies:
        full_e = pgather_if_pmap(local_energies, axis=0, tiled=True)
        center = jnp.median(full_e)
        dev = jnp.abs(full_e - center).mean()
        max_dev = self.max_deviation * dev
        return jnp.clip(local_energies, center - max_dev, center + max_dev)


class QuantileClipping(Clipping, PyTreeNode):
    max_deviation: float = field(pytree_node=False)
    quantile: float = field(pytree_node=False)

    @vmap(in_axes=-1, out_axes=-1)
    def __call__(self, local_energies: LocalEnergies) -> LocalEnergies:
        full_e = pgather_if_pmap(local_energies, axis=0, tiled=True)
        center = jnp.median(full_e)
        abs_diffs = jnp.abs(full_e - center)
        quantile = jnp.quantile(abs_diffs, self.quantile)
        max_dev = self.max_deviation * quantile
        return jnp.clip(local_energies, center - max_dev, center + max_dev)


CLIPPINGS = Modules[Clipping](
    {
        cls.__name__.lower().replace('clipping', ''): cls
        for cls in [NoneClipping, MeanClipping, MedianClipping, QuantileClipping]
    }
)
