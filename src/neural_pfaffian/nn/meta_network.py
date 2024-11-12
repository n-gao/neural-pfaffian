import math
from dataclasses import KW_ONLY

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float, PyTree

from neural_pfaffian.nn.edges import BesselRbf, EdgeEmbedding, NormEnvelope
from neural_pfaffian.nn.module import ParamMeta, ParamTypes
from neural_pfaffian.nn.ops import segment_mean, segment_sum
from neural_pfaffian.nn.utils import Activation, ActivationOrName, GatedLinearUnit
from neural_pfaffian.nn.wave_function import MetaNetworkP
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules


class MessagePassing(nn.Module):
    msg_dim: int
    out_dim: int
    activation: ActivationOrName

    @nn.compact
    def __call__(
        self,
        s_embed: Float[Array, 'n_nuc embedding_dim'],
        r_embed: Float[Array, 'n_nuc embedding_dim'],
        e_embed: Float[Array, 'n_nn message_dim'],
        norm: Float[Array, ' n_nn'],
        senders: npt.NDArray[np.int64],
        receivers: npt.NDArray[np.int64],
    ) -> jax.Array:
        activation = Activation(self.activation)
        inp = (
            nn.Dense(self.msg_dim)(s_embed)[senders]
            + nn.Dense(self.msg_dim, use_bias=False)(r_embed)[receivers]
        )
        inp = nn.LayerNorm()(inp)
        inp = activation(inp)
        inp *= nn.Dense(self.msg_dim, use_bias=False)(e_embed)
        msg = (
            jax.ops.segment_sum(inp, receivers, num_segments=r_embed.shape[0])
            * norm[:, None]
        )
        return activation(nn.Dense(self.out_dim, use_bias=False)(msg))


class Update(nn.Module):
    out_dim: int
    activation: ActivationOrName

    @nn.compact
    def __call__(self, n_embed: jax.Array, msg: jax.Array) -> jax.Array:
        y = GatedLinearUnit(self.out_dim, self.activation)(n_embed)
        y += msg  # message
        y = GatedLinearUnit(self.out_dim, self.activation)(y)
        y += n_embed  # residual
        return y


class MessagePassingNetwork(nn.Module):
    message_dim: int
    embedding_dim: int
    num_layers: int
    activation: ActivationOrName

    @nn.compact
    def __call__(
        self,
        s_embed: Float[Array, 'n_nuc embedding_dim'],
        r_embed: Float[Array, 'n_nuc embedding_dim'] | None,
        e_embed: Float[Array, 'n_nn message_dim'],
        e_norm: Float[Array, ' n_nn'],
        senders: npt.NDArray[np.int64],
        receivers: npt.NDArray[np.int64],
    ) -> Float[Array, 'n_nuc embedding_dim']:
        if r_embed is None:
            r_embed = s_embed

        norm = 1 / (segment_sum(e_norm, receivers, r_embed.shape[0]) + 1)

        embeddings = []
        for _ in range(self.num_layers):
            msg = MessagePassing(self.message_dim, self.embedding_dim, self.activation)(
                s_embed,
                r_embed,
                e_embed,
                norm,
                senders,
                receivers,
            )
            r_embed = Update(self.embedding_dim, self.activation)(r_embed, msg)
            embeddings.append(r_embed)
        return nn.Dense(self.embedding_dim)(jnp.concatenate(embeddings, axis=-1))


class OutputBias(nn.Module):
    meta: ParamMeta
    n_charges: int

    @nn.compact
    def __call__(self, systems: Systems) -> Array:
        shape = self.meta.shape_and_dtype.shape
        dtype = self.meta.shape_and_dtype.dtype
        out_dim = math.prod(shape)
        if not self.meta.bias:
            return jnp.zeros(shape, dtype=dtype)

        match self.meta.param_type:
            case ParamTypes.NUCLEI:
                bias = nn.Embed(self.n_charges, out_dim, dtype)(systems.flat_charges)
            case ParamTypes.NUCLEI_NUCLEI:
                idx_i, idx_j = systems.nuc_nuc_idx[:2]
                s_emb = nn.Embed(self.n_charges, out_dim, dtype)(systems.flat_charges)
                r_emb = nn.Embed(self.n_charges, out_dim, dtype)(systems.flat_charges)
                bias = (s_emb[idx_i] + r_emb[idx_j]) / jnp.sqrt(2)
            case ParamTypes.GLOBAL:
                bias = nn.Embed(1, out_dim, dtype)(
                    jnp.zeros((systems.n_mols,), dtype=jnp.int64)
                )
            case _:
                raise ValueError(f'Unknown param type {self.meta.param_type}')
        return bias.reshape(-1, *shape) * jnp.asarray(self.meta.std, dtype=dtype)


class ParamOut(nn.Module):
    meta: ParamMeta
    activation: ActivationOrName
    n_charges: int

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        n_embed: Float[Array, 'n_nuc embedding_dim'],
        nn_embed: Float[Array, 'n_nn message_dim'],
        g_embed: Float[Array, 'n_mols embedding_dim'],
    ) -> Array:
        shape = self.meta.shape_and_dtype.shape
        dtype = self.meta.shape_and_dtype.dtype
        out_dim = math.prod(shape)
        if self.meta.chunk_axis is not None:
            chunk_axis = self.meta.chunk_axis % len(shape)
            segments = shape[chunk_axis]
        else:
            chunk_axis = None
            segments = 1
        seg_out = out_dim // segments

        # Output bias
        bias = OutputBias(self.meta, self.n_charges)(systems)
        # Input selection
        match self.meta.param_type:
            case ParamTypes.NUCLEI:
                inp = n_embed
            case ParamTypes.NUCLEI_NUCLEI:
                inp = nn_embed
            case ParamTypes.GLOBAL:
                inp = g_embed
            case _:
                raise ValueError(f'Unknown param type {self.meta.param_type}')
        inp_dim = inp.shape[-1]

        # We can't shift if we have no bias. Thus, we scale for segments instead
        if segments > 1:
            assert self.meta.bias, 'Segments only supported with bias'
            # Shift for different segments
            segment_meta = self.meta.replace(
                shape_and_dtype=jax.ShapeDtypeStruct((segments, inp_dim), dtype),
                std=1.0,
                bias=True,
            )
            inp = inp[..., None, :] + OutputBias(segment_meta, self.n_charges)(systems)
        # Compute output
        result = GatedLinearUnit(
            seg_out, self.activation, inp_dim, normalize=self.meta.bias
        )(inp)
        # Reshape to output shape
        if chunk_axis is not None:
            # Move the segments into the right dimension
            target_shape = list(shape)
            del target_shape[chunk_axis]
            result = result.reshape(-1, segments, *target_shape)
            result = jnp.moveaxis(result, 1, chunk_axis + 1)
        else:
            result = result.reshape(-1, *shape)
        # Scale std
        result *= self.param(
            'std', jax.nn.initializers.constant(self.meta.std, jnp.float32), ()
        )
        # Add bias
        result = result + bias / jnp.sqrt(2)
        # Add mean
        result += jnp.asarray(self.meta.mean, dtype=dtype)
        return result


class GraphToParameters(nn.Module):
    out_structure: PyTree[ParamMeta]
    activation: ActivationOrName
    n_charges: int

    @nn.compact
    def __call__(
        self,
        systems: Systems,
        n_embed: Float[Array, 'n_nuc embedding_dim'],
        e_embed: Float[Array, 'n_nn message_dim'],
    ) -> PyTree[Array]:
        emb_dim = n_embed.shape[-1]
        # Prepare inputs
        nn_idx_i, nn_idx_j = systems.nuc_nuc_idx[:2]
        nn_embed = (
            nn.Dense(emb_dim)(n_embed)[nn_idx_i]
            + nn.Dense(emb_dim, use_bias=False)(n_embed)[nn_idx_j]
        ) / jnp.sqrt(2)
        nn_embed = Activation(self.activation)(nn_embed)
        g_embed = segment_mean(n_embed, systems.nuclei_molecule_mask, systems.n_mols)

        # Update
        n_embed = GatedLinearUnit(emb_dim, self.activation)(n_embed)
        nn_embed = GatedLinearUnit(emb_dim, self.activation)(nn_embed)
        g_embed = GatedLinearUnit(emb_dim, self.activation)(g_embed)

        # Normalize
        n_embed = nn.LayerNorm()(n_embed)
        nn_embed = nn.LayerNorm()(nn_embed)
        g_embed = nn.LayerNorm()(g_embed)

        # Scale nn_embed by edge embedding
        nn_embed *= nn.Dense(emb_dim, use_bias=False)(e_embed)

        def predict_param(path, meta: ParamMeta):
            key_path = ''.join(map(str, path))
            name = f'ParamOut_{meta.param_type.name}_{key_path}'
            return ParamOut(meta, self.activation, self.n_charges, name=name)(
                systems, n_embed, nn_embed, g_embed
            )

        result = jtu.tree_map_with_path(
            predict_param, self.out_structure, is_leaf=lambda x: isinstance(x, ParamMeta)
        )
        return result


class MetaGNN(nn.Module, MetaNetworkP):
    # Message passing
    message_dim: int
    embedding_dim: int
    num_layers: int
    activation: ActivationOrName

    # Edge embedding
    n_rbf: int

    _: KW_ONLY  # Will be set later
    # Output structure
    out_structure: PyTree[ParamMeta] | None = None
    # Set of charges that are needed
    charges: tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, systems: Systems) -> PyTree[Array]:
        assert self.charges is not None, 'Charges must be set'
        n_charges = len(self.charges)
        # We replace the charges in systems with "Pseudo" charges that are compact
        charge_to_idx = {charge: idx for idx, charge in enumerate(self.charges)}
        systems = systems.replace(
            charges=jtu.tree_map(lambda x: charge_to_idx[x], systems.charges),
        )
        flat_charges = systems.flat_charges

        # Node embeddings
        n_embed = nn.Embed(
            num_embeddings=n_charges,
            features=self.embedding_dim,
        )(flat_charges)

        # Edge embeddings
        edges = systems.nuc_nuc_dists
        e_ctr_idx = systems.nuc_nuc_idx[0]
        e_embed = EdgeEmbedding(
            self.message_dim,
            self.message_dim,
            self.n_rbf,
            self.activation,
            BesselRbf,
            n_charges,
        )(systems, edges, e_ctr_idx)
        e_norm = NormEnvelope(n_charges)(systems, edges, e_ctr_idx)

        # Message passing
        n_embed = MessagePassingNetwork(
            self.message_dim, self.embedding_dim, self.num_layers, self.activation
        )(n_embed, None, e_embed, e_norm, *systems.nuc_nuc_idx[:2])

        # Readout
        return GraphToParameters(
            self.out_structure,
            self.activation,
            n_charges,
        )(systems, n_embed, e_embed)


META_NETWORKS = Modules[MetaNetworkP]({'meta_gnn': MetaGNN})
