from dataclasses import KW_ONLY

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array

from neural_pfaffian.nn.module import ParamTypes, ReparamModule
from neural_pfaffian.nn.utils import normal_init
from neural_pfaffian.systems import Systems, chunk_electron_nuclei
from neural_pfaffian.utils import Modules


class Envelope(ReparamModule):
    _: KW_ONLY
    # These will all be set by the orbital module
    out_dim: int = 0
    pi_init: float = 1
    # If True, multiplies the output dimension by the number of nuclei in the molecule
    out_per_nuc: bool = False
    keep_distr: bool = False

    def __call__(self, systems: Systems) -> list[Array]: ...


class FullEnvelope(Envelope):
    @nn.compact
    def __call__(self, systems: Systems):
        param_type = ParamTypes.NUCLEI_NUCLEI if self.out_per_nuc else ParamTypes.NUCLEI
        leading_dim = systems.n_nn if self.out_per_nuc else systems.n_nuc
        pi, pi_meta = self.reparam(
            'pi',
            normal_init(0.0, self.pi_init),
            (leading_dim, self.out_dim),
            param_type=param_type,
            bias=False,
            keep_distr=self.keep_distr,
        )
        sigma, sigma_meta = self.reparam(
            'sigma',
            normal_init(1.0, 0.1),
            (leading_dim, self.out_dim),
            param_type=param_type,
            bias=True,
            keep_distr=self.keep_distr,
        )
        sigma = nn.softplus(sigma)
        result: list[Array] = []
        for pi, sigma, elec_nuc, (spins, charges) in zip(
            systems.group(pi, pi_meta.param_type.value.chunk_fn),
            systems.group(sigma, sigma_meta.param_type.value.chunk_fn),
            systems.group(systems.elec_nuc_dists, chunk_electron_nuclei),
            systems.unique_spins_and_charges,
        ):
            n_elec, n_nuc = sum(spins), len(charges)
            out_dim = self.out_dim * n_nuc if self.out_per_nuc else self.out_dim

            @jax.vmap  # vmap over different molecules
            def _envelopes(pi, sigma, elec_nuc):
                pi = pi.reshape(n_nuc, out_dim)
                sigma = sigma.reshape(n_nuc, out_dim)
                elec_nuc = elec_nuc.reshape(n_elec, n_nuc, 4)[..., -1]
                envelopes = jnp.exp(-jnp.einsum('en,no->eno', elec_nuc, sigma))
                return jnp.einsum('eno,no->eo', envelopes, pi)

            result.append(_envelopes(pi, sigma, elec_nuc))
        return result


class EfficientEnvelope(Envelope):
    env_per_nuc: int

    @nn.compact
    def __call__(self, systems: Systems):
        param_type = ParamTypes.NUCLEI_NUCLEI if self.out_per_nuc else ParamTypes.NUCLEI
        leading_dim = systems.n_nn if self.out_per_nuc else systems.n_nuc
        pi, pi_meta = self.reparam(
            'pi',
            normal_init(0.0, self.pi_init),
            (leading_dim, self.out_dim, self.env_per_nuc),
            param_type=param_type,
            bias=False,
            keep_distr=self.keep_distr,
        )
        sigma, sigma_meta = self.reparam(
            'sigma',
            normal_init(1.0, 0.1),
            (systems.n_nuc, self.env_per_nuc),
            param_type=ParamTypes.NUCLEI,
            bias=True,
            keep_distr=self.keep_distr,
        )
        sigma = nn.softplus(sigma)
        result: list[Array] = []
        for pi, sigma, elec_nuc, (spins, charges) in zip(
            systems.group(pi, pi_meta.param_type.value.chunk_fn),
            systems.group(sigma, sigma_meta.param_type.value.chunk_fn),
            systems.group(systems.elec_nuc_dists, chunk_electron_nuclei),
            systems.unique_spins_and_charges,
        ):
            n_elec, n_nuc = sum(spins), len(charges)
            out_dim = self.out_dim * n_nuc if self.out_per_nuc else self.out_dim

            @jax.vmap  # vmap over different molecules
            def _envelopes(pi, sigma, elec_nuc):
                pi = pi.reshape(n_nuc, out_dim, self.env_per_nuc)
                sigma = sigma.reshape(n_nuc, self.env_per_nuc)
                elec_nuc = elec_nuc.reshape(n_elec, n_nuc, 4)[..., -1]
                envelopes = jnp.exp(-jnp.einsum('en,no->eno', elec_nuc, sigma))
                return jnp.einsum('eno,nvo->ev', envelopes, pi)

            result.append(_envelopes(pi, sigma, elec_nuc))
        return result


ENVELOPES = Modules[Envelope](
    {
        envelope.__name__.lower().replace('envelope', ''): envelope
        for envelope in [FullEnvelope, EfficientEnvelope]
    }
)
