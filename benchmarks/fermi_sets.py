"""Times the Fermi Sets ansatz against FiRE + Slater.

Both models build the same n_elec x n_elec determinants; they differ in where the
electron-electron interaction enters. FiRE feeds it into the orbitals, which makes
their jacobian dense, while Fermi Sets keeps the orbitals single-particle and
routes the interaction into the determinant coefficients. Run with

    uv run --with-editable /path/to/folx python benchmarks/fermi_sets.py
"""

import argparse
import time

import folx
import jax
import jax.numpy as jnp
import numpy as np

from neural_pfaffian.hamiltonian import KineticEnergyOp, make_kinetic_energy
from neural_pfaffian.nn.antisymmetrizer import FermiSets, Slater
from neural_pfaffian.nn.embedding import FiRE, FiRELocal
from neural_pfaffian.nn.envelope import EfficientEnvelope
from neural_pfaffian.nn.jastrow import CuspJastrow
from neural_pfaffian.nn.wave_function import GeneralizedWaveFunction, WaveFunction
from neural_pfaffian.systems import Systems

EMBEDDING = dict(embedding_dim=256, filter_hidden_dim=64, filter_dim=32, n_envelopes=8)
FACTOR = dict(
    embedding_dim=64,
    filter_hidden_dim=32,
    filter_dim=16,
    n_envelopes=8,
    hidden_dims=(64,),
)
DETERMINANTS = 4
ENV_PER_NUC = 8


def hydrogen_chain(n_elec: int, spacing: float = 1.8) -> Systems:
    """A linear hydrogen chain with one electron per nucleus.

    Args:
        n_elec: Number of electrons, must be even.
        spacing: Nuclear distance in bohr.

    Returns:
        The system with electrons placed at the origin.
    """
    nuclei = np.zeros((n_elec, 3), np.float32)
    nuclei[:, 0] = np.arange(n_elec) * spacing
    return Systems(
        spins=((n_elec // 2, n_elec // 2),),
        charges=((1,) * n_elec,),
        electrons=jnp.zeros((n_elec, 3), jnp.float32),
        nuclei=jnp.asarray(nuclei),
        mol_data={},
    )


def make_models():
    """Builds the wave functions to compare.

    Returns:
        Dict from name to (embedding, antisymmetrizer).
    """
    envelope = EfficientEnvelope(ENV_PER_NUC)
    return {
        'fire+slater': (
            FiRE(**EMBEDDING, activation=jax.nn.silu),
            Slater(DETERMINANTS, envelope),
        ),
        'fermisets': (
            FiRELocal(**EMBEDDING, activation=jax.nn.silu, n_layer=2),
            FermiSets(
                DETERMINANTS, envelope, **FACTOR, activation=jax.nn.silu, post_layers=0
            ),
        ),
        'fermisets+post': (
            FiRELocal(**EMBEDDING, activation=jax.nn.silu, n_layer=2),
            FermiSets(
                DETERMINANTS, envelope, **FACTOR, activation=jax.nn.silu, post_layers=1
            ),
        ),
    }


def timeit(fn, *args, warmup: int = 3, iters: int = 20):
    """Times a jitted function, reporting compile time separately.

    Args:
        fn: Function to time; its output is blocked on.
        *args: Arguments of `fn`.
        warmup: Number of untimed calls after compilation.
        iters: Number of timed calls.

    Returns:
        Tuple of compile time, median and standard deviation of the runtime, all
        in seconds. All entries are NaN if the call runs out of memory.
    """
    nan = (float('nan'),) * 3
    try:
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        compile_time = time.perf_counter() - start
        for _ in range(warmup):
            jax.block_until_ready(fn(*args))
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            jax.block_until_ready(fn(*args))
            times.append(time.perf_counter() - start)
    except Exception as e:  # OOM on the larger systems
        print(f'    failed: {type(e).__name__}')
        return nan
    return compile_time, float(np.median(times)), float(np.std(times))


def benchmark(n_elec: int, fwd_batch: int, lapl_batch: int, max_batch: int):
    """Times the forward pass and the laplacian of every model.

    Args:
        n_elec: Number of electrons of the hydrogen chain.
        fwd_batch: Number of walkers of the forward pass.
        lapl_batch: Number of walkers of the laplacian.
        max_batch: Chunk size of the laplacian's batched vmap.

    Returns:
        Dict from model name to its measurements.
    """
    systems = hydrogen_chain(n_elec)
    results = {}
    for name, (embedding, antisymmetrizer) in make_models().items():
        wf = GeneralizedWaveFunction.create(
            WaveFunction(embedding, antisymmetrizer, [CuspJastrow()]), None, systems
        )
        params = wf.init(jax.random.key(0), systems)
        n_params = sum(x.size for x in jax.tree.leaves(params))

        fwd_systems = systems.init_electrons(jax.random.key(1), fwd_batch)
        lapl_systems = systems.init_electrons(jax.random.key(1), lapl_batch)
        reparams = wf.reparams(params, systems)

        forward = jax.jit(wf.batched_apply)
        kinetic = jax.jit(
            folx.batched_vmap(
                make_kinetic_energy(wf, KineticEnergyOp.FORWARD),
                max_batch_size=max_batch,
                in_axes=(None, systems.electron_vmap, None),
            )
        )
        fwd = timeit(forward, params, fwd_systems, reparams)
        lapl = timeit(kinetic, params, lapl_systems, reparams)
        results[name] = dict(
            n_params=n_params,
            fwd_compile=fwd[0],
            fwd=fwd[1] / fwd_batch,
            fwd_std=fwd[2] / fwd_batch,
            lapl_compile=lapl[0],
            lapl=lapl[1] / lapl_batch,
            lapl_std=lapl[2] / lapl_batch,
        )
        print(
            f'  {name:16s} params={n_params / 1e6:5.2f}M '
            f'fwd={results[name]["fwd"] * 1e3:8.3f} ms/walker '
            f'lapl={results[name]["lapl"] * 1e3:8.3f} ms/walker '
            f'(compile {fwd[0]:.1f}s / {lapl[0]:.1f}s)'
        )
        del wf, params, forward, kinetic
        jax.clear_caches()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--electrons', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--fwd-batch', type=int, default=64)
    parser.add_argument('--lapl-batch', type=int, default=8)
    parser.add_argument('--max-batch', type=int, default=4)
    args = parser.parse_args()

    print(f'backend={jax.default_backend()} device={jax.local_devices()[0].device_kind}')
    all_results = {}
    for n_elec in args.electrons:
        print(f'n_elec={n_elec}')
        all_results[n_elec] = benchmark(
            n_elec, args.fwd_batch, args.lapl_batch, args.max_batch
        )

    baseline, *rest = list(make_models())
    for what in ['fwd', 'lapl']:
        print(f'\n{what} (ms / walker)')
        header = [baseline] + [f'{n} (speedup)' for n in rest]
        print('| n_elec | ' + ' | '.join(header) + ' |')
        print('|---' * (len(header) + 1) + '|')
        for n_elec, res in all_results.items():
            cells = [f'{res[baseline][what] * 1e3:.3f}']
            cells += [
                f'{res[n][what] * 1e3:.3f} ({res[baseline][what] / res[n][what]:.2f}x)'
                for n in rest
            ]
            print(f'| {n_elec} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
