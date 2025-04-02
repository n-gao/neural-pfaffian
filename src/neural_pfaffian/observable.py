from typing import Any, Protocol
import jax.numpy as jnp
from flax.struct import PyTreeNode, field
from jaxtyping import Array, Float

from neural_pfaffian.nn.wave_function import GeneralizedWaveFunction
from neural_pfaffian.systems import Systems
from neural_pfaffian.utils import Modules, MovingAverage
from neural_pfaffian.utils.jax_utils import pmean_if_pmap, vmap

LocalEnergy = Float[Array, 'batch_size n_mols']
EnergyData = Float[Array, 'n_mols 2']


class Observable(Protocol):
    def init_systems[S: Systems](self, key: Array, systems: S) -> S: ...
    def log(self, systems: Systems, local_energy: LocalEnergy) -> Systems: ...
    def data(self, systems: Systems) -> dict[str, Any]: ...


class EnergyObservable(Observable, PyTreeNode):
    wave_function: GeneralizedWaveFunction = field(pytree_node=False)
    window_size: int = field(pytree_node=False)
    key: str = field(default='energy', pytree_node=False)

    @classmethod
    def _key(cls) -> str: ...

    def init_systems[S: Systems](self, key: Array, systems: S) -> S:
        if self.key not in systems.mol_data:
            avg = vmap(MovingAverage[EnergyData].init, in_axes=(0, None))(
                jnp.zeros((systems.n_mols, 2), dtype=jnp.float64), self.window_size
            )
            return systems.set_mol_data(self.key, avg)
        return systems

    def log(self, systems: Systems, local_energy: LocalEnergy) -> Systems:
        @vmap
        def update_state(state: MovingAverage[EnergyData], data: EnergyData):
            return state.update(data)

        energy = pmean_if_pmap(local_energy.mean(0))
        var = ((local_energy - energy) ** 2).mean(0)
        data = jnp.stack([energy, var], axis=-1)
        new_state = update_state(systems.get_mol_data(self.key), data)
        return systems.set_mol_data(self.key, new_state)

    def data(self, systems: Systems) -> dict[str, Any]:
        @vmap
        def value(state: MovingAverage[EnergyData]):
            return state.value()

        E, E_var = value(systems.get_mol_data(self.key)).T
        return dict(energy=dict(enumerate(E)), energy_var=dict(enumerate(E_var)))


OBSERVABLES = Modules[Observable](
    {cls.__name__.lower().replace('observable', ''): cls for cls in [EnergyObservable]}
)
