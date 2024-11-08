from neural_pfaffian.utils import Modules

from ..wave_function import AntisymmetrizerP
from .pfaffian import Pfaffian
from .slater import Slater

ANTISYMMETRIZERS = Modules[AntisymmetrizerP](
    {cls.__name__.lower(): cls for cls in [Pfaffian, Slater]}
)
