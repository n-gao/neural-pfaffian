from neural_pfaffian.utils import Modules

from ..wave_function import EmbeddingP
from .ferminet import FermiNet
from .moon import Moon
from .psiformer import PsiFormer

EMBEDDINGS = Modules[EmbeddingP](
    {embedding.__name__.lower(): embedding for embedding in [FermiNet, Moon, PsiFormer]}
)
