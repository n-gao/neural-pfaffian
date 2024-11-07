from neural_pfaffian.nn.ferminet import FermiNet
from neural_pfaffian.nn.moon import Moon
from neural_pfaffian.nn.psiformer import PsiFormer
from neural_pfaffian.nn.wave_function import EmbeddingP
from neural_pfaffian.utils import Modules

EMBEDDINGS = Modules[EmbeddingP](
    {embedding.__name__.lower(): embedding for embedding in [FermiNet, Moon, PsiFormer]}
)
