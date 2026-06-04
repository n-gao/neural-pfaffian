import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from neural_pfaffian.utils.jax_utils import jit


@jit
def softplus(x: ArrayLike, alpha: float = 0.0, beta: float = 1.0) -> Array:
    r"""Softplus activation function.

    Computes the element-wise function

    :math:`
      \mathrm{softplus}_{\alpha, \beta}(x) = \frac{1}{\beta}\log(\alpha + 1 + e^{\beta x})
    `

    Args:
      x : input array
    """
    return jnp.logaddexp(beta * x, jnp.log1p(alpha)) / beta
