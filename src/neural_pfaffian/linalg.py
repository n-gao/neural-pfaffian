import functools

import jax
import jax.numpy as jnp
from jax._src.ad_util import SymbolicZero
from jax.scipy.linalg import block_diag

from neural_pfaffian.utils.jax_utils import jit

try:
    import folx
except ImportError:
    folx = None


@jit
def householder(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    x0 = x[0]
    x_norm_squared = jnp.dot(x, x)
    x_norm = jnp.sqrt(x_norm_squared)
    sign = jnp.sign(x0)
    alpha = -sign * x_norm
    v = x - jnp.array([alpha] + [0] * (x.shape[0] - 1), dtype=x.dtype)
    # a faster way to compute the norm of v where v_0 = x_0 + alpha and v_i = x_i for i > 0
    v_rnorm = jax.lax.rsqrt(x_norm_squared - 2 * x0 * alpha + alpha * alpha)
    v *= v_rnorm
    return v, sign, alpha


@functools.partial(jax.custom_jvp)
@functools.partial(jnp.vectorize, signature='(n,n)->(),()', excluded=frozenset({1}))
def slog_pfaffian(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Computes the Pfaffian of a skew-symmetric matrix A using the householder transformation.
    """
    A = jnp.asarray(A)
    out_dtype = A.dtype
    A = A.astype(jnp.float64)
    dtype = A.dtype
    n = A.shape[0]
    if n % 2 == 1:
        return jnp.ones((), dtype=out_dtype), jnp.array(-jnp.inf, dtype=out_dtype)

    sign_pfaffian = jnp.ones((), dtype=dtype)
    log_pfaffian = jnp.zeros((), dtype=dtype)

    for i in range(n - 2):
        v, sign, alpha = householder(A[1:, 0])
        vw = 2 * jnp.einsum('a,bc,c->ab', v, A[1:, 1:], v)
        delta = vw - vw.mT
        A = A[1:, 1:] + delta

        if i % 2 == 0:
            sign_pfaffian *= sign
            log_pfaffian += jnp.log(jnp.abs(alpha))

    sign_pfaffian *= jnp.sign(A[-2, -1])
    log_pfaffian += jnp.log(jnp.abs(A[-2, -1]))
    return sign_pfaffian.astype(out_dtype), log_pfaffian.astype(out_dtype)


@slog_pfaffian.defjvp
def slog_pfaffian_jvp(primals, tangents):
    jnp.linalg.slogdet
    (A,) = primals
    (A_dot,) = tangents
    sign_pfaffian, log_pfaffian = slog_pfaffian(A)
    det_dot = jnp.einsum('...ij,...ji->...', jnp.linalg.inv(A), A_dot)
    sign_dot = jnp.zeros_like(sign_pfaffian)
    pfaffian_dot = det_dot / 2
    return (sign_pfaffian, log_pfaffian), (sign_dot, pfaffian_dot)


slog_pfaffian = jit(slog_pfaffian)


@jax.custom_jvp
def skewsymmetric_quadratic(x: jax.Array, A: jax.Array) -> jax.Array:
    result = x @ A @ x.mT
    # explicitly antisymmetrize the result for lower precisions
    if result.dtype != jnp.float64:
        return (result - result.mT) / 2
    return result


@functools.partial(skewsymmetric_quadratic.defjvp, symbolic_zeros=True)
def skewsymmetric_quadratic_jvp(primals, tangents):
    x, A = primals
    x_dot, A_dot = tangents
    y = skewsymmetric_quadratic(x, A)
    y_dot = jnp.zeros_like(y)
    if not isinstance(A_dot, SymbolicZero):
        y_dot += skewsymmetric_quadratic(x, A_dot)
    if not isinstance(x_dot, SymbolicZero):
        xAx_dot = jnp.einsum('...ab,...bc,...dc->...ad', x, A, x_dot)
        y_dot += xAx_dot - xAx_dot.mT
    return skewsymmetric_quadratic(x, A), y_dot


skewsymmetric_quadratic = jit(skewsymmetric_quadratic)


@jax.custom_jvp
def slogdet_skewsymmetric_quadratic(x: jax.Array, A: jax.Array):
    return jnp.linalg.slogdet(skewsymmetric_quadratic(x, A))


@functools.partial(slogdet_skewsymmetric_quadratic.defjvp, symbolic_zeros=True)
def slogdet_skewsymmetric_quadratic_jvp(primals, tangents):
    x, A = primals
    x_dot, A_dot = tangents
    sign, log_det = slogdet_skewsymmetric_quadratic(x, A)
    inv_xAx = inv_skewsymmetric_quadratic(x, A)
    log_det_dot = jnp.zeros_like(log_det)
    if not isinstance(x_dot, SymbolicZero):
        log_det_dot += 2 * jnp.einsum(
            '...ab,...cb,...cd,...da->...', A, x, inv_xAx, x_dot
        )
    if not isinstance(A_dot, SymbolicZero):
        log_det_dot -= (
            jnp.einsum('...ab,...ab->...', skewsymmetric_quadratic(x.mT, inv_xAx), A_dot)
            / 2
        )
    return (sign, log_det), (jnp.zeros_like(sign), log_det_dot)


slogdet_skewsymmetric_quadratic = jit(slogdet_skewsymmetric_quadratic)


@jax.jit
def det_skewsymmetric_quadratic(x: jax.Array, A: jax.Array) -> jax.Array:
    sign, logdet = slogdet_skewsymmetric_quadratic(x, A)
    return sign * jnp.exp(logdet)


@jax.custom_jvp
def slog_pfaffian_skewsymmetric_quadratic(
    x: jax.Array,
    A: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return slog_pfaffian(skewsymmetric_quadratic(x, A))


@functools.partial(slog_pfaffian_skewsymmetric_quadratic.defjvp, symbolic_zeros=True)
def slog_pfaffian_skewsymmetric_quadratic_jvp(primals, tangents):
    x, A = primals
    x_dot, A_dot = tangents
    sign, log_pf = slog_pfaffian_skewsymmetric_quadratic(x, A)
    log_pf_dot = jnp.zeros_like(log_pf)
    inv_xAx = inv_skewsymmetric_quadratic(x, A)
    if not isinstance(x_dot, SymbolicZero):
        log_pf_dot -= jnp.einsum('...ab,...cb,...dc,...da->...', A, x, inv_xAx, x_dot)
    if not isinstance(A_dot, SymbolicZero):
        log_pf_dot -= (
            jnp.einsum('...ab,...ab->...', skewsymmetric_quadratic(x.mT, inv_xAx), A_dot)
            / 2
        )
    return (sign, log_pf), (jnp.zeros_like(sign), log_pf_dot)


slog_pfaffian_skewsymmetric_quadratic = jit(slog_pfaffian_skewsymmetric_quadratic)


@jax.custom_jvp
def inv_skewsymmetric_quadratic(x: jax.Array, A: jax.Array) -> jax.Array:
    xAx = skewsymmetric_quadratic(x, A)
    result = jnp.linalg.inv(xAx.astype(jnp.float64)).astype(xAx.dtype)
    if result.dtype != jnp.float64:
        return (result - result.mT) / 2
    return result


@functools.partial(inv_skewsymmetric_quadratic.defjvp, symbolic_zeros=True)
def inv_skewsymmetric_quadratic_jvp(primals, tangents):
    x, A = primals
    x_dot, A_dot = tangents
    inv_xAx = inv_skewsymmetric_quadratic(x, A)
    inner = jnp.zeros_like(inv_xAx)
    if not isinstance(A_dot, SymbolicZero):
        inner += skewsymmetric_quadratic(x, A_dot)
    if not isinstance(x_dot, SymbolicZero):
        xAx_dot = jnp.einsum('...ab,...bc,...dc->...ad', x, A, x_dot)
        inner += xAx_dot - xAx_dot.mT
    return inv_xAx, skewsymmetric_quadratic(inv_xAx, inner)


inv_skewsymmetric_quadratic = jit(inv_skewsymmetric_quadratic)


# TODO: add tests for this
# Here we define the functions for folx such that we can use the forward-laplacian
if folx is not None:
    from folx.api import FunctionFlags
    from folx.custom_hessian import slogdet_jac_hessian_jac

    def skewsymmetric_quadratic_jac_hessian_jac(
        args,
        extra_args,
        merge,
        materialize_idx,
    ):
        (X,), A = merge(args, extra_args)
        assert isinstance(A, jax.Array), (
            'Laplacian for A being a function of X is not supported'
        )
        jac = X.jacobian.dense_array
        result = jnp.einsum('i...ab,...bc,i...dc->...ad', jac, A, jac)
        return result - result.mT

    def folx_slog_pfaffian_jac_hessian_jac(
        args,
        extra_args,
        merge,
        materialize_idx,
    ):
        signs, logdet = slogdet_jac_hessian_jac(
            args,
            extra_args,
            merge,
            materialize_idx,
        )
        return signs, logdet / 2

    def folx_slog_pfaffian(args, kwargs, sparsity_threshold: int):
        fwd_lapl_fn = folx.wrap_forward_laplacian(
            slog_pfaffian, custom_jac_hessian_jac=folx_slog_pfaffian_jac_hessian_jac
        )
        sign, logpf = fwd_lapl_fn(args, kwargs, sparsity_threshold=sparsity_threshold)
        sign = folx.warp_without_fwd_laplacian(lambda x: x)(
            (sign,), {}, sparsity_threshold=sparsity_threshold
        )
        return sign, logpf

    def folx_slog_pfaffian_skewsymmetric_quadratic(args, kwargs, sparsity_threshold: int):
        return folx.forward_laplacian(
            slog_pfaffian_skewsymmetric_quadratic.fun,
            sparsity_threshold=sparsity_threshold,
        )(*args)

    folx.register_function(
        'skewsymmetric_quadratic',
        folx.wrap_forward_laplacian(
            skewsymmetric_quadratic,
            name='skewsymmetric_quadratic',
            flags=FunctionFlags.JOIN_JVP,
            custom_jac_hessian_jac=skewsymmetric_quadratic_jac_hessian_jac,
        ),
    )
    folx.register_function('slog_pfaffian', folx_slog_pfaffian)
    folx.register_function(
        'slog_pfaffian_skewsymmetric_quadratic',
        folx_slog_pfaffian_skewsymmetric_quadratic,
    )


def cayley_transform(x: jax.Array) -> jax.Array:
    x = (x - x.mT) / 2
    I = jnp.eye(x.shape[-1], dtype=x.dtype)
    Q = jnp.linalg.solve(x + I, x - I)
    return Q @ Q


def to_skewsymmetric_orthogonal(x: jax.Array):
    # The skew-symmetric identity matrix
    J = block_diag(*[jnp.array([[0, 1], [-1, 0]], dtype=x.dtype)] * (x.shape[-1] // 2))
    return skewsymmetric_quadratic(cayley_transform(x), J)
