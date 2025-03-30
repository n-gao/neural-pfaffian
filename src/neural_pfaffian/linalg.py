import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.ad_util import SymbolicZero
from jax.scipy.linalg import block_diag
from jaxtyping import Array, Float

from neural_pfaffian.utils.jax_utils import jit, vectorize
from neural_pfaffian.utils.tree_utils import tree_to_dtype


@vectorize(signature='(n,n)->(),()')
def _slogpfaffian_2x2(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    assert A.shape == (2, 2)
    A = (A - A.mT) / 2  # make sure that gradients are correct
    a = A[0, 1]
    sign = jnp.sign(a)
    log_pfaffian = jnp.log(jnp.abs(a))
    return sign, log_pfaffian


@vectorize(signature='(n,n)->(),()')
def _slogpfaffian_4x4(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    assert A.shape == (4, 4)
    A = (A - A.mT) / 2  # make sure that gradients are correct
    a, b, c, d, e, f = A[np.triu_indices(4, 1)]
    pf = a * f - b * e + d * c
    sign = jnp.sign(pf)
    log_pfaffian = jnp.log(jnp.abs(pf))
    return sign, log_pfaffian


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
def _slog_pfaffian_general(A: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    elif n == 2:
        return tree_to_dtype(_slogpfaffian_2x2(A), out_dtype)

    sign_pfaffian = jnp.ones((), dtype=dtype)
    log_pfaffian = jnp.zeros((), dtype=dtype)

    # We use the householder transformation to reduce the matrix to 4x4
    # For 4x4, we can use the closed form solution
    for i in range(n - 4):
        v, sign, alpha = householder(A[1:, 0])
        vw = 2 * jnp.einsum('a,bc,c->ab', v, A[1:, 1:], v)
        delta = vw - vw.mT
        A = A[1:, 1:] + delta

        if i % 2 == 0:
            sign_pfaffian *= sign
            log_pfaffian += jnp.log(jnp.abs(alpha))

    # Use closed form solution for 4x4
    s_remaing, log_remaing = _slogpfaffian_4x4(A)
    sign_pfaffian *= s_remaing
    log_pfaffian += log_remaing
    return tree_to_dtype((sign_pfaffian, log_pfaffian), out_dtype)


@_slog_pfaffian_general.defjvp
def slog_pfaffian_jvp(primals, tangents):
    jnp.linalg.slogdet
    (A,) = primals
    (A_dot,) = tangents
    sign_pfaffian, log_pfaffian = _slog_pfaffian_general(A)
    A_inv = skewsymmetric_inv(A)
    det_dot = jnp.einsum('...ij,...ji->...', A_inv, A_dot)
    sign_dot = jnp.zeros_like(sign_pfaffian)
    pfaffian_dot = det_dot / 2
    return (sign_pfaffian, log_pfaffian), (sign_dot, pfaffian_dot)


_slog_pfaffian_general = jit(_slog_pfaffian_general)


@jit
@vectorize(signature='(n,n)->(),()')
def slog_pfaffian(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    match A.shape[-1]:
        case 2:
            sign, log_pfaffian = _slogpfaffian_2x2(A)
        case 4:
            sign, log_pfaffian = _slogpfaffian_4x4(A)
        case _:
            sign, log_pfaffian = _slog_pfaffian_general(A)
    return sign, log_pfaffian


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
    result = skewsymmetric_inv(xAx.astype(jnp.float64)).astype(xAx.dtype)
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


@vectorize(signature='(n,n)->(n,n)')
def _skewsymmetric_inv_2x2(A: jax.Array) -> jax.Array:
    a = A[0, 1]
    return jnp.array([[0, -1 / a], [1 / a, 0]])


@jax.custom_jvp
def skewsymmetric_inv(A: jax.Array) -> jax.Array:
    if A.shape[-1] == 2:
        return _skewsymmetric_inv_2x2(A)
    elif A.shape[-1] % 2 == 1:
        # These matrices are singular and cannot be inverted
        return jnp.full_like(A, jnp.nan)
    return jnp.linalg.inv(A)


@skewsymmetric_inv.defjvp
def skewsymmetric_inv_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents
    A_inv = skewsymmetric_inv(A)
    return A_inv, skewsymmetric_quadratic(A_inv, A_dot)


skewsymmetric_inv = jit(skewsymmetric_inv)


def antisymmetric_block_diagonal(n: int, dtype: jnp.dtype = jnp.float32):
    return block_diag(*[jnp.array([[0, 1], [-1, 0]], dtype=dtype)] * n)


@vectorize(signature='(n,n),(n,r)->(),()')
def slog_pfaffian_with_updates(
    X: Float[Array, 'n_el n_el'], B: Float[Array, 'n_el rank']
):
    # https://arxiv.org/pdf/2105.13098
    sign_X, logdet_X = slog_pfaffian(X)
    assert B.shape[-1] % 2 == 0
    C = antisymmetric_block_diagonal(B.shape[-1] // 2, dtype=X.dtype)
    Y = C.mT + skewsymmetric_quadratic(B.T, skewsymmetric_inv(X))
    sign_Y, logdet_Y = slog_pfaffian(Y)
    return -sign_X * sign_Y, logdet_X + logdet_Y


# TODO: add tests for this
# Here we define the functions for folx such that we can use the forward-laplacian
try:
    import folx
    from folx.api import FunctionFlags, FwdLaplArray, FwdLaplArgs
    from folx.custom_hessian import slogdet_jac_hessian_jac

    def skewsymmetric_quadratic_jac_hessian_jac(
        args: FwdLaplArgs,
        extra_args,
        merge,
        materialize_idx,
    ):
        if len(args.arrays) == 1:
            (X,), A = merge(args, extra_args)
        elif len(args) == 2:
            X, A = args.arrays
        else:
            raise ValueError('Invalid number of arguments')
        if not isinstance(X, FwdLaplArray):
            return 0
        result = jnp.zeros((1, 1), dtype=X.dtype)
        if isinstance(X, FwdLaplArray):
            A_ = A.x if isinstance(A, FwdLaplArray) else A
            X_jac = X.jacobian.dense_array
            result += jnp.einsum('i...ab,...bc,i...dc->...ad', X_jac, A_, X_jac)
            if isinstance(A, FwdLaplArray):
                A_jac = A.jacobian.dense_array
                result += 2 * jnp.einsum('i...ab,i...bc,...dc->...ad', X_jac, A_jac, X.x)
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

    def folx_slog_pfaffian_general(args, kwargs, sparsity_threshold: int):
        fwd_lapl_fn = folx.wrap_forward_laplacian(
            _slog_pfaffian_general,
            custom_jac_hessian_jac=folx_slog_pfaffian_jac_hessian_jac,
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
    folx.register_function('_slog_pfaffian_general', folx_slog_pfaffian_general)
    folx.register_function(
        'slog_pfaffian_skewsymmetric_quadratic',
        folx_slog_pfaffian_skewsymmetric_quadratic,
    )
    folx.register_function(
        'skewsymmetric_inv',
        folx.wrap_forward_laplacian(
            skewsymmetric_inv,
            name='inv',
            in_axes=(-2, -1),
        ),
    )
except ImportError:
    pass


def cayley_transform(x: jax.Array) -> jax.Array:
    x = (x - x.mT) / 2
    I = jnp.eye(x.shape[-1], dtype=x.dtype)
    Q = jnp.linalg.solve(x + I, x - I)
    return Q @ Q


def to_skewsymmetric_orthogonal(x: jax.Array):
    # The skew-symmetric identity matrix
    J = antisymmetric_block_diagonal(x.shape[-1] // 2, dtype=x.dtype)
    return skewsymmetric_quadratic(cayley_transform(x), J)
