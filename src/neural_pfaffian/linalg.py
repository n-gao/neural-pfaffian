import functools

import jax
import jax.numpy as jnp
import numpy as np
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


@functools.partial(jnp.vectorize, signature='(n,n)->(),()', excluded=frozenset({1}))
def _slog_pfaffian_householder(A: jax.Array) -> tuple[jax.Array, jax.Array]:
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


@jax.custom_jvp
def slog_pfaffian(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sign and log|pf| of skew-symmetric matrices.

    Uses a pallas kernel on float32 GPU inputs and a householder
    tridiagonalization otherwise.

    Args:
        A: Skew-symmetric matrices of shape (..., n, n).

    Returns:
        Tuple of sign and log|pf|, each of shape (...,).
    """
    from neural_pfaffian.kernels import slog_pfaffian_pallas, supports_slog_pfaffian

    A = jnp.asarray(A)
    if A.shape[-1] % 2 == 0 and supports_slog_pfaffian(A):
        return slog_pfaffian_pallas(A)
    return _slog_pfaffian_householder(A)


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


def _bordered_skewsymmetric(y: jax.Array, c: jax.Array) -> jax.Array:
    """Skew-symmetric border extension [[y, c], [-c^T, 0]].

    Args:
        y: Skew-symmetric matrices of shape (..., n, n).
        c: Border vectors of shape (..., n).

    Returns:
        Bordered matrices of shape (..., n + 1, n + 1).
    """
    n = y.shape[-1]
    batch = jnp.broadcast_shapes(y.shape[:-2], c.shape[:-1])
    Y = jnp.zeros((*batch, n + 1, n + 1), y.dtype)
    Y = Y.at[..., :n, :n].set(y)
    Y = Y.at[..., :n, n].set(c)
    return Y.at[..., n, :n].set(-c)


@jax.custom_jvp
def slog_pfaffian_bordered_quadratic(
    x: jax.Array,
    c: jax.Array,
    A: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sign and log|pf| of [[x A x^T, c], [-c^T, 0]] for odd electron counts.

    Args:
        x: Orbital matrices of shape (..., n, m), n odd.
        c: Border vectors of shape (..., n).
        A: Skew-symmetric matrices of shape (..., m, m).

    Returns:
        Tuple of sign and log|pf|, each of shape (...,).
    """
    return slog_pfaffian(_bordered_skewsymmetric(skewsymmetric_quadratic(x, A), c))


@functools.partial(slog_pfaffian_bordered_quadratic.defjvp, symbolic_zeros=True)
def slog_pfaffian_bordered_quadratic_jvp(primals, tangents):
    x, c, A = primals
    x_dot, c_dot, A_dot = tangents
    n = x.shape[-2]
    sign, log_pf = slog_pfaffian_bordered_quadratic(x, c, A)
    Z = _skew_inv(_bordered_skewsymmetric(skewsymmetric_quadratic(x, A), c))
    Z11, z = Z[..., :n, :n], Z[..., :n, n]
    log_pf_dot = jnp.zeros_like(log_pf)
    if not isinstance(x_dot, SymbolicZero):
        log_pf_dot -= jnp.einsum('...ab,...cb,...dc,...da->...', A, x, Z11, x_dot)
    if not isinstance(c_dot, SymbolicZero):
        log_pf_dot -= jnp.einsum('...n,...n->...', z, c_dot)
    if not isinstance(A_dot, SymbolicZero):
        log_pf_dot -= (
            jnp.einsum('...ab,...ab->...', skewsymmetric_quadratic(x.mT, Z11), A_dot) / 2
        )
    return (sign, log_pf), (jnp.zeros_like(sign), log_pf_dot)


slog_pfaffian_bordered_quadratic = jit(slog_pfaffian_bordered_quadratic)


def _skew_inv(y: jax.Array) -> jax.Array:
    """Inverse of skew-symmetric matrices, antisymmetrized for low precisions.

    Args:
        y: Skew-symmetric matrices of shape (..., n, n).

    Returns:
        The inverses, of shape (..., n, n).
    """
    result = jnp.linalg.inv(y.astype(jnp.float64)).astype(y.dtype)
    if result.dtype != jnp.float64:
        return (result - result.mT) / 2
    return result


@jax.custom_jvp
def inv_skewsymmetric_quadratic(x: jax.Array, A: jax.Array) -> jax.Array:
    return _skew_inv(skewsymmetric_quadratic(x, A))


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


# Here we define the functions for folx such that we can use the forward-laplacian
if folx is not None:
    from folx.api import FunctionFlags, FwdJacobian, FwdLaplArray

    from neural_pfaffian.kernels import (
        pfaffian_fwd_lapl,
        pfaffian_fwd_lapl_bordered,
        pfaffian_fwd_lapl_config,
    )

    # folx >=0.2.26 always joins the jacobian and laplacian jvp
    JOIN_JVP = getattr(FunctionFlags, 'JOIN_JVP', FunctionFlags.GENERAL)

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
        # log|pf A| = log|det A| / 2, so the JHJ trace of slogdet is halved
        A = args.x[0]
        A_inv = jnp.linalg.inv(A)
        M = jnp.einsum(
            '...ij,k...jd->k...id',
            A_inv,
            args.jacobian[0].construct_jac_for(materialize_idx),
        )
        trace = -jnp.einsum('k...id,k...di->...', M, M) / 2
        return jnp.zeros(A.shape[:-2], dtype=trace.dtype), trace

    def folx_slog_pfaffian(args, kwargs, sparsity_threshold: int):
        fwd_lapl_fn = folx.wrap_forward_laplacian(
            slog_pfaffian, custom_jac_hessian_jac=folx_slog_pfaffian_jac_hessian_jac
        )
        sign, logpf = fwd_lapl_fn(args, kwargs, sparsity_threshold=sparsity_threshold)
        sign = folx.warp_without_fwd_laplacian(lambda x: x)(
            (sign,), {}, sparsity_threshold=sparsity_threshold
        )
        return sign, logpf

    def _tangent_major_jacobian(jac: FwdJacobian) -> tuple[jax.Array, np.ndarray | None]:
        """Extracts the jacobian of x as a (tangents, ..., n, m) array.

        Sparse jacobians whose index mask is constant over the trailing matrix
        dimensions map every tangent slot to a single input coordinate per batch
        position, so their data can be contracted directly. Any other sparsity
        pattern is densified.

        Args:
            jac: Forward-laplacian jacobian of the orbital matrix x.

        Returns:
            Tuple of the jacobian array (tangent dimension leading) and the
            output index mask (None for dense jacobians).
        """
        if jac.x0_idx is None:
            return jac.data, None
        idx = np.broadcast_to(jac.x0_idx, jac.data.shape)
        if not (idx == idx[..., :1, :1]).all():
            return jac.dense_array, None
        out_idx = idx[..., 0, 0]
        # duplicate coordinates per position would be double counted
        sorted_idx = np.sort(out_idx, axis=0)
        if ((sorted_idx[:-1] == sorted_idx[1:]) & (sorted_idx[:-1] >= 0)).any():
            return jac.dense_array, None
        data = jac.data
        if (out_idx < 0).any():  # zero padded slots
            data = jnp.where(jnp.asarray(out_idx[..., None, None] >= 0), data, 0)
        return data, out_idx

    def folx_slog_pfaffian_skewsymmetric_quadratic(args, kwargs, sparsity_threshold: int):
        """Fused forward-laplacian of slog_pfaffian(skewsymmetric_quadratic(x, A)).

        Computes the jacobian and laplacian of log|pf(x A x^T)| directly from the
        jacobian of x without materializing the jacobian of the intermediate
        (elec, elec) matrix. On float32 GPU inputs a pallas kernel computes the
        JHJ trace in a single pass over the jacobian; otherwise layout-aware
        einsums are used.

        Args:
            args: Tuple of the orbital matrix x (FwdLaplArray) and the constant
                skew-symmetric matrix A.
            kwargs: Unused keyword arguments.
            sparsity_threshold: folx sparsity threshold (used by the fallback).

        Returns:
            Tuple of the constant sign array and the FwdLaplArray of log|pf|.
        """
        x, A = args
        if not isinstance(x, FwdLaplArray) or isinstance(A, FwdLaplArray):
            # A depending on the electrons is not fused
            return folx.forward_laplacian(
                slog_pfaffian_skewsymmetric_quadratic.fun,
                sparsity_threshold=sparsity_threshold,
            )(*args)

        xv = x.x
        J, out_idx = _tangent_major_jacobian(x.jacobian)
        *batch, n, m = xv.shape
        config = pfaffian_fwd_lapl_config(J, bordered=False)
        if config is not None and J.shape[1:-2] == tuple(batch):
            large, num_stages = config
            d = int(np.prod(batch, dtype=int))
            A_b = jnp.broadcast_to(A, (*batch, m, m)).reshape(d, m, m)
            J_b = J.reshape(J.shape[0], d, n, m)
            P = W = None
            if large:
                P = (xv @ A).reshape(d, n, m)
                W = jnp.einsum('k...nm,...ml->k...nl', J, A).reshape(J_b.shape)
            sign, log_pf, laplacian, jac = pfaffian_fwd_lapl(
                xv.reshape(d, n, m),
                A_b,
                J_b,
                x.laplacian.reshape(d, n, m),
                P=P,
                W=W,
                large=large,
                num_stages=num_stages,
            )
            jacobian = jac.T.reshape(J.shape[0], *batch)
            log_pf = FwdLaplArray(
                log_pf.reshape(tuple(batch)),
                FwdJacobian(jacobian, out_idx),
                laplacian.reshape(tuple(batch)),
            )
            return sign.reshape(tuple(batch)), log_pf

        if xv.dtype == jnp.float32 and jax.default_backend() == 'gpu':
            # beyond the kernel tile bounds the decomposed rules are faster
            return folx.forward_laplacian(
                slog_pfaffian_skewsymmetric_quadratic.fun,
                sparsity_threshold=sparsity_threshold,
            )(*args)

        sign, log_pf = slog_pfaffian(skewsymmetric_quadratic(xv, A))
        Z = inv_skewsymmetric_quadratic(xv, A)
        P = xv @ A
        G = Z @ P  # gradient of log|pf(x A x^T)| w.r.t. x
        jhj, jacobian = _fused_pfaffian_jhj(J, Z, P, A, G)
        laplacian = jhj + jnp.einsum('...nm,...nm->...', G, x.laplacian)
        return sign, FwdLaplArray(log_pf, FwdJacobian(jacobian, out_idx), laplacian)

    def _fused_pfaffian_jhj(J, Z, P, A, G):
        """JHJ trace and output jacobian of log|pf| from the jacobian of x.

        Computes tr(J^T H J) over tangents k: for v = J_k and C = v A x^T the
        quadratic form is tr(Z v A v^T) - tr(Z C Z C) + tr(Z C Z C^T), plus the
        output jacobian <Z P, v>, with k folded into the contractions. C and F
        hold -C and -(C Z); the signs cancel in the quadratic terms.

        Args:
            J: Tangent-major jacobian of x, shape (tangents, ..., n, m).
            Z: Inverse of the skew-symmetric matrix, shape (..., n, n).
            P: Product x A, shape (..., n, m).
            A: Skew-symmetric matrix broadcastable to (..., m, m).
            G: Gradient of log|pf| w.r.t. x, shape (..., n, m).

        Returns:
            Tuple of the JHJ trace (...,) and the jacobian (tangents, ...).
        """
        jacobian = jnp.einsum('k...nm,...nm->k...', J, G)
        W = jnp.einsum('k...nm,...ml->k...nl', J, A)  # v A
        S1 = jnp.einsum('k...nm,k...lm->...nl', W, J)  # sum_k (v A) v^T
        C = jnp.einsum('k...nm,...lm->k...nl', J, P)  # v (x A)^T
        F = jnp.einsum('k...nl,...lo->k...no', C, Z)
        S3 = jnp.einsum('k...ij,k...lj->...il', F, C)  # sum_k (C Z) C^T
        jhj = -jnp.einsum('...ij,...ij->...', Z, S1 + S3)
        jhj -= jnp.einsum('k...ij,k...ji->...', F, F)
        return jhj, jacobian

    def folx_slog_pfaffian_bordered_quadratic(args, kwargs, sparsity_threshold: int):
        """Fused forward-laplacian of the bordered pfaffian for odd systems.

        The heavy part is the even-case JHJ with the (n, n) block of the
        bordered inverse; the border adds per-tangent corrections
        2 u^T Z11 (C - C^T) z - (z u)^2 for the tangents u of c.

        Args:
            args: Tuple of the orbital matrix x, the border vector c (both
                FwdLaplArray) and the constant skew-symmetric matrix A.
            kwargs: Unused keyword arguments.
            sparsity_threshold: folx sparsity threshold (used by the fallback).

        Returns:
            Tuple of the constant sign array and the FwdLaplArray of log|pf|.
        """
        x, c, A = args
        if (
            not isinstance(x, FwdLaplArray)
            or not isinstance(c, FwdLaplArray)
            or isinstance(A, FwdLaplArray)
        ):
            return folx.forward_laplacian(
                slog_pfaffian_bordered_quadratic.fun,
                sparsity_threshold=sparsity_threshold,
            )(*args)

        xv, cv = x.x, c.x
        *batch, n, m = xv.shape

        # common tangent basis for the jacobians of x and c
        J = x.jacobian.dense_array
        U = c.jacobian.dense_array
        n_tangent = max(J.shape[0], U.shape[0])
        J, U = _pad_tangents(J, n_tangent), _pad_tangents(U, n_tangent)

        config = pfaffian_fwd_lapl_config(J, bordered=True)
        if (
            config is not None
            and J.shape[1:-2] == tuple(batch)
            and U.shape[1:-1] == tuple(batch)
        ):
            large, num_stages = config
            d = int(np.prod(batch, dtype=int))
            J_b = J.reshape(n_tangent, d, n, m)
            P = W = None
            if large:
                P = (xv @ A).reshape(d, n, m)
                W = jnp.einsum('k...nm,...ml->k...nl', J, A).reshape(J_b.shape)
            sign, log_pf, laplacian, jac = pfaffian_fwd_lapl_bordered(
                xv.reshape(d, n, m),
                jnp.broadcast_to(cv, (*batch, n)).reshape(d, n),
                jnp.broadcast_to(A, (*batch, m, m)).reshape(d, m, m),
                J_b,
                U.reshape(n_tangent, d, n),
                x.laplacian.reshape(d, n, m),
                c.laplacian.reshape(d, n),
                P=P,
                W=W,
                large=large,
                num_stages=num_stages,
            )
            log_pf = FwdLaplArray(
                log_pf.reshape(tuple(batch)),
                FwdJacobian.from_dense(jac.T.reshape(n_tangent, *batch)),
                laplacian.reshape(tuple(batch)),
            )
            return sign.reshape(tuple(batch)), log_pf

        if xv.dtype == jnp.float32 and jax.default_backend() == 'gpu':
            # beyond the kernel tile bounds the decomposed rules are faster
            return folx.forward_laplacian(
                slog_pfaffian_bordered_quadratic.fun,
                sparsity_threshold=sparsity_threshold,
            )(*args)

        sign, log_pf = slog_pfaffian_bordered_quadratic(xv, cv, A)
        Z = _skew_inv(_bordered_skewsymmetric(skewsymmetric_quadratic(xv, A), cv))
        Z11, z = Z[..., :n, :n], Z[..., :n, n]
        P = xv @ A
        G = Z11 @ P

        jhj, jacobian = _fused_pfaffian_jhj(J, Z11, P, A, G)

        # border terms with C = v A x^T: 2 u^T Z11 (C - C^T) z - (z u)^2
        pz = jnp.einsum('...nm,...n->...m', P, z)
        Cz = -jnp.einsum('k...nm,...m->k...n', J, pz)
        CTz = -jnp.einsum('...nm,k...m->k...n', P, jnp.einsum('k...nm,...n->k...m', J, z))
        Ztu = -jnp.einsum('...ij,k...j->k...i', Z11, U)  # Z11^T u
        zu = jnp.einsum('...n,k...n->k...', z, U)
        jhj += 2 * jnp.einsum('k...n,k...n->...', Ztu, Cz - CTz)
        jhj -= jnp.einsum('k...,k...->...', zu, zu)
        jacobian -= zu
        laplacian = jhj + jnp.einsum('...nm,...nm->...', G, x.laplacian)
        laplacian -= jnp.einsum('...n,...n->...', z, c.laplacian)
        return sign, FwdLaplArray(log_pf, FwdJacobian.from_dense(jacobian), laplacian)

    def _pad_tangents(J: jax.Array, n_tangent: int) -> jax.Array:
        """Zero-pads the leading tangent dimension to n_tangent rows."""
        if J.shape[0] == n_tangent:
            return J
        pad = [(0, n_tangent - J.shape[0])] + [(0, 0)] * (J.ndim - 1)
        return jnp.pad(J, pad)

    folx.register_function(
        'skewsymmetric_quadratic',
        folx.wrap_forward_laplacian(
            skewsymmetric_quadratic,
            name='skewsymmetric_quadratic',
            flags=JOIN_JVP,
            custom_jac_hessian_jac=skewsymmetric_quadratic_jac_hessian_jac,
        ),
    )
    folx.register_function('slog_pfaffian', folx_slog_pfaffian)
    folx.register_function(
        'slog_pfaffian_skewsymmetric_quadratic',
        folx_slog_pfaffian_skewsymmetric_quadratic,
    )
    folx.register_function(
        'slog_pfaffian_bordered_quadratic',
        folx_slog_pfaffian_bordered_quadratic,
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
