"""Pallas GPU kernels for the pfaffian wave function.

`pfaffian_fwd_lapl` (and its bordered variant for odd electron counts)
evaluates the full forward laplacian of log|pf(x A x^T)| per batch element in
a single kernel: it forms P = x A and the skew matrix y = P x^T, computes sign
and log|pf| by in-register householder tridiagonalization, inverts y by
Gauss-Jordan elimination, and streams the tangents v_k = J[k] once,
accumulating

    s_k = -<Z, (v A) v^T + (C Z) C^T> - <C Z, (C Z)^T>   with C = v (x A)^T
    j_k = <v, Z P>

without materializing any J-sized intermediates. `slog_pfaffian_pallas`
computes only sign and log|pf| for the plain forward pass. Matrix tiles are
zero-padded to powers of two; dots use IEEE precision (no tf32).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_HI = jax.lax.Precision.HIGHEST
# probing caps: larger tiles exhaust registers on all current GPUs
_MAX_PROBE_N = 128
_MAX_PROBE_M = 256
PRIMAL_MAX_N = 256


def _next_pow2(x: int, minimum: int = 16) -> int:
    p = minimum
    while p < x:
        p *= 2
    return p


def _mask(shape: tuple[int, int], rows: int, cols: int) -> jax.Array:
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    return (row < rows) & (col < cols)


def _load2d(ref, idx, rows: int, cols: int) -> jax.Array:
    """Loads a zero-padded 2D tile from the trailing dims of ref."""
    mask = _mask((ref.shape[-2], ref.shape[-1]), rows, cols)
    for i in idx:
        if isinstance(i, (pl.Slice, slice)):  # non-squeezed leading dim
            mask = mask[None]
    return pl.load(ref, (*idx, slice(None), slice(None)), mask=mask, other=0.0)


def _fwd_lapl_kernel(
    x_ref, A_ref, J_ref, lx_ref, *refs, n_k: int, n: int, m: int, large: bool = False
):
    """Forward, jacobian and laplacian of log|pf| in a single pass.

    In the large variant A_ref holds the precomputed W = J A (tangent-major)
    and refs starts with P_ref, so the (m, m) matrix A never enters the
    kernel. For the bordered variant refs continues with (c_ref, U_ref,
    lc_ref); the last four refs are always the outputs (sign, log, lapl, jac).
    """
    if large:
        P_ref, *refs = refs
    bordered = len(refs) == 7
    if bordered:
        c_ref, U_ref, lc_ref = refs[:3]
    sign_ref, log_ref, lapl_ref, jac_ref = refs[-4:]

    x = _load2d(x_ref, (), n, m)
    if large:
        P = _load2d(P_ref, (), n, m)
    else:
        A = _load2d(A_ref, (), m, m)
        P = jnp.dot(x, A, precision=_HI)
    y = jnp.dot(P, x.T, precision=_HI)
    y = (y - y.T) / 2
    n_p = y.shape[-1]
    row = jax.lax.broadcasted_iota(jnp.int32, (n_p,), 0)
    row2, col2 = row[:, None], row[None, :]

    if bordered:
        cv = _load1d(c_ref, (), n)
        y = y + cv[:, None] * (col2 == n) - cv[None, :] * (row2 == n)
        n_eff = n + 1
    else:
        n_eff = n
    sign, log_pf = _householder_slog_pf(y, n_eff)
    Z = _gauss_jordan_inv(y, n_eff)
    Z = (Z - Z.T) / 2
    if bordered:
        z = jnp.where(col2 == n, Z, 0.0).sum(1)
        Z = jnp.where((row2 < n) & (col2 < n), Z, 0.0)
    G = jnp.dot(Z, P, precision=_HI)

    lapl = (G * _load2d(lx_ref, (), n, m)).sum()
    if bordered:
        lapl -= (z * _load1d(lc_ref, (), n)).sum()

    def body(k, acc):
        v = _load2d(J_ref, (k,), n, m)
        if large:
            w = _load2d(A_ref, (k,), n, m)  # precomputed v A
        else:
            w = jnp.dot(v, A, precision=_HI)  # v A
        c = jnp.dot(v, P.T, precision=_HI)  # v (x A)^T = -(v A x^T)
        f = jnp.dot(c, Z, precision=_HI)
        s1 = jnp.dot(w, v.T, precision=_HI)
        s3 = jnp.dot(f, c.T, precision=_HI)
        s = -(Z * (s1 + s3)).sum() - (f * f.T).sum()
        j = (v * G).sum()
        if bordered:
            u = _load1d(U_ref, (k,), n)
            # 2 u^T Z (C - C^T) z - (z u)^2 with C = -c
            Dz = (c * z[:, None]).sum(0) - (c * z[None, :]).sum(1)
            Ztu = (Z * u[:, None]).sum(0)
            zu = (z * u).sum()
            s += 2 * (Ztu * Dz).sum() - zu * zu
            j -= zu
        pl.store(jac_ref, (0, pl.dslice(k, 1)), j.reshape(1))
        return acc + s

    total = jax.lax.fori_loop(0, n_k, body, jnp.zeros((), x.dtype))
    pl.store(sign_ref, (pl.dslice(0, 1),), sign.reshape(1))
    pl.store(log_ref, (pl.dslice(0, 1),), log_pf.reshape(1))
    pl.store(lapl_ref, (pl.dslice(0, 1),), (total + lapl).reshape(1))


def _load1d(ref, idx, rows: int) -> jax.Array:
    """Loads a zero-padded 1D tile from the trailing dim of ref."""
    mask = jax.lax.broadcasted_iota(jnp.int32, (ref.shape[-1],), 0) < rows
    for i in idx:
        if isinstance(i, (pl.Slice, slice)):  # non-squeezed leading dim
            mask = mask[None]
    return pl.load(ref, (*idx, slice(None)), mask=mask, other=0.0)


def _householder_slog_pf(M0: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Sign and log|pf| of a skew-symmetric register tile via householder."""
    row = jax.lax.broadcasted_iota(jnp.int32, (M0.shape[-1],), 0)
    dtype = M0.dtype

    def body(k, carry):
        M, sign, log = carry
        # column k below the diagonal
        col = jnp.where(row[None, :] == k, M, 0.0).sum(1)
        xv = jnp.where(row > k, col, 0.0)
        x_norm_sq = (xv * xv).sum()
        first = (row == k + 1).astype(dtype)
        x0 = (xv * first).sum()
        sign_x0 = jnp.sign(x0)
        alpha = -sign_x0 * jnp.sqrt(x_norm_sq)
        v = xv - alpha * first
        v = v * jax.lax.rsqrt(x_norm_sq - 2 * x0 * alpha + alpha * alpha)
        # skew two-sided reflection: M + 2 v (M v)^T - 2 (M v) v^T
        Mv = (M * v[None, :]).sum(1)
        M = M + 2 * (v[:, None] * Mv[None, :] - Mv[:, None] * v[None, :])
        # only every other pivot contributes to the pfaffian
        use = (k & 1) == 0
        sign = sign * jnp.where(use, sign_x0, 1.0)
        log = log + jnp.where(use, jnp.log(jnp.abs(alpha)), 0.0)
        return M, sign, log

    init = (M0, jnp.ones((), dtype), jnp.zeros((), dtype))
    M, sign, log = jax.lax.fori_loop(0, n - 2, body, init)
    last = jnp.where((row[:, None] == n - 2) & (row[None, :] == n - 1), M, 0.0).sum()
    sign = sign * jnp.sign(last)
    log = log + jnp.log(jnp.abs(last))
    return sign, log


def _gauss_jordan_inv(Y: jax.Array, n: int) -> jax.Array:
    """Inverse of the leading (n, n) block of a register tile, zero outside.

    Gauss-Jordan elimination with partial pivoting; rows and columns beyond n
    stay zero.
    """
    n_p = Y.shape[-1]
    row = jax.lax.broadcasted_iota(jnp.int32, (n_p,), 0)
    row2, col2 = row[:, None], row[None, :]
    E0 = ((row2 == col2) & (row2 < n)).astype(Y.dtype)

    def body(k, carry):
        M, E = carry
        colk = jnp.where(col2 == k, M, 0.0).sum(1)
        cand = jnp.where((row >= k) & (row < n), jnp.abs(colk), -1.0)
        r = jnp.where(cand == cand.max(), row, n_p).min()
        is_k, is_r = row2 == k, row2 == r

        def swap(M):
            rk = jnp.where(is_k, M, 0.0).sum(0)
            rr = jnp.where(is_r, M, 0.0).sum(0)
            return M + is_k * (rr - rk)[None, :] + is_r * (rk - rr)[None, :]

        M, E = swap(M), swap(E)
        pv = jnp.where(is_k & (col2 == k), M, 0.0).sum()
        rk_M = jnp.where(is_k, M, 0.0).sum(0) / pv
        rk_E = jnp.where(is_k, E, 0.0).sum(0) / pv
        factor = jnp.where(row == k, 0.0, jnp.where(col2 == k, M, 0.0).sum(1))
        M = jnp.where(is_k, rk_M[None, :], M - factor[:, None] * rk_M[None, :])
        E = jnp.where(is_k, rk_E[None, :], E - factor[:, None] * rk_E[None, :])
        return M, E

    _, E = jax.lax.fori_loop(0, n, body, (Y, E0))
    return E


def _slog_pfaffian_kernel(Y_ref, sign_ref, log_ref, *, n: int):
    sign, log = _householder_slog_pf(_load2d(Y_ref, (), n, n), n)
    pl.store(sign_ref, (pl.dslice(0, 1),), sign.reshape(1))
    pl.store(log_ref, (pl.dslice(0, 1),), log.reshape(1))


def supports_slog_pfaffian(A: jax.Array) -> bool:
    """Whether the pallas kernel can compute the pfaffian of these matrices.

    Args:
        A: Skew-symmetric matrices of shape (..., n, n).

    Returns:
        True if the kernel supports the dtype, backend and padded tile size.
    """
    n = A.shape[-1]
    return (
        jax.default_backend() == 'gpu'
        and A.dtype == jnp.float32
        and n % 2 == 0
        and n >= 4
        and _next_pow2(n) <= PRIMAL_MAX_N
    )


@jax.jit
def slog_pfaffian_pallas(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sign and log|pf| of skew-symmetric matrices via in-register householder.

    Args:
        A: Skew-symmetric matrices of shape (..., n, n), n even.

    Returns:
        Tuple of sign and log|pf|, each of shape (...,).
    """
    *batch, n, _ = A.shape
    d = 1
    for b in batch:
        d *= b
    n_p = _next_pow2(n)
    kernel = functools.partial(_slog_pfaffian_kernel, n=n)
    sign, log = pl.pallas_call(
        kernel,
        grid=(d,),
        in_specs=[pl.BlockSpec((None, n_p, n_p), lambda i: (i, 0, 0))],
        out_specs=[
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((d,), A.dtype),
            jax.ShapeDtypeStruct((d,), A.dtype),
        ],
    )(A.reshape(d, n, n))
    return sign.reshape(tuple(batch)), log.reshape(tuple(batch))


def _fwd_lapl_call(operands, J, *, large: bool, bordered: bool, num_stages: int):
    """Builds and invokes the pallas call for the fwd-laplacian kernel.

    Args:
        operands: Kernel operands in ref order.
        J: Tangent-major jacobian, defining the shapes.
        large: Whether W = J A is precomputed (A never enters the kernel).
        bordered: Whether the bordered variant is used.
        num_stages: Triton pipeline stages.

    Returns:
        Tuple of sign (batch,), log|pf| (batch,), laplacian (batch,) and the
        output jacobian (batch, tangents).
    """
    n_tangent, batch, n, m = J.shape
    n_p, m_p = _next_pow2(n), _next_pow2(m)
    k_p = _next_pow2(n_tangent, 8)
    mat = lambda: pl.BlockSpec((None, n_p, m_p), lambda d: (d, 0, 0))  # noqa: E731
    vec = lambda: pl.BlockSpec((None, n_p), lambda d: (d, 0))  # noqa: E731
    scalar = lambda: pl.BlockSpec((1,), lambda d: (d,))  # noqa: E731
    tangent_mat = pl.BlockSpec((k_p, None, n_p, m_p), lambda d: (0, d, 0, 0))
    in_specs = [
        mat(),
        tangent_mat if large else pl.BlockSpec((None, m_p, m_p), lambda d: (d, 0, 0)),
        tangent_mat,
        mat(),
    ]
    if large:
        in_specs.append(mat())
    if bordered:
        in_specs += [
            vec(),
            pl.BlockSpec((k_p, None, n_p), lambda d: (0, d, 0)),
            vec(),
        ]
    kernel = functools.partial(_fwd_lapl_kernel, n_k=n_tangent, n=n, m=m, large=large)
    return pl.pallas_call(
        kernel,
        grid=(batch,),
        compiler_params=dict(triton=dict(num_stages=num_stages)),
        in_specs=in_specs,
        out_specs=[
            scalar(),
            scalar(),
            scalar(),
            pl.BlockSpec((1, k_p), lambda d: (d, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch,), J.dtype),
            jax.ShapeDtypeStruct((batch,), J.dtype),
            jax.ShapeDtypeStruct((batch,), J.dtype),
            jax.ShapeDtypeStruct((batch, n_tangent), J.dtype),
        ],
    )(*operands)


@functools.partial(jax.jit, static_argnames=('large', 'num_stages'))
def pfaffian_fwd_lapl(
    x: jax.Array,
    A: jax.Array,
    J: jax.Array,
    lapl_x: jax.Array,
    P: jax.Array | None = None,
    W: jax.Array | None = None,
    large: bool = False,
    num_stages: int = 3,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward laplacian of log|pf(x A x^T)| in a single kernel pass.

    Args:
        x: Orbital matrices of shape (batch, n, m), n even.
        A: Skew-symmetric matrices of shape (batch, m, m).
        J: Jacobian of x, shape (tangents, batch, n, m).
        lapl_x: Laplacian of x, shape (batch, n, m).
        P: Product x A for the large variant, shape (batch, n, m).
        W: Tangent products J A for the large variant, same shape as J.
        large: Whether to use the large variant (A stays off chip).
        num_stages: Triton pipeline stages.

    Returns:
        Tuple of sign (batch,), log|pf| (batch,), laplacian (batch,) and the
        output jacobian (batch, tangents).
    """
    if large:
        operands = (x, W, J, lapl_x, P)
    else:
        operands = (x, A, J, lapl_x)
    return _fwd_lapl_call(
        operands, J, large=large, bordered=False, num_stages=num_stages
    )


@functools.partial(jax.jit, static_argnames=('large', 'num_stages'))
def pfaffian_fwd_lapl_bordered(
    x: jax.Array,
    c: jax.Array,
    A: jax.Array,
    J: jax.Array,
    U: jax.Array,
    lapl_x: jax.Array,
    lapl_c: jax.Array,
    P: jax.Array | None = None,
    W: jax.Array | None = None,
    large: bool = False,
    num_stages: int = 3,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward laplacian of the bordered pfaffian in a single kernel pass.

    Args:
        x: Orbital matrices of shape (batch, n, m), n odd.
        c: Border vectors of shape (batch, n).
        A: Skew-symmetric matrices of shape (batch, m, m).
        J: Jacobian of x, shape (tangents, batch, n, m).
        U: Jacobian of c, shape (tangents, batch, n).
        lapl_x: Laplacian of x, shape (batch, n, m).
        lapl_c: Laplacian of c, shape (batch, n).
        P: Product x A for the large variant, shape (batch, n, m).
        W: Tangent products J A for the large variant, same shape as J.
        large: Whether to use the large variant (A stays off chip).
        num_stages: Triton pipeline stages.

    Returns:
        Tuple of sign (batch,), log|pf| (batch,), laplacian (batch,) and the
        output jacobian (batch, tangents).
    """
    if large:
        operands = (x, W, J, lapl_x, P, c, U, lapl_c)
    else:
        operands = (x, A, J, lapl_x, c, U, lapl_c)
    return _fwd_lapl_call(
        operands, J, large=large, bordered=True, num_stages=num_stages
    )


@functools.lru_cache(maxsize=None)
def _fwd_lapl_compiles(
    n_p: int, m_p: int, large: bool, bordered: bool, num_stages: int
) -> bool:
    """Whether the kernel compiles for these padded tiles on this device.

    Compile-only probe (no execution); the result is cached per process, so
    each distinct tile configuration is probed once.
    """
    f32 = jnp.float32
    n = n_p - 1 if bordered else n_p
    arg = lambda *s: jax.ShapeDtypeStruct(s, f32)  # noqa: E731
    x, mat, tan = arg(1, n, m_p), arg(1, m_p, m_p), arg(8, 1, n, m_p)
    operands = [x, tan if large else mat, tan, x]
    if large:
        operands.append(x)
    if bordered:
        operands += [arg(1, n), arg(8, 1, n), arg(1, n)]

    def call(*args):
        return _fwd_lapl_call(
            args, args[2], large=large, bordered=bordered, num_stages=num_stages
        )

    try:
        jax.jit(call).lower(*operands).compile()
        return True
    except Exception:  # noqa: BLE001
        return False


def pfaffian_fwd_lapl_config(
    J: jax.Array | jax.ShapeDtypeStruct, bordered: bool
) -> tuple[bool, int] | None:
    """Selects the fastest kernel configuration that fits the current device.

    Probes the on-chip variant first (A resident, fastest), then the large
    variant (P = x A and W = J A precomputed outside the kernel).

    Args:
        J: Tangent-major jacobian of shape (tangents, ..., n, m).
        bordered: Whether the bordered variant is needed.

    Returns:
        Tuple (large, num_stages) of the chosen configuration, or None if no
        kernel fits (dtype, backend or tile limits).
    """
    n, m = J.shape[-2], J.shape[-1]
    n_p, m_p = _next_pow2(n), _next_pow2(m)
    if (
        jax.default_backend() != 'gpu'
        or J.dtype != jnp.float32
        or n_p > _MAX_PROBE_N
        or m_p > _MAX_PROBE_M
    ):
        return None
    for large in (False, True):
        for num_stages in (3, 1):
            if _fwd_lapl_compiles(n_p, m_p, large, bordered, num_stages):
                return large, num_stages
    return None


def kernel_capabilities() -> str:
    """Probes and formats the kernel capability matrix for the current device.

    Returns:
        A table of the selected configuration per electron count.
    """
    device = jax.devices()[0].device_kind
    lines = [f'pfaffian_fwd_lapl capabilities on {device} (m = 2n):']
    for n in (8, 16, 24, 32, 48, 64, 96, 128):
        J = jax.ShapeDtypeStruct((3 * n, 1, n, 2 * n), jnp.float32)
        config = pfaffian_fwd_lapl_config(J, bordered=False)
        if config is None:
            desc = 'no kernel (decomposed rules)'
        else:
            large, num_stages = config
            desc = f'{"large (A off chip)" if large else "on-chip"} kernel, num_stages={num_stages}'
        lines.append(f'  n <= {n}: {desc}')
    return '\n'.join(lines)
