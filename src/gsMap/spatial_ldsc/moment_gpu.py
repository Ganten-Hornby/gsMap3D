"""Optional GPU four-column moments; imported only after CUDA is selected.

Cell-specific weights and trait-specific SNP masks/jackknife blocks are retained.
Triton fuses the large reductions; JAX solves only the small normal equations.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
import triton
import triton.language as tl

from .moment_design import prepare_moment_design


@triton.jit
def _moments(
    LD,
    POS,
    FEATURES,
    LENGTHS,
    NBAR,
    H,
    OUT,
    C: tl.constexpr,
    PAD: tl.constexpr,
    TILES: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    F64: tl.constexpr,
    FAST: tl.constexpr,
    N_BLOCKS: tl.constexpr,
):
    tile = tl.program_id(0)
    trait = tl.program_id(2)
    block = tile // TILES
    subtile = tile % TILES
    r = subtile * BS + tl.arange(0, BS)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    width = tl.load(LENGTHS + trait) // N_BLOCKS
    s = block * width + r
    valid = r < width
    p = tl.load(POS + trait * PAD + s, mask=valid, other=0)
    ld = tl.load(LD + p[:, None] * C + c[None, :], mask=valid[:, None] & (c[None, :] < C), other=0)
    # SoA features: original N; scaled base/all-gene; response; base LD sum;
    # max(w_ld,1). Preserve original multiply/divide order for the spatial term.
    off = trait * 6 * PAD + s
    n = tl.load(FEATURES + off, mask=valid, other=0)
    b0 = tl.load(FEATURES + off + PAD, mask=valid, other=0)
    b1 = tl.load(FEATURES + off + 2 * PAD, mask=valid, other=0)
    y = tl.load(FEATURES + off + 3 * PAD, mask=valid, other=0)
    total = tl.load(FEATURES + off + 4 * PAD, mask=valid, other=0)
    wl = tl.load(FEATURES + off + 5 * PAD, mask=valid, other=1)
    nb = tl.load(NBAR + trait)
    h = tl.load(H + trait * C + c, mask=c < C, other=0)
    if FAST:
        x = (ld * n[:, None]) / nb
        k = (h[None, :] * n[:, None]) / 10000.0
    else:
        x = tl.div_rn(ld * n[:, None], nb)
        k = tl.div_rn(h[None, :] * n[:, None], 10000.0)
    denom = 1.0 + k * tl.maximum(ld + total[:, None], 1.0)
    if FAST:
        w = 1.0 / ((2.0 * denom * denom) * wl[:, None])
    else:
        w = tl.div_rn(1.0, (2.0 * denom * denom) * wl[:, None])
    w = tl.where(valid[:, None], w, 0.0)
    # Packed upper triangle of [spatial,base,all-gene,intercept], then X'y.
    for field in tl.static_range(14):
        if field == 0:
            v = x * x * w
        elif field == 1:
            v = x * b0[:, None] * w
        elif field == 2:
            v = x * b1[:, None] * w
        elif field == 3:
            v = x * w
        elif field == 4:
            v = (b0 * b0)[:, None] * w
        elif field == 5:
            v = (b0 * b1)[:, None] * w
        elif field == 6:
            v = b0[:, None] * w
        elif field == 7:
            v = (b1 * b1)[:, None] * w
        elif field == 8:
            v = b1[:, None] * w
        elif field == 9:
            v = w
        elif field == 10:
            v = x * y[:, None] * w
        elif field == 11:
            v = (b0 * y)[:, None] * w
        elif field == 12:
            v = (b1 * y)[:, None] * w
        else:
            v = y[:, None] * w
        if F64:
            value = tl.sum(v.to(tl.float64), axis=0).to(tl.float32)
        else:
            value = tl.sum(v, axis=0)
        out = ((trait * N_BLOCKS * TILES + tile) * 14 + field) * C + c
        tl.store(OUT + out, value, mask=c < C)


@partial(jax.jit, static_argnums=(0,))
def finish_moments(n_blocks, partials, nbar):
    packed = partials.sum(axis=2).transpose(0, 1, 3, 2)
    indices = jnp.array([[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]])
    xx, xy = packed[..., indices], packed[..., 10:14]
    full_x, full_y = xx.sum(axis=1), xy.sum(axis=1)
    full = jnp.linalg.solve(full_x, full_y[..., None])[..., 0]
    deleted = jnp.linalg.solve(full_x[:, None] - xx, (full_y[:, None] - xy)[..., None])[..., 0]
    pseudo = n_blocks * full[:, None, :, 0] - (n_blocks - 1) * deleted[..., 0]
    beta = pseudo.mean(axis=1)
    se = jnp.sqrt(jnp.sum((pseudo - beta[:, None]) ** 2, axis=1) / (n_blocks * (n_blocks - 1)))
    return jnp.stack((beta / nbar[:, None], se / nbar[:, None]), axis=1)


class GPUMomentRegression:
    """Returns results and explicit producer owners for the bounded async queue."""

    def __init__(self, sparse, prepared, n_blocks, ld_backend, trait_batch=4):
        design = prepare_moment_design(sparse, prepared, n_blocks)
        self.backend = ld_backend
        self.n_blocks = n_blocks
        self.trait_batch = trait_batch
        self.count = len(prepared)
        self.pad = design["positions"].shape[1]
        self.bs, self.bc = 256, 16
        self.tiles = triton.cdiv(int(design["lengths"].max()) // n_blocks, self.bs)
        self.design = {k: torch.as_tensor(v, device=ld_backend.device) for k, v in design.items()}

    def compute(self, marker):
        marker = np.ascontiguousarray(marker, dtype=np.float32)
        if marker.ndim != 2 or marker.shape[0] != self.design["q"].shape[1] or marker.shape[1] < 1:
            raise ValueError("Marker dimensions do not match SNP-gene matrix")
        if not np.isfinite(marker).all():
            raise ValueError("Non-finite marker scores")
        ld, gpu_marker = self.backend.compute_torch(marker)
        p = self.design
        # This small projection must use FP32 products, even if the embedding
        # stage enabled TF32 in the same process. Restore the caller's setting.
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            h = torch.clamp(
                p["numerator"] / (p["q"] @ gpu_marker + p["offset"]), 0, 1
            ).contiguous()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        cells = marker.shape[1]
        results, owners = [], [ld, gpu_marker, h]
        for start in range(0, self.count, self.trait_batch):
            end = min(start + self.trait_batch, self.count)
            output = torch.empty(
                (end - start, self.n_blocks, self.tiles, 14, cells),
                dtype=torch.float32,
                device=self.backend.device,
            )
            _moments[(self.n_blocks * self.tiles, triton.cdiv(cells, self.bc), end - start)](
                ld,
                p["positions"][start:end],
                p["features"][start:end],
                p["lengths"][start:end],
                p["nbar"][start:end],
                h[start:end],
                output,
                cells,
                self.pad,
                self.tiles,
                self.bs,
                self.bc,
                False,
                True,
                self.n_blocks,
                num_warps=4,
                enable_fp_fusion=True,
            )
            results.append(
                finish_moments(
                    self.n_blocks,
                    jax.dlpack.from_dlpack(output),
                    jax.dlpack.from_dlpack(p["nbar"][start:end]),
                )
            )
            if self.count > 16:
                # Hundreds of traits must not retain every SNP-tile workspace
                # across all in-flight cell chunks (several GiB on an 11G GPU).
                jax.block_until_ready(results[-1])
            else:
                owners.append(output)
        return results, owners

    def close(self):
        self.design.clear()
