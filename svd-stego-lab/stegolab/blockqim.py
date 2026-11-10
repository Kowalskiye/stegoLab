"""Block-transform steganography with Quantization Index Modulation (QIM).

Two methods share one engine:
  * SVDMethod: bits live in the largest singular value(s) of every 8x8 block.
  * PCAMethod: bits live in the projection of every block on chosen principal components.

QIM rule with step delta: a coefficient c carries bit b if round(c/delta) % 2 == b.
Embedding moves c to the nearest lattice point of the right parity (max move = delta),
rebuilds the block, rounds/clips to uint8, then VERIFIES by re-analysing the block.
If clipping or rounding broke a bit, the next-nearest lattice point is tried (up to MAX_MODES candidates).
"""
import numpy as np

from .lsb import key_to_seed


# ---------------------------------------------------------------- block helpers
def split_blocks(arr, B):
    H, W, C = arr.shape
    nbh, nbw = H // B, W // B
    if nbh == 0 or nbw == 0:
        raise ValueError("Image smaller than one block")
    core = arr[: nbh * B, : nbw * B, :]
    t = core.transpose(2, 0, 1).reshape(C, nbh, B, nbw, B).transpose(0, 1, 3, 2, 4)
    return t.reshape(-1, B, B)  # index = c*nbh*nbw + by*nbw + bx


def merge_blocks(arr, blocks, B):
    H, W, C = arr.shape
    nbh, nbw = H // B, W // B
    t = blocks.reshape(C, nbh, nbw, B, B).transpose(0, 1, 3, 2, 4).reshape(C, nbh * B, nbw * B)
    out = arr.copy()
    out[: nbh * B, : nbw * B, :] = t.transpose(1, 2, 0)
    return out


MAX_MODES = 7


def eligible_blocks(blocks):
    """Blocks that are perfectly flat AND saturated (all 0 or all 255) cannot carry a bit
    without clipping. The sender never touches them, so the receiver can recompute this
    exact rule on the stego image and skip the same blocks."""
    mn = blocks.min(axis=(1, 2))
    mx = blocks.max(axis=(1, 2))
    return ~((mn == mx) & ((mn == 0) | (mn == 255)))


class BlockQIMMethod:
    name = "block"

    def __init__(self, block=8, delta=24.0, k=1, key=None, robust=False):
        if block < 2:
            raise ValueError("block must be >= 2")
        if delta <= 0:
            raise ValueError("delta must be > 0")
        if not 1 <= k <= block:
            raise ValueError("k must be between 1 and block")
        self.block, self.delta, self.k, self.key = int(block), float(delta), int(k), key
        # robust=True: block positions never depend on image content, so JPEG/noise cannot misalign
        # the bit stream (flat black/white blocks then simply produce errors that ECC repairs).
        self.robust = bool(robust)
        self.last_report = {}

    def _elig(self, blocks):
        return np.ones(len(blocks), dtype=bool) if self.robust else eligible_blocks(blocks)

    # subclasses implement these two
    def analyze(self, blocks):  # (n,B,B) float -> (n,k)
        raise NotImplementedError

    def synth(self, blocks, coeffs, targets):  # -> (n,B,B) float
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _nblocks(self, arr):
        H, W, C = arr.shape
        return C * (H // self.block) * (W // self.block)

    def capacity_bits(self, arr):
        elig = self._elig(split_blocks(arr, self.block))
        return int(elig.sum()) * self.k

    def _order(self, total, n):
        if self.key is None:
            return np.arange(n)
        return np.random.default_rng(key_to_seed(self.key)).permutation(total)[:n]

    def _targets(self, c, bits, valid, mode):
        d = self.delta
        u = c / d
        base = np.rint((u - bits) / 2.0) * 2.0 + bits  # nearest lattice index with right parity
        side = np.where(u >= base, 1.0, -1.0)
        # candidate lattice points of the right parity, nearest first: 0, +s, -s, +3s... order
        offsets = (0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0)
        q = base + offsets[mode] * side
        return np.where(valid, q * d, c)

    def _prepare(self, arr):
        if arr.ndim != 3 or arr.dtype != np.uint8:
            raise ValueError("expected uint8 H x W x C array")

    # ------------------------------------------------------------------
    def embed_bits(self, arr, bits):
        self._prepare(arr)
        bits = np.asarray(bits, dtype=np.uint8)
        k, B = self.k, self.block
        blocks = split_blocks(arr, B)
        elig_idx = np.where(self._elig(blocks))[0]
        total = len(elig_idx)
        nblk = -(-len(bits) // k)
        if nblk > total:
            raise ValueError(f"Payload needs {nblk} blocks but image has {total} usable blocks")
        order = elig_idx[self._order(total, nblk)]
        sel = blocks[order].astype(np.float64)

        padded = np.zeros(nblk * k, dtype=np.uint8)
        padded[: len(bits)] = bits
        valid = np.zeros(nblk * k, dtype=bool)
        valid[: len(bits)] = True
        bm, valid = padded.reshape(nblk, k), valid.reshape(nblk, k)

        coeffs = self.analyze(sel)
        out_blocks = sel.copy()
        done = np.zeros(nblk, dtype=bool)
        for mode in range(MAX_MODES):
            idx = np.where(~done)[0]
            if idx.size == 0:
                break
            q = self._targets(coeffs[idx], bm[idx], valid[idx], mode)
            new = np.clip(np.rint(self.synth(sel[idx], coeffs[idx], q)), 0, 255)
            chk = self.analyze(new)
            par = np.rint(chk / self.delta).astype(np.int64) & 1
            ok = ((par == bm[idx]) | ~valid[idx]).all(axis=1) & self._elig(new)
            out_blocks[idx] = new
            done[idx[ok]] = True
        blocks_out = blocks.copy()
        blocks_out[order] = out_blocks.astype(np.uint8)
        self.last_report = {"blocks_used": int(nblk), "unverified_blocks": int((~done).sum())}
        return merge_blocks(arr, blocks_out, B)

    def extract_bits(self, arr, nbits):
        self._prepare(arr)
        k = self.k
        blocks = split_blocks(arr, self.block)
        elig_idx = np.where(self._elig(blocks))[0]
        total = len(elig_idx)
        nblk = -(-nbits // k)
        if nblk > total:
            raise ValueError("Requested more bits than the image can hold")
        order = elig_idx[self._order(total, nblk)]
        c = self.analyze(blocks[order].astype(np.float64))
        bits = (np.rint(c / self.delta).astype(np.int64) & 1).astype(np.uint8).reshape(-1)
        return bits[:nbits]


class SVDMethod(BlockQIMMethod):
    name = "svd"

    def analyze(self, blocks):
        return np.linalg.svd(blocks, compute_uv=False)[:, : self.k]

    def synth(self, blocks, coeffs, targets):
        U, S, Vt = np.linalg.svd(blocks)
        S = S.copy()
        S[:, : self.k] = targets
        return (U * S[:, None, :]) @ Vt

    def describe(self):
        return {"method": "svd", "block": self.block, "delta": self.delta, "singular_values_per_block": self.k, "keyed": self.key is not None, "robust": self.robust}


class PCAMethod(BlockQIMMethod):
    """Bits in projections on principal components. The PCA basis (mean + eigenvectors)
    is learned from the cover image and saved as a KEYFILE that the receiver needs."""
    name = "pca"

    def __init__(self, block=8, delta=24.0, components=(0, 1, 2), key=None, basis=None, robust=False):
        comps = tuple(int(c) for c in components)
        super().__init__(block=block, delta=delta, k=len(comps), key=key, robust=robust)
        if any(c < 0 or c >= block * block for c in comps):
            raise ValueError("component index out of range")
        self.components = comps
        self.basis = basis  # dict(mean, W, eigvals)

    def fit(self, arr):
        X = split_blocks(arr, self.block).reshape(-1, self.block ** 2).astype(np.float64)
        mu = X.mean(axis=0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        self.basis = {"mean": mu, "W": vecs, "eigvals": vals}
        return self.basis

    def _W(self):
        if self.basis is None:
            raise RuntimeError("PCA basis missing: call fit(cover) when embedding or load a keyfile when extracting")
        return self.basis["W"][:, list(self.components)]

    def embed_bits(self, arr, bits):
        if self.basis is None:
            self.fit(arr)
        return super().embed_bits(arr, bits)

    def analyze(self, blocks):
        X = blocks.reshape(len(blocks), -1)
        return (X - self.basis["mean"]) @ self._W()

    def synth(self, blocks, coeffs, targets):
        X = blocks.reshape(len(blocks), -1)
        return (X + (targets - coeffs) @ self._W().T).reshape(blocks.shape)

    def save_keyfile(self, path):
        np.savez(path, mean=self.basis["mean"], W=self.basis["W"], eigvals=self.basis["eigvals"],
                 block=self.block, delta=self.delta, components=np.array(self.components), robust=int(self.robust))

    @classmethod
    def from_keyfile(cls, path, key=None):
        z = np.load(path)
        basis = {"mean": z["mean"], "W": z["W"], "eigvals": z["eigvals"]}
        return cls(block=int(z["block"]), delta=float(z["delta"]), components=tuple(int(c) for c in z["components"]), key=key, basis=basis,
                   robust=bool(z["robust"]) if "robust" in z.files else False)

    def describe(self):
        return {"method": "pca", "block": self.block, "delta": self.delta, "components": list(self.components), "keyed": self.key is not None, "robust": self.robust}
