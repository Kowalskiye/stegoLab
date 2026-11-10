"""Bit-plane LSB steganography (vectorised, bit-plane selectable, optional keyed shuffling).

Built on top of RobinDavid/LSB-Steganography (MIT, see vendor/). The original fills bit
plane 0 first, then plane 1, ... walking pixels in raster order with channels interleaved.
This module keeps exactly that slot order but:
  * is fully vectorised with NumPy (the original loops per bit in pure Python),
  * lets you choose any bit planes (0 = LSB ... 7 = MSB) and channels,
  * can shuffle slot order with a secret key (harder to detect/extract),
  * works on bits so SVD/PCA methods can share the same payload framing.
"""
import hashlib

import numpy as np


def key_to_seed(key) -> int:
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "big")


class LSBMethod:
    name = "lsb"

    def __init__(self, planes=(0,), channels=None, key=None):
        self.planes = tuple(int(p) for p in planes)
        if not self.planes or any(p < 0 or p > 7 for p in self.planes):
            raise ValueError("planes must be integers in 0..7")
        self.channels = None if channels is None else tuple(channels)
        self.key = key

    # ---- slot geometry -------------------------------------------------
    def _chans(self, arr):
        ch = self.channels if self.channels is not None else tuple(range(arr.shape[2]))
        if any(c >= arr.shape[2] for c in ch):
            raise ValueError("channel index out of range")
        return ch

    def capacity_bits(self, arr) -> int:
        h, w, _ = arr.shape
        return len(self.planes) * h * w * len(self._chans(arr))

    def _order(self, total, n):
        if self.key is None:
            return np.arange(n)
        rng = np.random.default_rng(key_to_seed(self.key))
        return rng.permutation(total)[:n]

    def _slot_view(self, arr):
        ch = list(self._chans(arr))
        return arr[:, :, ch].reshape(-1), len(ch)

    # ---- embed / extract -------------------------------------------------
    def embed_bits(self, arr, bits):
        bits = np.asarray(bits, dtype=np.uint8)
        total = self.capacity_bits(arr)
        if len(bits) > total:
            raise ValueError(f"Payload needs {len(bits)} bits but capacity is {total}")
        out = arr.copy()
        flat, _ = self._slot_view(out)
        per_plane = flat.size
        order = self._order(total, len(bits))
        plane_idx = order // per_plane
        pos = order % per_plane
        for pi, p in enumerate(self.planes):
            m = plane_idx == pi
            if not m.any():
                continue
            sel = pos[m]
            v = flat[sel]
            v = (v & np.uint8(255 - (1 << p))) | (bits[m].astype(np.uint8) << np.uint8(p))
            flat[sel] = v
        ch = list(self._chans(out))
        out[:, :, ch] = flat.reshape(arr.shape[0], arr.shape[1], len(ch))
        return out

    def extract_bits(self, arr, nbits):
        total = self.capacity_bits(arr)
        if nbits > total:
            raise ValueError("Requested more bits than the image can hold")
        flat, _ = self._slot_view(arr)
        per_plane = flat.size
        order = self._order(total, nbits)
        plane_idx = order // per_plane
        pos = order % per_plane
        out = np.zeros(nbits, dtype=np.uint8)
        for pi, p in enumerate(self.planes):
            m = plane_idx == pi
            out[m] = (flat[pos[m]] >> np.uint8(p)) & 1
        return out

    def describe(self):
        return {"method": "lsb", "planes": list(self.planes), "channels": self.channels, "keyed": self.key is not None}
