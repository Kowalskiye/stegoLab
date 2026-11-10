"""Payload framing: header (magic, type, length, CRC32) + bytes <-> bit arrays.

Header layout (13 bytes): MAGIC(4) | TYPE(1) | LENGTH(4, big endian) | CRC32(4)
TYPE: 0 = text, 1 = file, 2 = image (PNG bytes)
"""
import io
import struct
import zlib

import numpy as np
from PIL import Image
from reedsolo import ReedSolomonError, RSCodec

MAGIC = b"SGL1"
HEADER_BYTES = 14          # MAGIC(4) KIND(1) ECC_NSYM(1) LENGTH(4) CRC32(4)
HEADER_NSYM = 10           # fixed Reed-Solomon parity protecting the header itself
HEADER_CODED_BYTES = HEADER_BYTES + HEADER_NSYM
TEXT, FILE, IMAGE = 0, 1, 2
EXT = {TEXT: ".txt", FILE: ".bin", IMAGE: ".png"}


class PayloadError(Exception):
    pass


def make_header(data: bytes, kind: int, nsym: int) -> bytes:
    return MAGIC + bytes([kind, nsym]) + struct.pack(">I", len(data)) + struct.pack(">I", zlib.crc32(data))


def parse_header(header: bytes):
    if len(header) < HEADER_BYTES or header[:4] != MAGIC:
        raise PayloadError("No valid hidden payload found (wrong method, parameters, key or keyfile?)")
    kind, nsym = header[4], header[5]
    length = struct.unpack(">I", header[6:10])[0]
    crc = struct.unpack(">I", header[10:14])[0]
    return kind, nsym, length, crc


def verify(data: bytes, crc: int):
    if zlib.crc32(data) != crc:
        raise PayloadError("Payload CRC mismatch: data was corrupted (raise --ecc/--repeat or --delta, and save as lossless PNG)")


def rs_encode(data: bytes, nsym: int) -> bytes:
    return bytes(RSCodec(nsym).encode(data)) if nsym > 0 else data


def rs_decode(data: bytes, nsym: int) -> bytes:
    if nsym <= 0:
        return data
    try:
        return bytes(RSCodec(nsym).decode(data)[0])
    except ReedSolomonError as e:
        raise PayloadError(f"Too many errors for Reed-Solomon to repair ({e}); raise --ecc or --repeat") from e


def coded_length(nbytes: int, nsym: int) -> int:
    """Length of RS-coded data (reedsolo splits into 255-byte codewords)."""
    if nsym <= 0 or nbytes == 0:
        return nbytes
    chunks = -(-nbytes // (255 - nsym))
    return nbytes + chunks * nsym


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def image_to_png_bytes(img: Image.Image, max_side=None, grayscale=False) -> bytes:
    img = img.convert("L" if grayscale else "RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fit_secret_image(img: Image.Image, max_bytes: int, grayscale=True) -> bytes:
    """Find the highest resolution thumbnail whose optimized PNG fits in max_bytes.
    
    Uses binary search on image dimension to maximize resolution and minimize downscaling loss.
    """
    low = 8
    high = max(img.size)
    best_data = None

    while low <= high:
        mid = (low + high) // 2
        try:
            data = image_to_png_bytes(img, max_side=mid, grayscale=grayscale)
            if len(data) <= max_bytes:
                best_data = data
                low = mid + 1  # Seek larger dimension
            else:
                high = mid - 1  # Seek smaller dimension
        except Exception:
            high = mid - 1

    if best_data is not None:
        return best_data

    # Fallback to direct check at minimum bound
    data = image_to_png_bytes(img, max_side=8, grayscale=grayscale)
    if len(data) <= max_bytes:
        return data

    raise PayloadError(f"Secret image cannot be shrunk to fit {max_bytes} bytes; use a bigger cover or smaller block size")
