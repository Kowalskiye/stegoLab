"""High-level hide / reveal: framing + Reed-Solomon ECC + repetition coding + metrics.

Bit stream layout:   [ header (RS-coded, fixed parity) ] x repeat  ||  [ body (RS-coded, nsym parity) ] x repeat
"""
import numpy as np

from . import payload as P


class CapacityError(Exception):
    pass


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _check(repeat, ecc):
    if repeat < 1 or repeat % 2 == 0:
        raise ValueError("repeat must be an odd number (1, 3, 5, ...)")
    if not 0 <= ecc <= 200 or ecc % 2:
        raise ValueError("ecc must be an even number of parity bytes between 0 and 200")


def _majority(bits, n, repeat):
    return (bits.reshape(repeat, n).sum(axis=0) * 2 > repeat).astype(np.uint8)


def bits_needed(nbytes, repeat=1, ecc=0):
    return (P.HEADER_CODED_BYTES + P.coded_length(nbytes, ecc)) * 8 * repeat


def max_payload_bytes(capacity_bits, repeat=1, ecc=0):
    """Largest payload (bytes) that fits into capacity_bits with the given coding."""
    lo, hi = 0, capacity_bits // 8
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bits_needed(mid, repeat, ecc) <= capacity_bits:
            lo = mid
        else:
            hi = mid - 1
    return lo


def hide(arr, data: bytes, kind, method, repeat=1, ecc=0):
    _check(repeat, ecc)
    header = P.rs_encode(P.make_header(data, kind, ecc), P.HEADER_NSYM)
    body = P.rs_encode(data, ecc)
    hb, bb = P.bytes_to_bits(header), P.bytes_to_bits(body)
    coded = np.concatenate([np.tile(hb, repeat), np.tile(bb, repeat)])
    cap = method.capacity_bits(arr)
    if len(coded) > cap:
        raise CapacityError(f"Payload needs {len(coded)} bits (repeat={repeat}, ecc={ecc}) but capacity is {cap} bits "
                            f"(max ~{max_payload_bytes(cap, repeat, ecc)} bytes). Use a bigger cover, lower --repeat/--ecc, "
                            f"or more planes/components/singular values.")
    stego = method.embed_bits(arr, coded)
    report = {"payload_bytes": len(data), "bits_embedded": int(len(coded)), "capacity_bits": int(cap),
              "capacity_used_pct": round(100 * len(coded) / cap, 2), "psnr_db": round(psnr(arr, stego), 2),
              "repeat": repeat, "ecc_parity_bytes": ecc, **method.describe(), **getattr(method, "last_report", {})}
    return stego, report


def reveal(arr, method, repeat=1):
    _check(repeat, 0)
    nh = P.HEADER_CODED_BYTES * 8
    cap = method.capacity_bits(arr)
    if nh * repeat > cap:
        raise P.PayloadError("Image too small to contain a payload")
    hbits = _majority(method.extract_bits(arr, nh * repeat), nh, repeat)
    try:
        header = P.rs_decode(P.bits_to_bytes(hbits), P.HEADER_NSYM)
    except P.PayloadError:
        header = b""
    kind, nsym, length, crc = P.parse_header(header)
    body_len = P.coded_length(length, nsym)
    if (nh + body_len * 8) * repeat > cap:
        raise P.PayloadError("Header describes more data than the image can hold (wrong parameters/key?)")
    allbits = method.extract_bits(arr, (nh + body_len * 8) * repeat)
    body = _majority(allbits[nh * repeat:], body_len * 8, repeat) if body_len else np.zeros(0, np.uint8)
    data = P.rs_decode(P.bits_to_bytes(body)[:body_len], nsym)[:length]
    P.verify(data, crc)
    return kind, data
