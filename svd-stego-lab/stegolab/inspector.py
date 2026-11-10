"""Layer inspection, bit-plane separation, raw string extraction, and multi-mode auto-decoding.

Inspired by StegOnline (georgeom.net/StegOnline) and zsteg forensics.
"""
import base64
import io
import os
import re
import tempfile
import zipfile

import numpy as np
from PIL import Image

from . import core, detect, payload as P
from .blockqim import SVDMethod
from .lsb import LSBMethod


def get_bitplane(arr: np.ndarray, channel: int, plane: int, tinted: bool = False, invert: bool = False) -> np.ndarray:
    """Extract an 8-bit image of a single bitplane for a specific color channel.
    
    channel: 0=R, 1=G, 2=B
    plane: 0 (LSB) to 7 (MSB)
    """
    if channel < 0 or channel >= arr.shape[2]:
        raise ValueError(f"Invalid channel {channel}")
    if plane < 0 or plane > 7:
        raise ValueError(f"Invalid plane {plane}")

    bit = (arr[:, :, channel] >> np.uint8(plane)) & 1
    if invert:
        bit = 1 - bit

    if not tinted:
        return (bit * 255).astype(np.uint8)

    # Tinted with channel color
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[:, :, channel] = (bit * 255).astype(np.uint8)
    return out


def get_composite_plane(arr: np.ndarray, plane: int, invert: bool = False) -> np.ndarray:
    """Extract a composite RGB image where each channel displays the specified bitplane."""
    if plane < 0 or plane > 7:
        raise ValueError(f"Invalid plane {plane}")
    out = np.zeros_like(arr)
    for c in range(min(3, arr.shape[2])):
        bit = (arr[:, :, c] >> np.uint8(plane)) & 1
        if invert:
            bit = 1 - bit
        out[:, :, c] = (bit * 255).astype(np.uint8)
    return out


def get_channel_layer(arr: np.ndarray, channel: int, invert: bool = False) -> np.ndarray:
    """Extract an isolated color channel (Red only, Green only, Blue only)."""
    out = np.zeros_like(arr)
    ch_data = arr[:, :, channel]
    if invert:
        ch_data = 255 - ch_data
    out[:, :, channel] = ch_data
    return out


def arr_to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def arr_to_b64_png(arr: np.ndarray) -> str:
    return base64.b64encode(arr_to_png_bytes(arr)).decode("ascii")


def generate_all_layers_zip(arr: np.ndarray) -> bytes:
    """Pack all 24 bit planes + 8 composite planes into a downloadable ZIP archive."""
    buf = io.BytesIO()
    ch_names = ["red", "green", "blue"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 24 bit planes
        for c, ch in enumerate(ch_names):
            for p in range(8):
                desc = "lsb" if p == 0 else ("msb" if p == 7 else f"bit{p}")
                fname = f"bitplanes/{ch}_bit{p}_{desc}.png"
                bp = get_bitplane(arr, c, p, tinted=False)
                zf.writestr(fname, arr_to_png_bytes(bp))

        # 2. 8 Composite RGB planes
        for p in range(8):
            desc = "lsb" if p == 0 else ("msb" if p == 7 else f"bit{p}")
            fname = f"composite/rgb_plane{p}_{desc}.png"
            cp = get_composite_plane(arr, p)
            zf.writestr(fname, arr_to_png_bytes(cp))

        # 3. Channel isolations
        for c, ch in enumerate(ch_names):
            fname = f"channels/{ch}_channel_only.png"
            cl = get_channel_layer(arr, c)
            zf.writestr(fname, arr_to_png_bytes(cl))

    return buf.getvalue()


def extract_raw_strings(arr: np.ndarray, planes=(0,), channels=(0, 1, 2), min_len=4, max_strings=30):
    """Extract raw bits from selected planes & channels in raster order and find printable ASCII strings.
    
    Similar to zsteg / StegOnline string extraction.
    """
    ch = list(channels)
    flat = arr[:, :, ch].reshape(-1)
    bits_list = []
    for p in planes:
        bits_list.append((flat >> np.uint8(p)) & 1)
    all_bits = np.concatenate(bits_list)
    raw_bytes = np.packbits(all_bits).tobytes()

    pattern = re.compile(rb"[\x20-\x7e]{" + str(min_len).encode() + rb",}")
    matches = pattern.findall(raw_bytes[:200000])  # scan first 200KB for speed
    results = []
    seen = set()
    for m in matches:
        s = m.decode("ascii", errors="replace").strip()
        if len(s) >= min_len and s not in seen:
            seen.add(s)
            results.append(s)
            if len(results) >= max_strings:
                break
    return results


def _format_payload_result(kind: int, data: bytes):
    res = {
        "kind": {P.TEXT: "text", P.FILE: "file", P.IMAGE: "image"}.get(kind, "file"),
        "bytes": len(data),
    }
    if kind == P.TEXT:
        res["text"] = data.decode("utf-8", errors="replace")
    elif kind == P.IMAGE:
        res["data_b64"] = base64.b64encode(data).decode("ascii")
        try:
            im = Image.open(io.BytesIO(data))
            res["image_size"] = f"{im.width}x{im.height}"
        except Exception:
            res["image_size"] = "unknown"
    else:
        res["data_b64"] = base64.b64encode(data).decode("ascii")
    return res


def auto_decode_all_modes(arr: np.ndarray, key: str = None, keyfile_bytes: bytes = None):
    """Run extraction across LSB and SVD modes and compile a unified report."""
    report = {
        "summary": {
            "payload_found": False,
            "found_modes": [],
            "verdict": "No structured hidden payload found",
        },
        "lsb": {"status": "not_found", "configs_tested": 0},
        "svd": {"status": "not_found", "configs_tested": 0},
        "raw_strings": {},
    }

    # ----------------------------------------------------
    # 1. LSB AUTO-DECODE (Scan all planes 0-7, channels, and multi-plane sets)
    # ----------------------------------------------------
    lsb_candidates = []
    # Single planes 0..7 across RGB and single channels
    for p in range(8):
        desc = "LSB" if p == 0 else ("MSB" if p == 7 else f"Bit {p}")
        lsb_candidates.append(([p], None, f"Plane {p} {desc} (All Channels RGB)"))
        lsb_candidates.append(([p], [0], f"Plane {p} {desc} (Red Only)"))
        lsb_candidates.append(([p], [1], f"Plane {p} {desc} (Green Only)"))
        lsb_candidates.append(([p], [2], f"Plane {p} {desc} (Blue Only)"))
    # Multi-plane candidate groups
    lsb_candidates.extend([
        ([0, 1], None, "Planes 0 & 1 (RGB)"),
        ([0, 1, 2], None, "Planes 0, 1 & 2 (RGB)"),
        ([0, 1, 2, 3], None, "Planes 0-3 (RGB)"),
        ([4, 5, 6, 7], None, "Planes 4-7 MSB (RGB)"),
    ])

    keys_to_test = [key] if key else [None]
    if key and None not in keys_to_test:
        keys_to_test.append(None)

    lsb_found_items = []
    seen_payload_hashes = set()

    for planes, chans, label in lsb_candidates:
        for test_key in keys_to_test:
            report["lsb"]["configs_tested"] += 1
            try:
                m = LSBMethod(planes=planes, channels=chans, key=test_key)
                kind, data = core.reveal(arr, m)
                data_hash = hash(data)
                if data_hash in seen_payload_hashes:
                    continue
                seen_payload_hashes.add(data_hash)
                payload_info = _format_payload_result(kind, data)
                item = {
                    "status": "found",
                    "planes": planes,
                    "channels": chans or [0, 1, 2],
                    "key_used": test_key is not None,
                    "label": label,
                    "payload": payload_info,
                }
                lsb_found_items.append(item)
                if not report["summary"]["payload_found"]:
                    report["summary"]["payload_found"] = True
                if "LSB" not in report["summary"]["found_modes"]:
                    report["summary"]["found_modes"].append("LSB")
            except Exception:
                continue

    if lsb_found_items:
        # Build a fresh dict (not the same object as lsb_found_items[0]) to avoid
        # circular reference when "all_found" contains items[0] itself.
        first = lsb_found_items[0]
        report["lsb"] = {
            "status": "found",
            "planes": first["planes"],
            "channels": first["channels"],
            "key_used": first["key_used"],
            "label": first["label"],
            "payload": first["payload"],
            "all_found": lsb_found_items,
        }
    else:
        report["lsb"]["all_found"] = []

    # Extract raw ASCII strings for common planes (StegOnline / zsteg style)
    try:
        raw_rgb = extract_raw_strings(arr, planes=(0,), channels=(0, 1, 2))
        raw_r = extract_raw_strings(arr, planes=(0,), channels=(0,))
        raw_g = extract_raw_strings(arr, planes=(0,), channels=(1,))
        raw_b = extract_raw_strings(arr, planes=(0,), channels=(2,))
        report["raw_strings"] = {
            "plane0_rgb": raw_rgb[:15],
            "plane0_red": raw_r[:10],
            "plane0_green": raw_g[:10],
            "plane0_blue": raw_b[:10],
        }
    except Exception:
        report["raw_strings"] = {}

    # ----------------------------------------------------
    # 2. SVD AUTO-DECODE
    # ----------------------------------------------------
    # Run detector to find candidate lattice peaks
    analysis = detect.analyze(arr)
    detected_delta = analysis.get("svd_lattice", {}).get("delta")
    svd_deltas = [24.0, 16.0, 32.0, 12.0, 8.0, 20.0]
    if detected_delta and detected_delta not in svd_deltas:
        svd_deltas.insert(0, float(detected_delta))

    svd_found = False
    # If a lattice delta was detected, prioritize testing it first with both robust modes
    for robust in [False, True]:
        if svd_found:
            break
        for delta in svd_deltas:
            if svd_found:
                break
            for k in [1, 2]:
                if svd_found:
                    break
                for test_key in keys_to_test:
                    report["svd"]["configs_tested"] += 1
                    try:
                        m = SVDMethod(block=8, delta=delta, k=k, key=test_key, robust=robust)
                        kind, data = core.reveal(arr, m)
                        payload_info = _format_payload_result(kind, data)
                        report["svd"] = {
                            "status": "found",
                            "delta": delta,
                            "k": k,
                            "robust": robust,
                            "key_used": test_key is not None,
                            "payload": payload_info,
                        }
                        report["summary"]["payload_found"] = True
                        report["summary"]["found_modes"].append("SVD")
                        svd_found = True
                        break
                    except Exception:
                        continue

    if not svd_found:
        report["svd"]["lattice_info"] = analysis.get("svd_lattice", {})

    # ----------------------------------------------------
    # 3. MACHINE IDENTIFICATION CODE (MIC / TRACKING DOTS)
    # ----------------------------------------------------
    try:
        from . import mic
        mic_info = mic.detect_mic_in_pixels(arr)
        if mic_info:
            report["mic"] = mic_info
            report["summary"]["mic_detected"] = True
            if not report["summary"]["payload_found"]:
                report["summary"]["verdict"] = f"Machine Identification Code detected! Created: {mic_info['metadata']['date_str']} at {mic_info['metadata']['time_str']} (ID: #{mic_info['metadata']['custom_number']})"
    except Exception:
        pass

    # Verdict synthesis
    if report["summary"]["found_modes"]:
        modes_str = " & ".join(report["summary"]["found_modes"])
        report["summary"]["verdict"] = f"Payload successfully decoded via {modes_str}!"
    elif not report["summary"].get("mic_detected"):
        report["summary"]["verdict"] = "No standard structured payload detected. Check raw strings or try manual settings."

    return report

