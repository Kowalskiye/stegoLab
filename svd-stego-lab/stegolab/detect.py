"""High-accuracy blind steganalysis for LSB, SVD, and Machine Identification Codes.

Implements multi-scale statistical forensics:
1. Multi-scale & per-channel LSB analysis (Westfeld-Pfitzmann & bit correlation)
2. Prefix & full-scale SVD Quantization Index Modulation (QIM) lattice resonance
3. Direct stego framing signature probe (detects exact payload & parameters)
4. Printer tracking dot / Machine Identification Code (MIC) scan
"""
import numpy as np
from scipy import stats

from . import payload as P
from .blockqim import eligible_blocks, split_blocks
from .lsb import LSBMethod


DEFAULT_DELTAS = np.arange(6.0, 48.01, 0.2)


# -------------------------------------------------------------------------
# 1. HIGH-ACCURACY LSB STEGANALYSIS
# -------------------------------------------------------------------------
def lsb_chi_square(arr, windows=(16384, 65536)):
    """Calibrated per-channel and joint Chi-Square test on Pairs of Values (PoVs).
    
    Evaluates Red, Green, Blue, and combined channels on sample windows of sufficient
    statistical power (>= 16,384 pixels) to avoid small-sample false positives.
    """
    results = {}
    channels = {"R": 0, "G": 1, "B": 2, "RGB": [0, 1, 2]}
    max_overall_p = 0.0
    suspected_channels = []

    for name, ch in channels.items():
        if isinstance(ch, list):
            flat = arr[:, :, ch].reshape(-1)
        else:
            flat = arr[:, :, ch].reshape(-1)

        best_p = 0.0
        best_win = len(flat)

        test_windows = list(windows) + [len(flat)]
        for w in test_windows:
            if w > len(flat):
                continue
            sub = flat[:w]
            h = np.bincount(sub, minlength=256).astype(np.float64)
            even, odd = h[0::2], h[1::2]
            exp = (even + odd) / 2.0
            m = exp >= 15
            if m.sum() < 40:
                continue
            chi = (((even[m] - exp[m]) ** 2) / exp[m]).sum()
            p = 1.0 - stats.chi2.cdf(chi, df=int(m.sum()) - 1)
            if p > best_p:
                best_p = float(p)
                best_win = w

        results[name] = {"p_embedded": round(best_p, 4), "window_size": best_win}
        if best_p > 0.95 and name != "RGB":
            suspected_channels.append(name)
        if best_p > max_overall_p:
            max_overall_p = best_p

    return {
        "p_embedded": round(max_overall_p, 4),
        "detected": max_overall_p > 0.95,
        "suspected_channels": suspected_channels or (["RGB"] if max_overall_p > 0.95 else []),
        "channel_details": results,
    }


# -------------------------------------------------------------------------
# 2. ENHANCED SVD LATTICE DETECTOR (MULTI-SCALE & PREFIX AWARE)
# -------------------------------------------------------------------------
def _lattice_scan(values, deltas):
    """Scan candidate steps delta using circular mean R = |mean(exp(2*pi*i*v/delta))|."""
    v = np.asarray(values, dtype=np.float64)
    R = np.empty(len(deltas))
    for lo in range(0, len(deltas), 64):
        d = deltas[lo : lo + 64]
        R[lo : lo + 64] = np.abs(np.exp(2j * np.pi * v[:, None] / d[None, :]).mean(axis=0))
    med = float(np.median(R))
    mad = float(1.4826 * np.median(np.abs(R - med))) + 1e-6
    i = int(np.argmax(R))
    z = (R[i] - med) / mad
    return {
        "best_delta": round(float(deltas[i]), 1),
        "R": float(R[i]),
        "background": med,
        "z": float(z),
        "n": int(len(v)),
    }


def svd_lattice(arr, deltas=DEFAULT_DELTAS):
    """Multi-scale SVD lattice detector.
    
    Detects periodic QIM lattices both on full images and on localized block prefixes
    (vital for short messages where 95% of the cover blocks are untouched).
    """
    blocks = split_blocks(arr, 8)
    eligible = eligible_blocks(blocks)
    if eligible.sum() < 16:
        return {"detected": False, "z": 0.0, "R": 0.0, "best_delta": 24.0}

    s = np.linalg.svd(blocks[eligible].astype(np.float64), compute_uv=False)[:, 0]

    # Test candidate prefix block counts + full image
    prefixes = [64, 128, 256, 512, 1024, len(s)]
    best_scan = None
    best_score = -1.0

    for n in prefixes:
        if n > len(s):
            continue
        sub = s[:n]
        res = _lattice_scan(sub, deltas)
        # Score combines z-significance and peak magnitude R
        score = res["z"] * (1.0 + res["R"])
        if score > best_score:
            best_score = score
            best_scan = res
            best_scan["blocks_scanned"] = n

    # Detection threshold: z > 4.5 and peak R > 0.20, or strong z > 7.0
    detected = (best_scan["z"] > 4.5 and best_scan["R"] > 0.20) or (best_scan["z"] > 7.0)
    best_scan["detected"] = bool(detected)
    best_scan["delta"] = best_scan["best_delta"]
    return best_scan


# -------------------------------------------------------------------------
# 3. DIRECT SIGNATURE & MIC INSPECTOR
# -------------------------------------------------------------------------
def probe_stego_signature(arr):
    """Probe for standard SGL1 framing headers in candidate bit planes and channels."""
    nh = P.HEADER_CODED_BYTES * 8
    # Test common LSB arrangements
    for planes in ([0], [0, 1], [7]):
        for ch in (None, [0], [1], [2]):
            try:
                m = LSBMethod(planes=planes, channels=ch)
                bits = m.extract_bits(arr, nh)
                header_raw = P.bits_to_bytes(bits)
                try:
                    h = P.rs_decode(header_raw, P.HEADER_NSYM)
                except Exception:
                    h = header_raw
                if len(h) >= 4 and h[:4] == P.MAGIC:
                    kind, nsym, length, crc = P.parse_header(h)
                    kind_name = {P.TEXT: "text", P.FILE: "file", P.IMAGE: "image"}.get(kind, "file")
                    return {
                        "found": True,
                        "method": "LSB",
                        "kind": kind_name,
                        "payload_bytes": length,
                        "planes": planes,
                        "channels": "RGB" if ch is None else str(ch),
                    }
            except Exception:
                pass
    return {"found": False}


def probe_mic_tracking(arr):
    """Probe for printer-style Machine Identification Code tracking dots."""
    try:
        from . import mic
        detected = mic.detect_mic_in_pixels(arr)
        if detected:
            return {"found": True, "metadata": detected["metadata"]}
    except Exception:
        pass
    return {"found": False}


# -------------------------------------------------------------------------
# 4. MASTER COMPREHENSIVE STEGANALYSIS
# -------------------------------------------------------------------------
def analyze(arr):
    """Run comprehensive multi-tier blind steganalysis on an image."""
    sig = probe_stego_signature(arr)
    chi = lsb_chi_square(arr)
    sv = svd_lattice(arr)
    mic_info = probe_mic_tracking(arr)

    lsb_suspected = sig["found"] or chi["detected"]
    svd_suspected = sv.get("detected", False)
    mic_suspected = mic_info["found"]

    flags = {
        "lsb_suspected": bool(lsb_suspected),
        "svd_suspected": bool(svd_suspected),
        "mic_suspected": bool(mic_suspected),
        "signature_verified": bool(sig["found"]),
    }

    # Synthesize human-readable accurate verdict
    findings = []
    if sig["found"]:
        findings.append(f"Structured LSB Payload Confirmed ({sig['payload_bytes']} bytes, {sig['kind']})")
    elif chi["detected"]:
        ch_str = ", ".join(chi["suspected_channels"]) or "RGB"
        findings.append(f"LSB Replacement Anomaly Detected in channel(s) [{ch_str}] (p={chi['p_embedded']})")

    if sv.get("detected"):
        findings.append(f"SVD QIM Periodic Lattice Detected at delta={sv['delta']} (peak R={sv['R']:.2f}, Z={sv['z']:.1f})")

    if mic_info["found"]:
        meta = mic_info["metadata"]
        findings.append(f"Printer Machine Identification Code (MIC) Stamped: {meta['date_str']} at {meta['time_str']} (ID: #{meta['custom_number']})")

    if findings:
        verdict = "SUSPICIOUS — " + "; ".join(findings)
    else:
        verdict = "clean"

    return {
        "verdict": verdict,
        "flags": flags,
        "signature": sig,
        "lsb_chi_square": chi,
        "svd_lattice": sv,
        "mic": mic_info,
    }
