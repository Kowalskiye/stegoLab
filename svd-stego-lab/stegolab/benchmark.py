"""Benchmark sweep: capacity vs. imperceptibility vs. detectability vs. robustness."""
import csv
import io
import os

import numpy as np
from PIL import Image

from . import core, detect, payload as P
from .blockqim import PCAMethod, SVDMethod
from .imageio import load_image
from .lsb import LSBMethod


def _jpeg(arr, q):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _noise(arr, sigma, rng):
    return np.clip(np.rint(arr + rng.normal(0, sigma, arr.shape)), 0, 255).astype(np.uint8)


def configs():
    return [
        ("LSB plane0", lambda: LSBMethod([0]), 0),
        ("LSB planes0-1", lambda: LSBMethod([0, 1]), 0),
        ("SVD d=12", lambda: SVDMethod(8, 12, robust=True), 32),
        ("SVD d=24", lambda: SVDMethod(8, 24, robust=True), 32),
        ("SVD d=48", lambda: SVDMethod(8, 48, robust=True), 32),
        ("SVD d=24 k=2", lambda: SVDMethod(8, 24, k=2, robust=True), 32),
        ("PCA d=12 c=3", lambda: PCAMethod(8, 12, (0, 1, 2), robust=True), 32),
        ("PCA d=24 c=3", lambda: PCAMethod(8, 24, (0, 1, 2), robust=True), 32),
        ("PCA d=48 c=3", lambda: PCAMethod(8, 48, (0, 1, 2), robust=True), 32),
    ]


def run(cover_path, out_dir, fill=0.5, seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    cover = load_image(cover_path)
    rng = np.random.default_rng(seed)
    rows = []
    for name, mk, ecc in configs():
        m = mk()
        cap = m.capacity_bits(cover)
        n = int(core.max_payload_bytes(cap, 1, ecc) * fill)
        data = rng.bytes(n)
        stego, rep = core.hide(cover, data, P.FILE, m, ecc=ecc)
        try:
            ok = core.reveal(stego, m)[1] == data
        except Exception:
            ok = False
        det = detect.analyze(stego)
        # raw bit error rate after attacks (no ECC) on the same embedding
        nbits = rep["bits_embedded"]
        coded = m.extract_bits(stego, nbits)
        ber = {}
        for label, att in (("jpeg95", _jpeg(stego, 95)), ("jpeg75", _jpeg(stego, 75)), ("noise1", _noise(stego, 1.0, rng))):
            ber[label] = float(np.mean(m.extract_bits(att, nbits) != coded))
        rows.append({"config": name, "payload_bytes": n, "capacity_bytes": cap // 8, "psnr_db": rep["psnr_db"],
                     "roundtrip_ok": ok, "svd_z": round(det["svd_lattice"]["z"], 1), "pca_z": round(det["pca_lattice"]["z"], 1),
                     "chi_p": round(det["lsb_chi_square"]["p_embedded"], 3), "flagged": det["verdict"],
                     "BER_jpeg95": round(ber["jpeg95"], 4), "BER_jpeg75": round(ber["jpeg75"], 4), "BER_noise_sigma1": round(ber["noise1"], 4)})
        print(f"{name:15s} cap={cap // 8:6d}B psnr={rep['psnr_db']:5.1f} ok={ok} detect={det['verdict']:10s} BER jpeg95={ber['jpeg95']:.3f} noise={ber['noise1']:.3f}")
    with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for r in rows:
        ax[0].scatter(r["capacity_bytes"], r["psnr_db"]); ax[0].annotate(r["config"], (r["capacity_bytes"], r["psnr_db"]), fontsize=7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("capacity (bytes)"); ax[0].set_ylabel("PSNR (dB)"); ax[0].set_title("capacity vs. image quality")
    names = [r["config"] for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x - 0.2, [r["BER_jpeg95"] for r in rows], 0.4, label="JPEG q95")
    ax[1].bar(x + 0.2, [r["BER_noise_sigma1"] for r in rows], 0.4, label="noise sigma=1")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax[1].set_ylabel("raw bit error rate"); ax[1].set_title("robustness (lower is better)"); ax[1].legend()
    ax[2].bar(x, [max(r["svd_z"], r["pca_z"]) for r in rows]); ax[2].axhline(detect.Z_THRESHOLD, color="r", ls="--", label="detection threshold")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax[2].set_ylabel("lattice-detector z"); ax[2].set_title("SVD/PCA lattice detector (LSB is caught by chi-square instead)", fontsize=9); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "benchmark.png"), dpi=110); plt.close(fig)
    print("wrote", os.path.join(out_dir, "benchmark.csv"), "and benchmark.png")
    return rows
