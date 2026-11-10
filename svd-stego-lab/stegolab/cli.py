"""Command line interface:  python -m stegolab <command> ..."""
import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

from . import core, detect, payload as P, viz
from .blockqim import PCAMethod, SVDMethod
from .imageio import load_image, save_png
from .lsb import LSBMethod


def add_method_args(p, extract=False):
    p.add_argument("--method", choices=["lsb", "svd", "pca"], default="svd", help="embedding method (default: svd)")
    p.add_argument("--planes", type=int, nargs="+", default=[0], help="[lsb] bit planes to use, 0=LSB..7=MSB (default 0)")
    p.add_argument("--channels", type=int, nargs="+", default=None, help="[lsb] colour channels 0=R 1=G 2=B (default all)")
    p.add_argument("--delta", type=float, default=24.0, help="[svd/pca] QIM step. Bigger = more robust but more visible (default 24)")
    p.add_argument("--block", type=int, default=8, help="[svd/pca] block size (default 8)")
    p.add_argument("--k", type=int, default=1, help="[svd] singular values used per block (default 1)")
    p.add_argument("--components", type=int, nargs="+", default=[0, 1, 2], help="[pca] principal components used (default 0 1 2)")
    p.add_argument("--key", default=None, help="secret passphrase that shuffles where bits are stored (optional)")
    p.add_argument("--robust", action="store_true", help="[svd/pca] fixed block positions: survives mild noise/JPEG when combined with --ecc/--repeat")
    p.add_argument("--repeat", type=int, default=1, help="odd repetition factor (majority vote), default 1")
    p.add_argument("--keyfile", default=None, help="[pca] PCA basis keyfile (.npz). Written on embed, required on extract")
    if not extract:
        p.add_argument("--ecc", type=int, default=None, help="Reed-Solomon parity bytes per 255-byte block (even). Default: 0 for lsb/svd(k=1), 32 for pca or svd(k>1)")


def build_method(a, cover=None, extract=False):
    if a.method == "lsb":
        return LSBMethod(planes=a.planes, channels=a.channels, key=a.key)
    if a.method == "svd":
        return SVDMethod(block=a.block, delta=a.delta, k=a.k, key=a.key, robust=a.robust)
    if extract:
        if not a.keyfile:
            sys.exit("PCA extraction needs --keyfile (the keyfile written during embedding)")
        return PCAMethod.from_keyfile(a.keyfile, key=a.key)
    return PCAMethod(block=a.block, delta=a.delta, components=a.components, key=a.key, robust=a.robust)


def default_ecc(a):
    if a.ecc is not None:
        return a.ecc
    return 32 if (a.method == "pca" or (a.method == "svd" and (a.k > 1 or a.robust))) else 0


def cmd_embed(a):
    cover = load_image(a.cover)
    m = build_method(a, cover)
    ecc = default_ecc(a)
    cap = m.capacity_bits(cover) if a.method != "pca" else _pca_capacity(m, cover)
    room = core.max_payload_bytes(cap, a.repeat, ecc)
    if a.text is not None:
        data, kind = a.text.encode("utf-8"), P.TEXT
    elif a.file:
        data, kind = open(a.file, "rb").read(), P.FILE
    elif a.image:
        gray = a.gray or a.method != "lsb"
        data, kind = P.fit_secret_image(Image.open(a.image), room, grayscale=gray), P.IMAGE
        print(f"secret image shrunk to fit: {len(data)} bytes PNG ({'grayscale' if gray else 'colour'})")
    else:
        sys.exit("Give one of --text, --file or --image")
    try:
        stego, rep = core.hide(cover, data, kind, m, repeat=a.repeat, ecc=ecc)
    except core.CapacityError as e:
        sys.exit(f"ERROR: {e}")
    save_png(stego, a.out)
    if a.method == "pca":
        kf = a.keyfile or a.out + ".pcakey.npz"
        m.save_keyfile(kf)
        rep["keyfile"] = kf
        print(f"PCA keyfile written: {kf}  (share it with the receiver; keep it secret)")
    print(json.dumps(rep, indent=2))
    if a.diff:
        viz.diff_figure(cover, stego, a.diff)
        print("difference figure:", a.diff)


def _pca_capacity(m, cover):
    return m.capacity_bits(cover)


def cmd_extract(a):
    stego = load_image(a.stego)
    m = build_method(a, extract=True)
    try:
        kind, data = core.reveal(stego, m, repeat=a.repeat)
    except P.PayloadError as e:
        sys.exit(f"ERROR: {e}")
    if kind == P.TEXT:
        print("Hidden text:\n" + data.decode("utf-8", errors="replace"))
        if a.out:
            open(a.out, "wb").write(data)
    else:
        out = a.out or ("extracted" + P.EXT.get(kind, ".bin"))
        open(out, "wb").write(data)
        print(f"Hidden {'image' if kind == P.IMAGE else 'file'} ({len(data)} bytes) written to {out}")


def cmd_capacity(a):
    cover = load_image(a.cover)
    m = build_method(a, cover)
    ecc = default_ecc(a)
    cap = m.capacity_bits(cover)
    print(f"cover {cover.shape[1]}x{cover.shape[0]}  method={a.method}  raw capacity={cap} bits ({cap // 8} bytes)")
    print(f"max payload with repeat={a.repeat}, ecc={ecc}: {core.max_payload_bytes(cap, a.repeat, ecc)} bytes")


def cmd_detect(a):
    arr = load_image(a.image)
    r = detect.analyze(arr)
    print(json.dumps(r, indent=2, default=float))
    if a.plot_dir:
        os.makedirs(a.plot_dir, exist_ok=True)
        viz.bitplane_grid(arr, os.path.join(a.plot_dir, "bitplanes.png"))
        viz.svd_spectrum_figure(arr, os.path.join(a.plot_dir, "svd_spectrum.png"))
        viz.pca_scree_figure(arr, os.path.join(a.plot_dir, "pca_scree.png"))
        viz.lattice_profile_figure(arr, os.path.join(a.plot_dir, "lattice_profile.png"))
        print("plots written to", a.plot_dir)
    sys.exit(0 if r["verdict"] == "clean" else 3)


def cmd_bitplanes(a):
    viz.bitplane_grid(load_image(a.image), a.out, channel=a.channel)
    print("written", a.out)


def cmd_compare(a):
    c, s = load_image(a.cover), load_image(a.stego)
    if c.shape != s.shape:
        sys.exit("images differ in size")
    viz.diff_figure(c, s, a.out)
    print(f"PSNR = {core.psnr(c, s):.2f} dB; figure written to {a.out}")


def cmd_benchmark(a):
    from .benchmark import run
    run(a.cover, a.out_dir, fill=a.fill)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stegolab", description="Bit-plane / SVD / PCA image steganography lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("embed", help="hide text, a file or an image inside a cover image")
    p.add_argument("--cover", required=True); p.add_argument("--out", required=True, help="output stego image (.png)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text"); g.add_argument("--file"); g.add_argument("--image", help="secret image to hide")
    p.add_argument("--gray", action="store_true", help="store a secret image as grayscale (automatic for svd/pca)")
    p.add_argument("--diff", help="also save a cover/stego/difference figure")
    add_method_args(p); p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("extract", help="recover the hidden payload")
    p.add_argument("--stego", required=True); p.add_argument("--out", help="output file (text is printed)")
    add_method_args(p, extract=True); p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("capacity", help="how much can this cover hold?")
    p.add_argument("--cover", required=True); add_method_args(p); p.set_defaults(fn=cmd_capacity)

    p = sub.add_parser("detect", help="blind steganalysis (chi-square + SVD/PCA lattice detectors)")
    p.add_argument("--image", required=True); p.add_argument("--plot-dir"); p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("bitplanes", help="visualise the 8 bit planes of one channel")
    p.add_argument("--image", required=True); p.add_argument("--out", default="bitplanes.png"); p.add_argument("--channel", type=int, default=0)
    p.set_defaults(fn=cmd_bitplanes)

    p = sub.add_parser("compare", help="PSNR and amplified difference between cover and stego")
    p.add_argument("--cover", required=True); p.add_argument("--stego", required=True); p.add_argument("--out", default="diff.png")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("benchmark", help="sweep methods/deltas: PSNR, capacity, detectability, JPEG robustness")
    p.add_argument("--cover", required=True); p.add_argument("--out-dir", default="benchmark_out"); p.add_argument("--fill", type=float, default=0.5)
    p.set_defaults(fn=cmd_benchmark)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
