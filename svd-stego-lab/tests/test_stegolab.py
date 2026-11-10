import io
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stegolab import core, detect, payload as P  # noqa: E402
from stegolab.blockqim import PCAMethod, SVDMethod  # noqa: E402
from stegolab.lsb import LSBMethod  # noqa: E402

COVER = np.asarray(Image.open(os.path.join(ROOT, "samples", "cover_coffee.png")).convert("RGB"), dtype=np.uint8)
ASTRO = np.asarray(Image.open(os.path.join(ROOT, "samples", "cover_astronaut.png")).convert("RGB"), dtype=np.uint8)
MSG = b"Rendezvous at the north bridge at 0600. " * 5


def methods():
    return [
        ("lsb0", lambda: LSBMethod([0]), 0),
        ("lsb_planes_keyed", lambda: LSBMethod([0, 1, 2], key="k"), 0),
        ("svd", lambda: SVDMethod(8, 24), 0),
        ("svd_k2_keyed", lambda: SVDMethod(8, 20, k=2, key="s3cret"), 32),
        ("pca", lambda: PCAMethod(8, 24, (0, 1, 2)), 32),
        ("pca_keyed", lambda: PCAMethod(8, 24, (0, 1), key="s3cret"), 32),
    ]


@pytest.mark.parametrize("name,mk,ecc", methods())
def test_text_roundtrip(name, mk, ecc):
    m = mk()
    stego, rep = core.hide(COVER, MSG, P.TEXT, m, ecc=ecc)
    kind, data = core.reveal(stego, m)
    assert kind == P.TEXT and data == MSG
    assert rep["psnr_db"] > 30


@pytest.mark.parametrize("name,mk,ecc", methods())
def test_survives_png_roundtrip(name, mk, ecc):
    m = mk()
    stego, _ = core.hide(COVER, MSG, P.FILE, m, ecc=ecc)
    buf = io.BytesIO()
    Image.fromarray(stego).save(buf, format="PNG")
    buf.seek(0)
    again = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    assert core.reveal(again, m)[1] == MSG


def test_image_payload_roundtrip_svd():
    secret = Image.open(os.path.join(ROOT, "samples", "secret_logo.png"))
    m = SVDMethod(8, 24)
    room = core.max_payload_bytes(m.capacity_bits(COVER))
    data = P.fit_secret_image(secret, room, grayscale=True)
    stego, _ = core.hide(COVER, data, P.IMAGE, m)
    kind, out = core.reveal(stego, m)
    assert kind == P.IMAGE and Image.open(io.BytesIO(out)).size[0] > 8


def test_wrong_key_fails():
    stego, _ = core.hide(COVER, MSG, P.TEXT, SVDMethod(8, 24, key="right"))
    with pytest.raises(P.PayloadError):
        core.reveal(stego, SVDMethod(8, 24, key="wrong"))


def test_wrong_delta_fails():
    stego, _ = core.hide(COVER, MSG, P.TEXT, SVDMethod(8, 24))
    with pytest.raises(P.PayloadError):
        core.reveal(stego, SVDMethod(8, 16))


def test_clean_image_has_no_payload():
    for m in (LSBMethod([0]), SVDMethod(8, 24)):
        with pytest.raises(P.PayloadError):
            core.reveal(COVER, m)


def test_capacity_error_message():
    with pytest.raises(core.CapacityError):
        core.hide(COVER, os.urandom(5000), P.FILE, SVDMethod(8, 24))


def test_repetition_and_ecc_repair_damage():
    m = LSBMethod([0])
    stego, _ = core.hide(COVER, MSG, P.TEXT, m, repeat=3, ecc=20)
    damaged = stego.copy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 2000, 30)
    damaged.reshape(-1)[idx] ^= 1  # flip some LSBs inside the used region
    assert core.reveal(damaged, m, repeat=3)[1] == MSG


def test_lsb_plane_only_changes_that_plane():
    m = LSBMethod([2])
    stego, _ = core.hide(COVER, MSG, P.TEXT, m)
    diff = stego.astype(int) ^ COVER.astype(int)
    assert diff.max() <= 4 and (diff & ~4).sum() == 0


def test_svd_distortion_bounded():
    stego, rep = core.hide(COVER, MSG * 6, P.TEXT, SVDMethod(8, 24))
    assert rep["unverified_blocks"] == 0
    assert np.abs(stego.astype(int) - COVER.astype(int)).max() <= 24


def test_astronaut_saturated_regions_with_ecc():
    m = PCAMethod(8, 24, (0, 1, 2))
    stego, _ = core.hide(ASTRO, MSG * 10, P.TEXT, m, ecc=40)
    assert core.reveal(stego, m)[1] == MSG * 10


def test_pca_keyfile_roundtrip(tmp_path):
    m = PCAMethod(8, 24, (0, 1, 2))
    stego, _ = core.hide(COVER, MSG, P.TEXT, m, ecc=16)
    kf = tmp_path / "k.npz"
    m.save_keyfile(kf)
    m2 = PCAMethod.from_keyfile(kf)
    assert core.reveal(stego, m2)[1] == MSG


def test_robust_mode_survives_noise():
    m = SVDMethod(8, 48, robust=True)
    stego, _ = core.hide(ASTRO, MSG, P.TEXT, m, ecc=32)
    noisy = np.clip(np.rint(stego + np.random.default_rng(1).normal(0, 1, stego.shape)), 0, 255).astype(np.uint8)
    assert core.reveal(noisy, m)[1] == MSG


def test_detector_clean_vs_stego():
    assert detect.analyze(COVER)["verdict"] == "clean"
    m = SVDMethod(8, 24)
    n = core.max_payload_bytes(m.capacity_bits(COVER)) // 2
    stego, _ = core.hide(COVER, os.urandom(n), P.FILE, m)
    r = detect.analyze(stego)
    assert r["flags"]["svd_suspected"] and abs(r["svd_lattice"]["best_delta"] - 24) < 0.6
    lsb, _ = core.hide(COVER, os.urandom(60000), P.FILE, LSBMethod([0]))
    assert detect.analyze(lsb)["flags"]["lsb_suspected"]


def test_compatible_with_original_RobinDavid_LSBSteg():
    """Data hidden by the ORIGINAL repo code (plane 0, raster, 64-bit length) is readable by our LSB engine."""
    pytest.importorskip("cv2")
    pytest.importorskip("docopt")
    from vendor.LSBSteg_original import LSBSteg
    img = COVER[:, :, ::-1].copy()  # BGR like cv2.imread
    secret = b"original repo format"
    out = LSBSteg(img.copy()).encode_binary(secret)
    m = LSBMethod([0])
    length = int("".join(map(str, m.extract_bits(out, 64))), 2)
    bits = m.extract_bits(out, 64 + 8 * length)[64:]
    assert length == len(secret) and P.bits_to_bytes(bits) == secret


def test_cli_end_to_end(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = tmp_path / "s.png"
    base = [sys.executable, "-m", "stegolab"]
    cover = os.path.join(ROOT, "samples", "cover_coffee.png")
    r = subprocess.run(base + ["embed", "--cover", cover, "--out", str(out), "--method", "pca", "--text", "cli works"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(base + ["extract", "--stego", str(out), "--method", "pca", "--keyfile", str(out) + ".pcakey.npz"], env=env, capture_output=True, text=True)
    assert "cli works" in r.stdout


def test_web_api():
    from app import app
    c = app.test_client()
    buf = io.BytesIO(); Image.fromarray(COVER).save(buf, format="PNG"); buf.seek(0)
    r = c.post("/api/embed", data={"cover": (buf, "c.png"), "method": "svd", "payload_type": "text", "text": "hi"}, content_type="multipart/form-data")
    assert r.status_code == 200
    import base64
    st = base64.b64decode(r.get_json()["stego_png"])
    r = c.post("/api/extract", data={"stego": (io.BytesIO(st), "s.png"), "method": "svd"}, content_type="multipart/form-data")
    assert r.get_json()["text"] == "hi"


def test_mic_encoding_and_detection():
    from stegolab import mic
    matrix = mic.encode_mic_matrix(day=10, month=11, year=26, hour=13, minute=5, custom_number=42)
    assert matrix["metadata"]["day"] == 10
    assert matrix["metadata"]["month"] == 11
    assert matrix["metadata"]["year"] == 26
    assert matrix["metadata"]["time_str"] == "13:05"
    assert matrix["metadata"]["custom_number"] == 42

    # Embed in image and detect
    arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    stamped = mic.embed_mic_in_pixels(arr, 10, 11, 26, 13, 5, 42)
    detected = mic.detect_mic_in_pixels(stamped)
    assert detected is not None
    assert detected["metadata"]["day"] == 10
    assert detected["metadata"]["month"] == 11
    assert detected["metadata"]["year"] == 26
    assert detected["metadata"]["time_str"] == "13:05"
    assert detected["metadata"]["custom_number"] == 42


def test_inspector_and_auto_decode():
    from stegolab import inspector
    # 1. Bitplane extraction
    bp = inspector.get_bitplane(COVER, channel=0, plane=0)
    assert bp.shape == COVER.shape[:2] and bp.dtype == np.uint8

    # 2. Composite plane
    comp = inspector.get_composite_plane(COVER, plane=0)
    assert comp.shape == COVER.shape and comp.dtype == np.uint8

    # 3. Auto decode on LSB
    stego_lsb, _ = core.hide(COVER, b"LSB Auto Test", P.TEXT, LSBMethod([0]))
    res = inspector.auto_decode_all_modes(stego_lsb)
    assert res["summary"]["payload_found"] is True
    assert "LSB" in res["summary"]["found_modes"]
    assert res["lsb"]["payload"]["text"] == "LSB Auto Test"
