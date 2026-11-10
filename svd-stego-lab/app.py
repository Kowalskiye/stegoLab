"""Web UI for SVD Stego Lab (StegOnline Suite). Run: python app.py -> http://127.0.0.1:5000
Everything runs locally; uploaded images are processed in memory and never stored.
"""
import base64
import io
import os
import tempfile
from types import SimpleNamespace

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image

from stegolab import core, detect, inspector, mic, payload as P, viz
from stegolab.blockqim import SVDMethod
from stegolab.cli import build_method, default_ecc
from stegolab.lsb import LSBMethod

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def _img(fileobj):
    return np.asarray(Image.open(fileobj).convert("RGB"), dtype=np.uint8).copy()


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ints(s, default):
    s = (s or "").replace(",", " ").split()
    return [int(x) for x in s] if s else list(default) if default is not None else None


def _args(form):
    ecc = form.get("ecc", "").strip()
    channels_val = form.get("channels", "").strip()
    channels = _ints(channels_val, None) if channels_val else None

    return SimpleNamespace(
        method=form.get("method", "svd"),
        planes=_ints(form.get("planes"), [0]),
        channels=channels,
        delta=float(form.get("delta") or 24),
        block=int(form.get("block") or 8),
        k=int(form.get("k") or 1),
        key=form.get("key") or None,
        repeat=int(form.get("repeat") or 1),
        robust=form.get("robust") in ("on", "true", "1"),
        ecc=int(ecc) if ecc else None,
    )


def _err(msg, code=400):
    return jsonify({"error": str(msg)}), code


@app.get("/")
@app.get("/about")
def index():
    return render_template("index.html")


@app.get("/api/sample/<path:filename>")
def get_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.post("/api/mic/preview")
def api_mic_preview():
    try:
        data = request.get_json(silent=True) or request.form
        day = int(data.get("day") or 1)
        month = int(data.get("month") or 1)
        year = int(data.get("year") or 26)
        hour = int(data.get("hour") or 0)
        minute = int(data.get("minute") or 0)
        custom_number = int(data.get("custom_number") or 0)
        res = mic.encode_mic_matrix(day, month, year, hour, minute, custom_number)
        return jsonify(res)
    except Exception as e:
        return _err(f"MIC preview error: {e}")


# -------------------------------------------------------------
# 1. AUTO-DECODE (RUNS THROUGH ALL 3 MODES + RAW STRINGS)
# -------------------------------------------------------------
@app.post("/api/auto_decode")
def api_auto_decode():
    try:
        if "stego" not in request.files:
            return _err("No stego image uploaded")
        stego_arr = _img(request.files["stego"])
        key = request.form.get("key") or None

        kf_file = request.files.get("keyfile")
        kf_bytes = kf_file.read() if kf_file and kf_file.filename else None

        report = inspector.auto_decode_all_modes(stego_arr, key=key, keyfile_bytes=kf_bytes)
        return jsonify(report)
    except Exception as e:
        return _err(f"Auto-decode error: {e}")


# -------------------------------------------------------------
# 2. EMBED / ENCRYPT (SUPPORTS TEXT, IMAGE, FILE + RGB & BITPLANES)
# -------------------------------------------------------------
@app.post("/api/embed")
def api_embed():
    try:
        a = _args(request.form)
        if "cover" not in request.files:
            return _err("No cover image uploaded")
        cover = _img(request.files["cover"])

        if a.method == "both":
            # Dual Layer Embedding (Both SVD and LSB)
            text_val = request.form.get("text", "").strip()
            has_file = "payload_file" in request.files and bool(request.files["payload_file"].filename)

            svd_m = SVDMethod(block=a.block, delta=a.delta, k=a.k, key=a.key, robust=a.robust)
            lsb_m = LSBMethod(planes=a.planes or [0], channels=a.channels, key=a.key)
            ecc_svd = default_ecc(SimpleNamespace(method="svd", k=a.k, robust=a.robust, ecc=a.ecc))
            ecc_lsb = 0

            stego = cover.copy()
            rep_combined = {"method": "both", "layers": {}}
            room = lsb_m.capacity_bits(cover) // 8

            if text_val and has_file:
                # 1. Embed Secret Text into SVD layer FIRST (robust transform domain)
                text_data = text_val.encode("utf-8")
                stego, rep_svd = core.hide(stego, text_data, P.TEXT, svd_m, repeat=a.repeat, ecc=ecc_svd)
                rep_combined["layers"]["svd"] = rep_svd

                # 2. Embed Secret Image/File into LSB layer SECOND (large bit-plane capacity)
                ptype = request.form.get("payload_type", "image")
                file_obj = request.files["payload_file"]
                room_lsb = core.max_payload_bytes(lsb_m.capacity_bits(cover), a.repeat, ecc_lsb)
                if ptype == "image":
                    file_data = P.fit_secret_image(Image.open(file_obj), room_lsb, grayscale=False)
                    kind = P.IMAGE
                else:
                    file_data = file_obj.read()
                    kind = P.FILE
                stego, rep_lsb = core.hide(stego, file_data, kind, lsb_m, repeat=a.repeat, ecc=ecc_lsb)
                rep_combined["layers"]["lsb"] = rep_lsb

            elif has_file:
                # Secret Image/File: Embed into high-capacity LSB layer
                ptype = request.form.get("payload_type", "image")
                file_obj = request.files["payload_file"]
                room_lsb = core.max_payload_bytes(lsb_m.capacity_bits(cover), a.repeat, ecc_lsb)
                if ptype == "image":
                    file_data = P.fit_secret_image(Image.open(file_obj), room_lsb, grayscale=False)
                    kind = P.IMAGE
                else:
                    file_data = file_obj.read()
                    kind = P.FILE
                stego, rep_lsb = core.hide(stego, file_data, kind, lsb_m, repeat=a.repeat, ecc=ecc_lsb)
                rep_combined["layers"]["lsb"] = rep_lsb

            else:
                # Text message provided: embed redundantly into BOTH SVD and LSB layers!
                if not text_val:
                    text_val = "Secret dual-layer steganography message"
                text_bytes = text_val.encode("utf-8")
                # 1. Embed in SVD
                stego, rep_svd = core.hide(stego, text_bytes, P.TEXT, svd_m, repeat=a.repeat, ecc=ecc_svd)
                rep_combined["layers"]["svd"] = rep_svd
                # 2. Embed in LSB
                stego, rep_lsb = core.hide(stego, text_bytes, P.TEXT, lsb_m, repeat=a.repeat, ecc=ecc_lsb)
                rep_combined["layers"]["lsb"] = rep_lsb

            rep_combined["psnr_db"] = round(core.psnr(cover, stego), 2)
            rep = rep_combined

        else:
            # Single method (LSB or SVD)
            # When SVD is selected with MIC enabled, default to robust=True and Reed-Solomon ECC (16 bytes)
            # to protect against bit perturbations from tracking dots and saturated block borders.
            enable_mic_requested = request.form.get("enable_mic") in ("on", "true", "1")
            if a.method == "svd" and enable_mic_requested and not a.robust:
                a.robust = True
                if a.ecc is None or a.ecc == 0:
                    a.ecc = 16

            m = build_method(a, cover)
            ecc = default_ecc(a)
            room = core.max_payload_bytes(m.capacity_bits(cover), a.repeat, ecc)

            ptype = request.form.get("payload_type", "text")
            if ptype == "text":
                text_val = request.form.get("text", "")
                data, kind = text_val.encode("utf-8"), P.TEXT
            elif ptype == "file":
                if "payload_file" not in request.files:
                    return _err("No file uploaded")
                data, kind = request.files["payload_file"].read(), P.FILE
            else:
                if "payload_file" not in request.files:
                    return _err("No secret image uploaded")
                gray = request.form.get("gray") in ("on", "true", "1") or a.method != "lsb"
                data = P.fit_secret_image(Image.open(request.files["payload_file"]), room, grayscale=gray)
                kind = P.IMAGE

            stego, rep = core.hide(cover, data, kind, m, repeat=a.repeat, ecc=ecc)

        # Machine Identification Code (MIC / Tracking Dots) stamping
        if request.form.get("enable_mic") in ("on", "true", "1"):
            mic_day = int(request.form.get("mic_day") or 1)
            mic_month = int(request.form.get("mic_month") or 1)
            mic_year = int(request.form.get("mic_year") or 26)
            mic_hour = int(request.form.get("mic_hour") or 0)
            mic_minute = int(request.form.get("mic_minute") or 0)
            mic_custom = int(request.form.get("mic_custom") or 0)
            stego = mic.embed_mic_in_pixels(stego, mic_day, mic_month, mic_year, mic_hour, mic_minute, mic_custom)
            rep["mic"] = mic.encode_mic_matrix(mic_day, mic_month, mic_year, mic_hour, mic_minute, mic_custom)["metadata"]

        out = {
            "report": rep,
            "stego_png": _png_b64(stego),
            "max_payload_bytes": room,
        }
        return jsonify(out)
    except (core.CapacityError, P.PayloadError, ValueError, KeyError, SystemExit) as e:
        return _err(e)
    except Exception as e:
        return _err(f"Embed error: {e}")


# -------------------------------------------------------------
# 3. MANUAL / CUSTOM EXTRACT
# -------------------------------------------------------------
@app.post("/api/extract")
def api_extract():
    try:
        a = _args(request.form)
        if "stego" not in request.files:
            return _err("No stego image uploaded")
        stego = _img(request.files["stego"])

        if a.method == "both":
            results = {"method": "both", "payloads": []}
            try:
                m_lsb = LSBMethod(planes=a.planes or [0], channels=a.channels, key=a.key)
                kind_l, data_l = core.reveal(stego, m_lsb, repeat=a.repeat)
                p_l = {"method": "LSB", "kind": {P.TEXT: "text", P.FILE: "file", P.IMAGE: "image"}.get(kind_l, "file"), "bytes": len(data_l)}
                if kind_l == P.TEXT:
                    p_l["text"] = data_l.decode("utf-8", errors="replace")
                else:
                    p_l["data_b64"] = base64.b64encode(data_l).decode("ascii")
                results["payloads"].append(p_l)
            except Exception:
                pass

            try:
                m_svd = SVDMethod(block=a.block, delta=a.delta, k=a.k, key=a.key, robust=a.robust)
                kind_s, data_s = core.reveal(stego, m_svd, repeat=a.repeat)
                p_s = {"method": "SVD", "kind": {P.TEXT: "text", P.FILE: "file", P.IMAGE: "image"}.get(kind_s, "file"), "bytes": len(data_s)}
                if kind_s == P.TEXT:
                    p_s["text"] = data_s.decode("utf-8", errors="replace")
                else:
                    p_s["data_b64"] = base64.b64encode(data_s).decode("ascii")
                results["payloads"].append(p_s)
            except Exception:
                pass

            mic_detected = mic.detect_mic_in_pixels(stego)
            if mic_detected:
                results["mic"] = mic_detected
            return jsonify(results)

        m = build_method(a, extract=True)
        kind, data = core.reveal(stego, m, repeat=a.repeat)
        res = {
            "kind": {P.TEXT: "text", P.FILE: "file", P.IMAGE: "image"}.get(kind, "file"),
            "bytes": len(data),
        }
        if kind == P.TEXT:
            res["text"] = data.decode("utf-8", errors="replace")
        else:
            res["data_b64"] = base64.b64encode(data).decode("ascii")

        # Check for MIC tracking dots in stego image
        mic_detected = mic.detect_mic_in_pixels(stego)
        if mic_detected:
            res["mic"] = mic_detected

        return jsonify(res)
    except (P.PayloadError, ValueError, KeyError, SystemExit, EOFError, OSError) as e:
        return _err(e)
    except Exception as e:
        return _err(f"Extraction error: {e}")


# -------------------------------------------------------------
# 4. DOWNLOAD ALL LAYERS AS ZIP
# -------------------------------------------------------------
@app.post("/api/layers/zip")
def api_layers_zip():
    try:
        if "image" not in request.files:
            return _err("No image uploaded")
        arr = _img(request.files["image"])
        zip_bytes = inspector.generate_all_layers_zip(arr)
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name="all_layers_and_bitplanes.zip",
        )
    except Exception as e:
        return _err(f"Failed to generate layer zip: {e}")


# -------------------------------------------------------------
# 5. DETECT & FORENSICS
# -------------------------------------------------------------
@app.post("/api/detect")
def api_detect():
    try:
        if "image" not in request.files:
            return _err("No image uploaded")
        arr = _img(request.files["image"])
        r = detect.analyze(arr)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lp.png")
            viz.lattice_profile_figure(arr, p)
            prof_b64 = base64.b64encode(open(p, "rb").read()).decode("ascii")
        return jsonify({**r, "profile_png": prof_b64})
    except Exception as e:
        return _err(f"Detect error: {e}")


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
