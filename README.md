<div align="center">

# stegoLab

<p align="center">
  <strong>Steganography & Forensics</strong>
</p>

<p align="center">
  <a href="https://github.com/Kowalskiye/stegoLab"><img src="https://img.shields.io/badge/stegoLab-v2.0-0071e3?style=for-the-badge&logo=apple&logoColor=white" alt="stegoLab"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-2997ff?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-3.0-white?style=for-the-badge&logo=flask&logoColor=black" alt="Flask"></a>
  <a href="https://vercel.com"><img src="https://img.shields.io/badge/Vercel-Ready-black?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-30d158?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  Hide <strong>text, files, or complete images</strong> invisibly within the pixels of a cover image using spatial and transform-domain steganography, inspect 24 bit planes with keyboard navigation, and analyze files with blind steganalysis.
</p>

<p align="center">
  <a href="#about-the-project">About</a> &bull;
  <a href="#key-features">Features</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#web-interface">Web Interface</a> &bull;
  <a href="#cli-usage">CLI Usage</a> &bull;
  <a href="#steganography-methods">Methods</a> &bull;
  <a href="#deployment">Deployment</a> &bull;
  <a href="#license">License</a>
</p>

---

</div>

<a id="about-the-project"></a>
## About the Project

**stegoLab** is an in-memory steganography laboratory and forensic analysis tool designed with a minimalist user interface. It combines spatial-domain least significant bit manipulation with transform-domain singular value decomposition quantization index modulation (SVD QIM) and color-laser-printer-style tracking dots (Machine Identification Codes).

Everything runs locally in volatile memory (RAM)—uploaded cover files, secret payloads, and encryption keys are processed in real time and never stored on disk or sent over external networks.

---

<a id="key-features"></a>
## Key Features

- **Multi-Method Embedding:**
  - **Bit-Plane LSB:** Vectorized NumPy bit-plane embedding across selectable color channels (R, G, B) and bit planes (0 LSB through 7 MSB).
  - **SVD QIM:** Quantization Index Modulation on dominant singular values ($\sigma_1$) of $8 \times 8$ pixel blocks with verify-and-retry lattice enforcement.
  - **Dual-Layer Mode:** Embeds structured text into the SVD transform domain and high-capacity secret images into LSB bit planes simultaneously without cross-talk.
- **Universal Multi-Payload Decrypt:** Probes bit planes, channel combinations, and SVD lattice configurations to extract and display all embedded payloads (text, images, and raw files) simultaneously.
- **Interactive Bit-Plane Inspector:** Decomposes images into 24 distinct bit planes and composite RGB layers. Navigate between bit planes seamlessly using the keyboard arrow keys ($\leftarrow$ / $\rightarrow$ or $\uparrow$ / $\downarrow$).
- **Machine Identification Codes (MIC):** Implements an authentic 8-row binary printer tracking dot matrix encoding timestamps, dates, and custom identifier numbers with orientation corner brackets.
- **Blind Steganalysis:** Multi-scale SVD lattice resonance detection $R(\Delta)$, calibrated Chi-Square ($\chi^2$) analysis, and structured header signature detection.

---

<a id="steganography-methods"></a>
## Steganography Methods Comparison

| Technique | Carrier Domain | Typical Capacity (600&times;400 Cover) | Fidelity | Robustness |
| :--- | :--- | :--- | :--- | :--- |
| **Bit-Plane LSB** | Spatial color bit planes (0&ndash;7) | ~90 KB per plane | PSNR &gt; 54 dB | Lossless PNG/BMP only |
| **SVD QIM** | Singular values of $8 \times 8$ blocks | ~1.4 KB per singular value | PSNR 40&ndash;52 dB | Survives noise &amp; mild compression |
| **Dual Layer (LSB + SVD)** | Independent spatial &amp; transform | ~90 KB (LSB) + ~1.4 KB (SVD) | PSNR &gt; 44 dB | Multi-payload separation |
| **MIC Tracking Dots** | Blue channel bit plane 0 grid | Stamp metadata | Invisible | Microscope &amp; bitplane reveal |

---

<a id="installation"></a>
## Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Pip package manager

### Setup

```bash
git clone https://github.com/Kowalskiye/stegoLab.git
cd stegoLab

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Run test suite:
```bash
python -m pytest -q
```

---

<a id="web-interface"></a>
## Web Interface

Start the local server:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

- **encrypt.** &mdash; Select your cover image, payload type (Text, Image, or File), embedding technique, and optional MIC tracking dots.
- **decrypt.** &mdash; Drop any stego image to automatically detect and extract all hidden payloads.
- **analyze.** &mdash; Run statistical Chi-Square and SVD lattice spectrum scans.
- **layers.** &mdash; Inspect individual color channel bit planes. Use the keyboard arrow keys ($\leftarrow$ / $\rightarrow$ or $\uparrow$ / $\downarrow$) to cycle planes 0 to 7.

---

<a id="cli-usage"></a>
## CLI Usage

All capabilities are also available via the command-line interface:

```bash
# Check maximum payload capacity
python -m stegolab capacity --cover cover.png --method svd

# Hide text via SVD QIM
python -m stegolab embed --cover cover.png --out stego.png --method svd --text "Secret meeting at dawn" --delta 24

# Extract payload
python -m stegolab extract --stego stego.png --method svd --delta 24

# Hide an image inside an image via LSB bit plane 0
python -m stegolab embed --cover cover.png --out stego_img.png --method lsb --planes 0 --image secret.png

# Blind steganalysis
python -m stegolab detect --image stego.png --plot-dir plots/
```

---

<a id="deployment"></a>
## Deployment

### Vercel Serverless

The repository includes a ready-to-use [`vercel.json`](vercel.json) configuration:

```bash
# Deploy using Vercel CLI
vercel
```
Or import the repository directly into the [Vercel Dashboard](https://vercel.com/new).

### Docker

```bash
docker build -t stegolab .
docker run -p 5000:5000 stegolab
```

---

<a id="license"></a>
## License

This project is licensed under the MIT License &mdash; see the [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) files for details.
