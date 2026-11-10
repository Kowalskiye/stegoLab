"""Plots and visual forensics (matplotlib, headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .blockqim import eligible_blocks, split_blocks  # noqa: E402


def bitplane_grid(arr, path, channel=0):
    """8 panels: bit plane 7 (MSB) ... 0 (LSB) of one colour channel."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, p in zip(axes.ravel(), range(7, -1, -1)):
        ax.imshow(((arr[:, :, channel] >> p) & 1) * 255, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"bit plane {p}" + (" (LSB)" if p == 0 else " (MSB)" if p == 7 else ""), fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Bit planes of channel {channel} (0=R,1=G,2=B)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def diff_figure(cover, stego, path, gain=None):
    d = np.abs(cover.astype(np.int16) - stego.astype(np.int16)).max(axis=2)
    gain = gain or max(1, int(255 / max(d.max(), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cover); axes[0].set_title("cover")
    axes[1].imshow(stego); axes[1].set_title("stego")
    axes[2].imshow(np.clip(d * gain, 0, 255), cmap="inferno"); axes[2].set_title(f"|difference| x{gain} (max change {d.max()})")
    for a in axes:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def svd_spectrum_figure(arr, path, nblocks=3000):
    """Average singular-value spectrum + sigma_1 histogram (lattice peaks show up as comb)."""
    blocks = split_blocks(arr, 8)
    blocks = blocks[eligible_blocks(blocks)].astype(np.float64)
    s = np.linalg.svd(blocks, compute_uv=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].semilogy(np.arange(1, 9), s.mean(axis=0), "o-")
    ax[0].set_title("mean singular values of 8x8 blocks"); ax[0].set_xlabel("index i"); ax[0].set_ylabel("sigma_i")
    sel = s[:nblocks, 0]
    ax[1].hist(sel, bins=np.arange(sel.min(), min(sel.max(), sel.min() + 400), 1.0), color="#444")
    ax[1].set_title("sigma_1 histogram (zoom): comb pattern => QIM embedding"); ax[1].set_xlabel("sigma_1")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def pca_scree_figure(arr, path):
    blocks = split_blocks(arr, 8)
    X = blocks[eligible_blocks(blocks)].reshape(-1, 64).astype(np.float64)
    Xc = X - X.mean(axis=0)
    vals = np.sort(np.linalg.eigvalsh((Xc.T @ Xc) / max(len(X) - 1, 1)))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(np.arange(1, 65), vals, "o-", ms=3); ax[0].set_title("PCA scree plot (8x8 blocks)"); ax[0].set_xlabel("component"); ax[0].set_ylabel("eigenvalue")
    ax[1].plot(np.arange(1, 65), np.cumsum(vals) / vals.sum(), "o-", ms=3); ax[1].set_title("cumulative explained variance"); ax[1].set_xlabel("components")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def lattice_profile_figure(arr, path, deltas=None):
    from .detect import DEFAULT_DELTAS
    deltas = DEFAULT_DELTAS if deltas is None else deltas
    blocks = split_blocks(arr, 8)
    s = np.linalg.svd(blocks[eligible_blocks(blocks)].astype(np.float64), compute_uv=False)[:, 0]
    R = np.array([np.abs(np.exp(2j * np.pi * s / d).mean()) for d in deltas])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(deltas, R); ax.axhline(np.median(R), color="gray", ls="--", lw=1)
    i = int(np.argmax(R)); ax.annotate(f"max at delta={deltas[i]:.1f}", (deltas[i], R[i]), textcoords="offset points", xytext=(8, -4))
    ax.set_xlabel("candidate step delta"); ax.set_ylabel("R(delta)"); ax.set_title("SVD lattice detector: peak = QIM step")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
