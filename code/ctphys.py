"""
ctphys.py -- image-domain CT physics measurements used in Article 9.

All functions are written for 2-D slices in Hounsfield units (HU) with known pixel
spacing (mm). Conventions follow ICRU Report 87 / AAPM TG-233 for the noise power
spectrum (NPS):

    NPS(fx, fy) = (dx * dy) / (Nx * Ny) * < |DFT{ n(x, y) - fit(x, y) }|^2 >

where n is a zero-mean noise-only ROI, fit is a low-order polynomial detrend, and <.>
is the ensemble average over ROIs. Units: HU^2 * mm^2. With this normalisation the
integral of the NPS over the whole frequency plane equals the pixel variance
(discrete Parseval): sum(NPS) * dfx * dfy = var(n), with dfx = 1/(Nx*dx).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- NPS
def detrend2(roi: np.ndarray, order: int = 2) -> np.ndarray:
    """Subtract a least-squares 2-D polynomial of total degree `order` (0 = mean only)."""
    ny, nx = roi.shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    x = (x - x.mean()) / nx
    y = (y - y.mean()) / ny
    cols = [np.ones_like(x)]
    for d in range(1, order + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    A = np.stack([c.ravel() for c in cols], 1)
    coef, *_ = np.linalg.lstsq(A, roi.ravel().astype(float), rcond=None)
    return roi - (A @ coef).reshape(roi.shape)


def nps2d(rois: np.ndarray, dx: float, dy: float | None = None, order: int = 2) -> np.ndarray:
    """2-D NPS (fftshifted, DC at centre) from a stack of noise ROIs, shape (K, N, N)."""
    dy = dx if dy is None else dy
    rois = np.asarray(rois, dtype=float)
    K, ny, nx = rois.shape
    acc = np.zeros((ny, nx))
    for r in rois:
        acc += np.abs(np.fft.fft2(detrend2(r, order))) ** 2
    return np.fft.fftshift(acc / K) * dx * dy / (nx * ny)


def freq_axes(n: int, d: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(n, d))


def radial_profile(img2d: np.ndarray, d: float, nbins: int | None = None):
    """Azimuthal average of a centred 2-D spectrum. Returns (f_centres [1/mm], profile)."""
    n = img2d.shape[0]
    f = freq_axes(n, d)
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    df = 1.0 / (n * d)
    nbins = nbins or n // 2
    edges = np.arange(nbins + 1) * df
    idx = np.digitize(fr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[ok], weights=img2d.ravel()[ok], minlength=nbins)
    c = np.bincount(idx[ok], minlength=nbins)
    prof = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return (edges[:-1] + edges[1:]) / 2, prof


def nps_summary(nps: np.ndarray, d: float) -> dict:
    """Variance (integral), mean frequency and peak frequency of the radial NPS."""
    n = nps.shape[0]
    df = 1.0 / (n * d)
    var = float(nps.sum() * df * df)
    f, p = radial_profile(nps, d)
    good = np.isfinite(p)
    w = p[good] * f[good]                       # ring weight ~ 2*pi*f*NPS(f)
    fmean = float((f[good] * w).sum() / w.sum())
    fpeak = float(f[good][np.nanargmax(p[good])])
    return {"variance": var, "f_mean": fmean, "f_peak": fpeak}


def nnps_distance(p1: np.ndarray, p2: np.ndarray, f: np.ndarray) -> float:
    """1-Wasserstein distance (in 1/mm) between two radial NPS curves treated as
    distributions of noise power over radial frequency (ring weight 2*pi*f*NPS).
    Scale-invariant: depends on noise TEXTURE, not on noise magnitude."""
    good = np.isfinite(p1) & np.isfinite(p2)
    f = f[good]
    w1 = np.clip(p1[good], 0, None) * f
    w2 = np.clip(p2[good], 0, None) * f
    c1 = np.cumsum(w1) / w1.sum()
    c2 = np.cumsum(w2) / w2.sum()
    df = np.diff(f, prepend=f[0] - (f[1] - f[0]))
    return float(np.sum(np.abs(c1 - c2) * df))


# ------------------------------------------------------------- kernel emulation
def estimate_transfer(src: list[np.ndarray], dst: list[np.ndarray], d: float,
                      nbins: int | None = None, eps: float = 1e-12):
    """Least-squares radial transfer function T(f) such that DFT(dst) ~ T(|f|) * DFT(src).

    For each radial ring: T = sum Re(D * conj(S)) / sum |S|^2 accumulated over all slices.
    Valid when dst and src are reconstructions of the SAME projection data with two
    linear shift-invariant kernels (FBP, same geometry)."""
    n = src[0].shape[0]
    f = freq_axes(n, d)
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    df = 1.0 / (n * d)
    nbins = nbins or int(np.ceil(fr.max() / df)) + 1
    idx = np.minimum((fr / df).astype(int), nbins - 1).ravel()
    num = np.zeros(nbins)
    den = np.zeros(nbins)
    for s, t in zip(src, dst):
        S = np.fft.fftshift(np.fft.fft2(s)).ravel()
        D = np.fft.fftshift(np.fft.fft2(t)).ravel()
        num += np.bincount(idx, weights=np.real(D * np.conj(S)), minlength=nbins)
        den += np.bincount(idx, weights=np.abs(S) ** 2, minlength=nbins)
    T = num / np.maximum(den, eps)
    fc = (np.arange(nbins) + 0.5) * df
    return fc, T


def apply_radial_filter(img: np.ndarray, fc: np.ndarray, T: np.ndarray, d: float) -> np.ndarray:
    """Filter an image by a radial transfer function T(|f|) (linear interpolation)."""
    n = img.shape[0]
    f = np.fft.fftfreq(n, d)
    fx, fy = np.meshgrid(f, f)
    H = np.interp(np.hypot(fx, fy), fc, T, right=T[-1])
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))


# --------------------------------------------------------------- ROI sampling
def sample_rois(noise: np.ndarray, mask: np.ndarray, n: int, size: int, rng) -> np.ndarray:
    """Random square ROIs fully inside `mask` (bool), non-overlap not enforced."""
    H, W = noise.shape
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        y = rng.integers(0, H - size)
        x = rng.integers(0, W - size)
        if mask[y:y + size, x:x + size].all():
            out.append(noise[y:y + size, x:x + size])
    return np.asarray(out)


# ----------------------------------------------- v2: mixed pixel spacing support
def transfer_sums(src: list[np.ndarray], dst: list[np.ndarray], d: float, df: float, nbins: int):
    """Per-ring sums for the least-squares radial transfer, binned on a COMMON PHYSICAL
    frequency grid (bin width df [1/mm]) so images with different pixel spacing d can be
    pooled. Returns (num, den); T = num/den. Bins beyond this image's reach stay 0."""
    n = src[0].shape[0]
    f = np.fft.fftfreq(n, d)
    fx, fy = np.meshgrid(f, f)
    idx = (np.hypot(fx, fy) / df).astype(int).ravel()
    ok = idx < nbins
    num = np.zeros(nbins); den = np.zeros(nbins)
    for s, t in zip(src, dst):
        S = np.fft.fft2(s).ravel(); D = np.fft.fft2(t).ravel()
        num += np.bincount(idx[ok], weights=np.real(D * np.conj(S))[ok], minlength=nbins)
        den += np.bincount(idx[ok], weights=(np.abs(S) ** 2)[ok], minlength=nbins)
    return num, den


def finalize_transfer(num: np.ndarray, den: np.ndarray, df: float, rel_tiny: float = 1e-9):
    """T = num/den on the bin centres; empty bins (den ~ 0) take the last valid value
    (forward fill), so the filter is defined up to the largest radial frequency needed."""
    valid = den > rel_tiny * den.max()
    assert valid[0] or valid[1], "no data in the lowest bins"
    T = np.full(len(num), np.nan)
    T[valid] = num[valid] / den[valid]
    last = T[np.argmax(valid)]
    for i in range(len(T)):
        if np.isfinite(T[i]):
            last = T[i]
        else:
            T[i] = last
    fc = (np.arange(len(num)) + 0.5) * df
    return fc, T, valid


def radial_on_grid(nps: np.ndarray, d: float, fgrid: np.ndarray) -> np.ndarray:
    """Radial NPS profile of a centred 2-D NPS with pixel spacing d, linearly interpolated
    onto a common physical frequency grid fgrid [1/mm] (NaN outside the measured range)."""
    f, p = radial_profile(nps, d)
    g = np.isfinite(p)
    return np.interp(fgrid, f[g], p[g], left=np.nan, right=np.nan)


def transfer_sums_rois(src_rois: np.ndarray, dst_rois: np.ndarray, d: float, df: float, nbins: int):
    """Cross-spectral least-squares transfer from PAIRED ROIs (same positions), Hann-windowed,
    binned on the common physical grid. Used to estimate T only inside tissue, away from
    the -1024 HU clipping of air/lung that makes the whole-image relation non-linear."""
    K, n, _ = np.asarray(src_rois).shape
    w = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftfreq(n, d)
    idx = (np.hypot(*np.meshgrid(f, f)) / df).astype(int).ravel()
    ok = idx < nbins
    num = np.zeros(nbins); den = np.zeros(nbins)
    for s, t in zip(src_rois, dst_rois):
        S = np.fft.fft2((s - s.mean()) * w).ravel(); D = np.fft.fft2((t - t.mean()) * w).ravel()
        num += np.bincount(idx[ok], weights=np.real(D * np.conj(S))[ok], minlength=nbins)
        den += np.bincount(idx[ok], weights=(np.abs(S) ** 2)[ok], minlength=nbins)
    return num, den
