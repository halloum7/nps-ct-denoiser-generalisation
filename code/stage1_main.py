
# ============================ STAGE 1 : audit + NPS + kernel-emulation test ============================
import os, re, glob, json, time
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pydicom
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working/stage1"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
cands = [p for p in glob.glob("/kaggle/input/**/Original Data", recursive=True)]
assert len(cands) >= 1, "Original Data folder not found"
ROOT = sorted(cands, key=len)[0]
print("ROOT", ROOT)

# ---------------------------------------------------------------- 1. inventory from paths
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.IMA"), recursive=True))
def parse(p):
    return dict(path=p,
                dose="FD" if "Full Dose" in p else ("QD" if "Quarter Dose" in p else "?"),
                thick="1mm" if "1mm" in p else ("3mm" if "3mm" in p else "?"),
                kernel="D45" if "D45" in p else ("B30" if "B30" in p else "?"),
                patient=(re.search(r"/(L\d{3})/", p) or re.search(r"(L\d{3})", p)).group(1))
inv = pd.DataFrame([parse(p) for p in files])
assert (inv[["dose", "thick", "kernel"]] != "?").all().all(), "unparsed path"
counts = inv.pivot_table(index="patient", columns=["thick", "kernel", "dose"], values="path", aggfunc="count")
counts.to_csv(f"{OUT}/counts.csv")
print("n files", len(inv)); print(counts)

# ---------------------------------------------------------------- 2. headers (no pixels)
FIELDS = ["ConvolutionKernel", "SliceThickness", "KVP", "XRayTubeCurrent", "Exposure", "CTDIvol",
          "RescaleSlope", "RescaleIntercept", "Rows", "Columns", "InstanceNumber", "Manufacturer",
          "ManufacturerModelName"]
def hdr(p):
    d = pydicom.dcmread(p, stop_before_pixels=True)
    r = {k: (str(getattr(d, k)) if hasattr(d, k) else None) for k in FIELDS}
    r["z"] = float(d.ImagePositionPatient[2]) if hasattr(d, "ImagePositionPatient") else np.nan
    r["px"] = float(d.PixelSpacing[0]) if hasattr(d, "PixelSpacing") else np.nan
    return r
with ThreadPoolExecutor(32) as ex:
    H = list(ex.map(hdr, inv.path))
inv = pd.concat([inv, pd.DataFrame(H)], axis=1)
inv.drop(columns=["path"]).to_csv(f"{OUT}/headers.csv", index=False)
hsum = {}
for c in ["ConvolutionKernel", "SliceThickness", "KVP", "Manufacturer", "ManufacturerModelName", "Rows", "Columns"]:
    hsum[c] = inv.groupby(["thick", "kernel", "dose"])[c].agg(lambda s: sorted(set(map(str, s)))[:6]).to_dict()
for c in ["XRayTubeCurrent", "Exposure", "CTDIvol", "px"]:
    v = pd.to_numeric(inv[c], errors="coerce")
    hsum[c] = inv.assign(v=v).groupby(["dose"])["v"].describe()[["count", "mean", "min", "max"]].round(3).to_dict()
print("header summary:", json.dumps({k: {str(a): b for a, b in v.items()} for k, v in hsum.items()}, default=str)[:3000])

# ---------------------------------------------------------------- 3. pairing FD<->QD and B30<->D45 by z
inv["zr"] = inv.z.round(2)
pair_rows = []
for (pt, th), g in inv.groupby(["patient", "thick"]):
    sets = {(k, d): set(g[(g.kernel == k) & (g.dose == d)].zr) for k in ["B30", "D45"] for d in ["FD", "QD"]}
    allk = set.intersection(*sets.values())
    pair_rows.append(dict(patient=pt, thick=th,
                          **{f"n_{k}_{d}": len(s) for (k, d), s in sets.items()},
                          n_all4=len(allk),
                          dup_z=int(g.duplicated(["kernel", "dose", "zr"]).sum())))
pairs = pd.DataFrame(pair_rows); pairs.to_csv(f"{OUT}/pairing.csv", index=False); print(pairs)

# ---------------------------------------------------------------- 4. slice loader
lut = {(r.patient, r.thick, r.kernel, r.dose, r.zr): r.path for r in inv.itertuples()}
def load_hu(pt, th, k, d, z):
    ds = pydicom.dcmread(lut[(pt, th, k, d, z)])
    return ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

def body_mask(fd_hu, erode_px=12):
    m = fd_hu > -300
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_fill_holes(m)
    return ndimage.binary_erosion(m, iterations=erode_px)


import zlib
def seed_of(*a):
    return zlib.crc32(repr(a).encode())

def analyze_thickness(store, px, th, TRAIN_T, TEST_T, OUT, K_ROI=64, N_ROI=20, make_fig=True):
    """store: {(pt, z): {(kernel, dose): HU image}}. Returns dict of results."""
    res = {}
    def rois_of(img, m, *key):
        return sample_rois(img, m, N_ROI, K_ROI, np.random.default_rng(seed_of(*key)))
    masks = {key: body_mask(im[("B30", "FD")]) for key, im in store.items()}
    # ---- NPS of the difference image QD-FD, per kernel (all patients)
    radial, nps2 = {}, {}
    for k in ["B30", "D45"]:
        R = [rois_of(im[(k, "QD")] - im[(k, "FD")], masks[key], key, k) for key, im in store.items()]
        R = np.concatenate([r for r in R if len(r)])
        P = nps2d(R, px, order=2); f, p = radial_profile(P, px)
        radial[k] = p; nps2[k] = P
        res[f"{th}_{k}_nps"] = {**nps_summary(P, px), "n_rois": int(len(R))}
    radial["f"] = f
    pd.DataFrame(radial).to_csv(f"{OUT}/nps_radial_{th}.csv", index=False)
    res[f"{th}_nps_W1_B30_vs_D45"] = nnps_distance(radial["B30"], radial["D45"], f)
    # ---- kernel emulation D45 -> B30: transfer learned on TRAIN_T, tested on TEST_T (FD images)
    tr = [key for key in store if key[0] in TRAIN_T]; te = [key for key in store if key[0] in TEST_T]
    fc, T = estimate_transfer([store[k][("D45", "FD")] for k in tr], [store[k][("B30", "FD")] for k in tr], px)
    pd.DataFrame({"f": fc, "T_D45_to_B30": T}).to_csv(f"{OUT}/transfer_{th}.csv", index=False)
    rows, Re, Rr, Rd = [], [], [], []
    for key in te:
        im, m = store[key], masks[key]
        sig_ref = float(np.std((im[("B30", "QD")] - im[("B30", "FD")])[m]))
        for d in ["FD", "QD"]:
            emu = apply_radial_filter(im[("D45", d)], fc, T, px)
            e_emu = float(np.sqrt(np.mean((emu - im[("B30", d)])[m] ** 2)))
            e_raw = float(np.sqrt(np.mean((im[("D45", d)] - im[("B30", d)])[m] ** 2)))
            rows.append(dict(patient=key[0], z=key[1], dose=d, rmse_emulated=e_emu, rmse_unfiltered=e_raw,
                             sigma_diff_B30=sig_ref, ratio_emulated=e_emu / sig_ref, ratio_unfiltered=e_raw / sig_ref))
        nD = im[("D45", "QD")] - im[("D45", "FD")]
        Re.append(rois_of(apply_radial_filter(nD, fc, T, px), m, key, "emu"))
        Rr.append(rois_of(im[("B30", "QD")] - im[("B30", "FD")], m, key, "emu"))   # same ROI positions
        Rd.append(rois_of(nD, m, key, "emu"))
    em = pd.DataFrame(rows); em.to_csv(f"{OUT}/emulation_{th}.csv", index=False)
    cat = lambda L: np.concatenate([r for r in L if len(r)])
    Pe, Pr, Pd = nps2d(cat(Re), px), nps2d(cat(Rr), px), nps2d(cat(Rd), px)
    _, pe = radial_profile(Pe, px); _, pr = radial_profile(Pr, px); _, pd45 = radial_profile(Pd, px)
    res[f"{th}_emulation"] = {
        "median_ratio_emulated_FD": float(em[em.dose == "FD"].ratio_emulated.median()),
        "median_ratio_unfiltered_FD": float(em[em.dose == "FD"].ratio_unfiltered.median()),
        "median_ratio_emulated_QD": float(em[em.dose == "QD"].ratio_emulated.median()),
        "median_ratio_unfiltered_QD": float(em[em.dose == "QD"].ratio_unfiltered.median()),
        "W1_emulatedNoise_vs_realB30": nnps_distance(pe, pr, f),
        "W1_D45noise_vs_realB30": nnps_distance(pd45, pr, f),
        "var_emulated_noise": nps_summary(Pe, px)["variance"],
        "var_real_B30_noise": nps_summary(Pr, px)["variance"],
        "n_test_slices": int(len(te))}
    if make_fig:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for k, c in [("B30", "C0"), ("D45", "C1")]:
            ax[0].plot(f, radial[k] / np.nansum(radial[k] * f), c, label=f"{k} (QD-FD)")
        ax[0].set_xlabel("f (1/mm)"); ax[0].set_ylabel("normalised radial NPS"); ax[0].legend(); ax[0].set_title(th)
        ax[1].plot(fc, T); ax[1].set_xlabel("f (1/mm)"); ax[1].set_ylabel("T(f): D45 -> B30"); ax[1].axhline(1, ls=":", c="k")
        ax[2].plot(f, pr, label="real B30 noise (test patients)"); ax[2].plot(f, pe, "--", label="emulated from D45")
        ax[2].plot(f, pd45, ":", label="D45 noise"); ax[2].set_yscale("log"); ax[2].set_xlabel("f (1/mm)")
        ax[2].set_ylabel("NPS (HU^2 mm^2)"); ax[2].legend()
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_stage1_{th}.png", dpi=150); plt.close(fig)
    return res

PATIENTS = sorted(inv.patient.unique())
TRAIN_T = PATIENTS[:5]; TEST_T = PATIENTS[5:]          # transfer estimated on 5, tested on 5
K_SLICES, ROI, N_ROI = 12, 64, 20
rng = np.random.default_rng(2026)
results = {"patients": PATIENTS, "transfer_train": TRAIN_T, "transfer_test": TEST_T}

for th in ["1mm", "3mm"]:
    px = float(inv[inv.thick == th].px.median())
    store = {}
    for pt in PATIENTS:
        zs = sorted(set.intersection(*[set(inv[(inv.patient == pt) & (inv.thick == th) & (inv.kernel == k) & (inv.dose == d)].zr)
                                        for k in ["B30", "D45"] for d in ["FD", "QD"]]))
        if len(zs) < K_SLICES + 4:
            print("WARN few slices", pt, th, len(zs)); continue
        sel = [zs[i] for i in np.linspace(2, len(zs) - 3, K_SLICES).round().astype(int)]
        for z in sel:
            store[(pt, z)] = {(k, d): load_hu(pt, th, k, d, z) for k in ["B30", "D45"] for d in ["FD", "QD"]}
    print(th, "slices loaded:", len(store), "px", px, "t=%.0fs" % (time.time() - T0))
    # ---- registration check: sub-pixel shift FD vs QD and B30 vs D45 (phase correlation, 1/50 px)
    from skimage.registration import phase_cross_correlation
    from scipy.signal.windows import tukey
    sh = []
    for (pt, z), im in store.items():
        for a, b, lab in [(("B30", "FD"), ("B30", "QD"), "FDvsQD_B30"), (("D45", "FD"), ("D45", "QD"), "FDvsQD_D45"),
                          (("B30", "FD"), ("D45", "FD"), "B30vsD45_FD")]:
            A = np.clip(im[a], -200, 300); B = np.clip(im[b], -200, 300)
            w = np.outer(tukey(A.shape[0], 0.25), tukey(A.shape[1], 0.25))
            # plain cross-correlation + Tukey window (validated: |error| <= 0.08 px on synthetic CT, 5 known shifts; 'phase' normalisation was biased)
            s_, _, _ = phase_cross_correlation(A * w, B * w, upsample_factor=50, normalization=None)
            sh.append(dict(patient=pt, z=z, pair=lab, dy=float(s_[0]), dx=float(s_[1])))
    sh = pd.DataFrame(sh); sh.to_csv(f"{OUT}/shifts_{th}.csv", index=False)
    results[f"{th}_shift_px_max_abs"] = sh.groupby("pair")[["dy", "dx"]].apply(lambda g: float(np.abs(g.values).max())).to_dict()
    print("max |shift| px:", results[f"{th}_shift_px_max_abs"])
    results.update(analyze_thickness(store, px, th, TRAIN_T, TEST_T, OUT, ROI, N_ROI))

results["runtime_s"] = round(time.time() - T0, 1)
json.dump(results, open(f"{OUT}/stage1_results.json", "w"), indent=1, default=float)
import shutil; shutil.make_archive("/kaggle/working/stage1_outputs", "zip", OUT)
print("STAGE1_JSON=" + json.dumps(results, default=float))
