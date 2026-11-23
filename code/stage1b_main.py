
# ============================ STAGE 1b : per-patient pixel spacing, LOPO emulation, CIs, 3 mm diagnosis
# Pre-registered success criteria (fixed after synthetic-truth testing, BEFORE any real-data run of 1b):
#   Estimators of T(f) D45->B30 (all leave-one-patient-out, physical frequency grid):
#     'img'   : whole-image least squares on FD images      -> image-level realism
#     'roi'   : cross-spectrum of paired Hann-windowed tissue ROIs of the noise images (QD-FD)
#               -> noise-texture realism (immune to the -1024 HU clipping of air/lung)
#     'noise' : whole-image least squares on noise images (reported only; biased by clipping in tests)
#   C1 per patient ('img'):  median ratio_emulated(QD) <= 0.5 * median ratio_unfiltered(QD)
#   C2 per patient ('roi'):  W1(emulated noise, real B30 noise) <= 0.2 * W1(D45 noise, real B30 noise)
#   C3 per patient ('roi'):  |var(emulated noise)/var(real B30 noise) - 1| <= 0.10   (diagnostic)
#   Method accepted for a thickness if C1('img') and C2('roi') hold in >= 9/10 patients.
import os, re, glob, json, time, zlib, shutil
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pydicom
from scipy import ndimage, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working/stage1b"; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
ROOT = sorted(glob.glob("/kaggle/input/**/Original Data", recursive=True), key=len)[0]
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.IMA"), recursive=True))
assert len(files) == 33256, len(files)

def parse(p):
    return dict(path=p, dose="FD" if "Full Dose" in p else "QD",
                thick="1mm" if "1mm" in p else "3mm",
                kernel="D45" if "D45" in p else "B30",
                patient=re.search(r"(L\d{3})", p).group(1))
inv = pd.DataFrame([parse(p) for p in files])

def hdr(p):
    d = pydicom.dcmread(p, stop_before_pixels=True)
    return dict(z=float(d.ImagePositionPatient[2]), px=float(d.PixelSpacing[0]), py=float(d.PixelSpacing[1]),
                rows=int(d.Rows), cols=int(d.Columns), kvp=str(d.KVP), model=str(d.ManufacturerModelName))
with ThreadPoolExecutor(32) as ex:
    inv = pd.concat([inv, pd.DataFrame(list(ex.map(hdr, inv.path)))], axis=1)
inv["zr"] = inv.z.round(2)
assert (inv.px == inv.py).all(), "non-square pixels"
assert ((inv.rows == 512) & (inv.cols == 512)).all()

# per-patient geometry: pixel spacing must be identical across the 4 series of a patient/thickness
geo = inv.groupby(["patient", "thick"]).agg(px_min=("px", "min"), px_max=("px", "max"),
                                            kvp=("kvp", lambda s: "/".join(sorted(set(s)))),
                                            model=("model", lambda s: "/".join(sorted(set(s))))).reset_index()
geo.to_csv(f"{OUT}/geometry_per_patient.csv", index=False)
print(geo.to_string())
assert (geo.px_min == geo.px_max).all(), "pixel spacing varies within a patient series"
PX = {(r.patient, r.thick): r.px_min for r in geo.itertuples()}

lut = {(r.patient, r.thick, r.kernel, r.dose, r.zr): r.path for r in inv.itertuples()}
def load(pt, th, k, d, z):
    ds = pydicom.dcmread(lut[(pt, th, k, d, z)])
    raw = ds.pixel_array
    hu = raw.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return hu, int((raw == raw.min()).sum()), int(raw.min())

def tissue_mask(b30fd, erode_px=12):
    """Patient body (largest component > -300 HU, holes filled), restricted to non-lung tissue
    (-300 < HU), eroded so 64x64 ROIs stay inside homogeneous-enough tissue."""
    m = b30fd > -300
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    body = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    body = ndimage.binary_fill_holes(body)
    tis = body & (ndimage.uniform_filter((b30fd > -300).astype(float), 9) > 0.999)
    return ndimage.binary_erosion(tis, iterations=erode_px)

def seed_of(*a):
    return zlib.crc32(repr(a).encode())

PATIENTS = sorted(inv.patient.unique()); assert len(PATIENTS) == 10
K_SLICES, ROI, N_ROI = 20, 64, 20
DF = 0.01                                   # common physical frequency grid for transfer [1/mm]
NB = int(np.ceil(1.2 / DF))                 # up to 1.2 /mm  (> sqrt(2) * Nyquist of the finest grid)
results = {"criteria": "C1(img): ratio_emu_QD <= 0.5*ratio_unf_QD; C2(roi): W1_emu <= 0.2*W1_D45; C3(roi) |var ratio-1|<=0.10; accept if C1(img)&C2(roi) in >=9/10",
           "K_SLICES": K_SLICES, "ROI": ROI, "N_ROI": N_ROI, "DF": DF}

def boot_ci(x, stat=np.median, B=10000, seed=2026):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(B)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def run_thickness(th, inv, PX, load, PATIENTS, OUT, results):
    # ---------------------------------------------------- load K_SLICES evenly spaced quadruplets per patient
    store, clip = {}, []
    for pt in PATIENTS:
        g = inv[(inv.patient == pt) & (inv.thick == th)]
        zs = sorted(set.intersection(*[set(g[(g.kernel == k) & (g.dose == d)].zr) for k in ["B30", "D45"] for d in ["FD", "QD"]]))
        sel = [zs[i] for i in np.linspace(2, len(zs) - 3, K_SLICES).round().astype(int)]
        for z in sel:
            im = {}
            for k in ["B30", "D45"]:
                for d in ["FD", "QD"]:
                    hu, nmin, rmin = load(pt, th, k, d, z); im[(k, d)] = hu
                    clip.append(dict(patient=pt, z=z, kernel=k, dose=d, n_at_min=nmin, raw_min=rmin))
            store[(pt, z)] = im
    clip = pd.DataFrame(clip); clip.to_csv(f"{OUT}/clipping_{th}.csv", index=False)
    print(th, "loaded", len(store), "t=%.0fs" % (time.time() - T0))
    masks = {key: tissue_mask(im[("B30", "FD")]) for key, im in store.items()}
    # clipping inside the analysed tissue: fraction of pixels at the lowest stored value
    cl_in = {f"{k}_{d}": {"frac_pixels_at_min_value": float(clip[(clip.kernel == k) & (clip.dose == d)].n_at_min.mean() / (512 * 512)),
                          "raw_min_values": sorted(set(map(int, clip[(clip.kernel == k) & (clip.dose == d)].raw_min)))[:5]}
             for k in ["B30", "D45"] for d in ["FD", "QD"]}

    # ---------------------------------------------------- per-patient transfer sums (image- and noise-based)
    sums_img, sums_noise, sums_roi, n_troi, troi_size = {}, {}, {}, {}, {}
    for pt in PATIENTS:
        keys = [k for k in store if k[0] == pt]; d = PX[(pt, th)]
        sums_img[pt] = transfer_sums([store[k][("D45", "FD")] for k in keys], [store[k][("B30", "FD")] for k in keys], d, DF, NB)
        sums_noise[pt] = transfer_sums([store[k][("D45", "QD")] - store[k][("D45", "FD")] for k in keys],
                                       [store[k][("B30", "QD")] - store[k][("B30", "FD")] for k in keys], d, DF, NB)
        # tissue-ROI cross-spectral estimator: paired Hann-windowed ROIs of the noise images, largest size giving >= 40 ROIs
        for TS in (128, 96, 64):
            sr, dr = [], []
            for k in keys:
                m = masks[k]; r = np.random.default_rng(seed_of("T", th, k, TS)); tries = 0; got = 0
                nD = store[k][("D45", "QD")] - store[k][("D45", "FD")]; nB = store[k][("B30", "QD")] - store[k][("B30", "FD")]
                while got < 6 and tries < 400:
                    tries += 1; y = r.integers(0, m.shape[0] - TS); x = r.integers(0, m.shape[1] - TS)
                    if m[y:y + TS, x:x + TS].all():
                        sr.append(nD[y:y + TS, x:x + TS]); dr.append(nB[y:y + TS, x:x + TS]); got += 1
            if len(sr) >= 40:
                break
        assert len(sr) >= 20, f"too few tissue ROIs for transfer: {pt} {th} {len(sr)}"
        sums_roi[pt] = transfer_sums_rois(np.asarray(sr), np.asarray(dr), d, DF, NB)
        troi_size[pt] = TS
        n_troi[pt] = len(sr)

    def lopo(sums, held):
        num = sum(sums[p][0] for p in PATIENTS if p != held); den = sum(sums[p][1] for p in PATIENTS if p != held)
        return finalize_transfer(num, den, DF)

    # common physical grid for radial NPS: up to the smallest Nyquist among patients
    fny = min(0.5 / PX[(p, th)] for p in PATIENTS)
    FG = np.arange(0.5, int(fny / 0.02)) * 0.02

    rows, slices, curves = [], [], {}
    for pt in PATIENTS:
        d = PX[(pt, th)]; keys = [k for k in store if k[0] == pt]
        TR = {"img": lopo(sums_img, pt)[:2], "noise": lopo(sums_noise, pt)[:2], "roi": lopo(sums_roi, pt)[:2]}
        R = {n: [] for n in ["B30", "D45", "emu_img", "emu_noise", "emu_roi"]}
        for key in keys:
            im, m = store[key], masks[key]
            nB = im[("B30", "QD")] - im[("B30", "FD")]; nD = im[("D45", "QD")] - im[("D45", "FD")]
            r = np.random.default_rng(seed_of(th, key))
            ys, xs = [], []
            H, W = m.shape; tries = 0                      # SAME ROI positions for every noise image of the slice
            while len(ys) < N_ROI and tries < 50 * N_ROI:
                tries += 1; y = r.integers(0, H - ROI); x = r.integers(0, W - ROI)
                if m[y:y + ROI, x:x + ROI].all():
                    ys.append(y); xs.append(x)
            sig = float(np.std(nB[m])) if m.any() else np.nan
            for tag in ["img", "noise", "roi"]:
                fc, T = TR[tag]
                emuN = apply_radial_filter(nD, fc, T, d)
                R[f"emu_{tag}"] += [emuN[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
                for dose in ["FD", "QD"]:
                    e = apply_radial_filter(im[("D45", dose)], fc, T, d)
                    slices.append(dict(patient=pt, z=key[1], est=tag, dose=dose, sigma_B30_noise=sig,
                                       ratio_emulated=float(np.sqrt(np.mean((e - im[("B30", dose)])[m] ** 2))) / sig,
                                       ratio_unfiltered=float(np.sqrt(np.mean((im[("D45", dose)] - im[("B30", dose)])[m] ** 2))) / sig))
            R["B30"] += [nB[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
            R["D45"] += [nD[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
        assert len(R["B30"]) >= 100, f"too few ROIs for {pt} {th}: {len(R['B30'])}"
        P = {n: nps2d(np.asarray(v), d, order=2) for n, v in R.items()}
        prof = {n: radial_on_grid(p, d, FG) for n, p in P.items()}
        curves[pt] = prof
        summ = {n: nps_summary(p, d) for n, p in P.items()}
        sl = pd.DataFrame([s for s in slices if s["patient"] == pt])
        row = dict(patient=pt, px=d, n_rois=len(R["B30"]), n_transfer_rois=n_troi[pt], transfer_roi_size=troi_size[pt],
                   var_B30=summ["B30"]["variance"], var_D45=summ["D45"]["variance"],
                   fmean_B30=summ["B30"]["f_mean"], fmean_D45=summ["D45"]["f_mean"],
                   fpeak_B30=summ["B30"]["f_peak"], fpeak_D45=summ["D45"]["f_peak"],
                   W1_D45_vs_B30=nnps_distance(prof["D45"], prof["B30"], FG))
        for tag in ["img", "noise", "roi"]:
            s = sl[sl.est == tag]
            row.update({f"{tag}_ratio_emu_FD": float(s[s.dose == "FD"].ratio_emulated.median()),
                        f"{tag}_ratio_unf_FD": float(s[s.dose == "FD"].ratio_unfiltered.median()),
                        f"{tag}_ratio_emu_QD": float(s[s.dose == "QD"].ratio_emulated.median()),
                        f"{tag}_ratio_unf_QD": float(s[s.dose == "QD"].ratio_unfiltered.median()),
                        f"{tag}_W1_emu_vs_B30": nnps_distance(prof[f"emu_{tag}"], prof["B30"], FG),
                        f"{tag}_var_ratio": summ[f"emu_{tag}"]["variance"] / summ["B30"]["variance"]})
            row[f"{tag}_C1"] = row[f"{tag}_ratio_emu_QD"] <= 0.5 * row[f"{tag}_ratio_unf_QD"]
            row[f"{tag}_C2"] = row[f"{tag}_W1_emu_vs_B30"] <= 0.2 * row["W1_D45_vs_B30"]
            row[f"{tag}_C3"] = abs(row[f"{tag}_var_ratio"] - 1) <= 0.10
        rows.append(row)
    pp = pd.DataFrame(rows); pp.to_csv(f"{OUT}/per_patient_{th}.csv", index=False)
    pd.DataFrame(slices).to_csv(f"{OUT}/per_slice_{th}.csv", index=False)
    print(pp.round(4).to_string())

    # ---------------------------------------------------- is the kernel transfer defined in mm (physical) ?
    Tp_phys = {p: finalize_transfer(*sums_img[p], DF)[1] for p in PATIENTS}
    fc_phys = (np.arange(NB) + 0.5) * DF
    band = (fc_phys >= 0.05) & (fc_phys <= 0.6)
    Mphys = np.array([Tp_phys[p][band] for p in PATIENTS])
    spread_phys = float(np.mean(np.abs(Mphys - np.median(Mphys, 0))))
    # same per-patient transfers expressed in cycles/pixel, compared on the equivalent band
    DFp = DF * float(np.median([PX[(p, th)] for p in PATIENTS]))
    Tp_pix = {}
    for p in PATIENTS:
        keys = [k for k in store if k[0] == p]
        Tp_pix[p] = finalize_transfer(*transfer_sums([store[k][("D45", "FD")] for k in keys],
                                                      [store[k][("B30", "FD")] for k in keys], 1.0, DFp, NB), DFp)[1]
    Mpix = np.array([Tp_pix[p][band] for p in PATIENTS])
    spread_pix = float(np.mean(np.abs(Mpix - np.median(Mpix, 0))))

    # ---------------------------------------------------- summary with patient-level bootstrap CIs
    summary = {"pixel_spacing_range": [float(pp.px.min()), float(pp.px.max())],
               "clipping_whole_image": cl_in,
               "transfer_spread_physical_vs_pixel": [spread_phys, spread_pix],
               "nps": {c: {"median": float(pp[c].median()), "ci95": boot_ci(pp[c]), "min": float(pp[c].min()), "max": float(pp[c].max())}
                       for c in ["var_B30", "var_D45", "fmean_B30", "fmean_D45", "fpeak_B30", "fpeak_D45", "W1_D45_vs_B30"]}}
    for tag in ["img", "noise", "roi"]:
        e, u = pp[f"{tag}_ratio_emu_QD"], pp[f"{tag}_ratio_unf_QD"]
        summary[tag] = {c: {"median": float(pp[f"{tag}_{c}"].median()), "ci95": boot_ci(pp[f"{tag}_{c}"]),
                            "min": float(pp[f"{tag}_{c}"].min()), "max": float(pp[f"{tag}_{c}"].max())}
                        for c in ["ratio_emu_FD", "ratio_unf_FD", "ratio_emu_QD", "ratio_unf_QD", "W1_emu_vs_B30", "var_ratio"]}
        summary[tag]["wilcoxon_emu_vs_unf_QD_p"] = float(stats.wilcoxon(e, u).pvalue)
        summary[tag]["n_C1"] = int(pp[f"{tag}_C1"].sum()); summary[tag]["n_C2"] = int(pp[f"{tag}_C2"].sum())
        summary[tag]["n_C3"] = int(pp[f"{tag}_C3"].sum())
        summary[tag]["ACCEPT"] = bool(summary[tag]["n_C1"] >= 9 and summary[tag]["n_C2"] >= 9)
    summary["ACCEPT"] = bool(summary["img"]["n_C1"] >= 9 and summary["roi"]["n_C2"] >= 9)
    results[th] = summary

    # ---------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for p in PATIENTS:
        cb, cd = curves[p]["B30"], curves[p]["D45"]
        ax[0].plot(FG, cb / np.nansum(cb * FG), "C0", alpha=.5); ax[0].plot(FG, cd / np.nansum(cd * FG), "C1", alpha=.5)
        ax[1].plot(fc_phys[band], Tp_phys[p][band], "k", alpha=.5)
        ax[2].plot(FG, curves[p]["B30"], "C0", alpha=.5); ax[2].plot(FG, curves[p]["emu_roi"], "C3--", alpha=.5)
    ax[0].set_title(f"{th}: normalised radial NPS (blue B30, orange D45), 10 patients")
    ax[1].set_title("T(f) D45->B30 per patient (physical freq.)"); ax[2].set_title("real B30 noise (blue) vs emulated (red, LOPO)")
    ax[2].set_yscale("log")
    for a in ax: a.set_xlabel("f (1/mm)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_stage1b_{th}.png", dpi=150); plt.close(fig)
    del store, masks

    return results

if __name__ == "__main__" and not os.environ.get("STAGE1B_TEST"):
    for th in ["1mm", "3mm"]:
        run_thickness(th, inv, PX, load, PATIENTS, OUT, results)
    results["runtime_s"] = round(time.time() - T0, 1)
    json.dump(results, open(f"{OUT}/stage1b_results.json", "w"), indent=1, default=float)
    shutil.make_archive("/kaggle/working/stage1b_outputs", "zip", OUT)
    print("STAGE1B_JSON=" + json.dumps(results, default=float))
