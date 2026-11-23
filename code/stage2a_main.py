# ============================ STAGE 2a (CPU) : hybrid kernel operator, family checks, W1 matrices, H5 data, cache
# Pre-registered (protocol doc, 2026-11-14):
#   T_h = (1-w) T_img + w T_roi, w = logistic((f-0.06)/0.015), clipped >= 1e-3 ; K_alpha = T_h^alpha (physical f).
#   (crossover moved from 0.15 to 0.06 /mm after the synthetic-truth test, BEFORE any real-data run: T_img is
#    clipping-biased in 0.05-0.15 /mm while T_roi is exact there; T_roi is resolution-limited below ~2/(ROI*px).)
#   Check 1: LOPO T_h at alpha=1 satisfies C1, C2 AND C3 (Stage-1 definitions) in >= 9/10 patients.
#   Check 2: W1(alpha, 0) strictly increasing in alpha for every patient.
#   Check 3: each off-family kernel at W1 >= 0.01 /mm from the nearest family member. Off-family kernels
#   (revised after the synthetic test showed s=1.25 lies ON the family path): (i) T_h(f/0.8) at alpha=1;
#   (ii) order-8 Butterworth 1/(1+(f/fc)^8) on D45 with fc set so that its pooled NPS f_mean equals alpha=0.5.
import os, re, glob, json, time, zlib, shutil, hashlib
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDS = {1: ["L067", "L291"], 2: ["L109", "L143"], 3: ["L192", "L310"], 4: ["L333", "L096"], 5: ["L506", "L286"]}
VAL = {1: "L143", 2: "L310", 3: "L096", 4: "L286", 5: "L291"}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_CONDS = [(a, 1.0) for a in ALPHAS] + [(a, 0.5) for a in (0.0, 0.5, 1.0)]
BETAS_TEST = [0.5, 0.75, 1.0]
OFF_S = [0.8]
BW_ORDER = 8
FX, FW = 0.06, 0.015
K_SLICES, ROI, N_ROI, DF = 20, 64, 20, 0.01
NB = int(np.ceil(1.2 / DF))
FC = (np.arange(NB) + 0.5) * DF

def tissue_mask(b30fd, erode_px=12):
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

def hybrid(Ti, Tr):
    w = 1.0 / (1.0 + np.exp(-(FC - FX) / FW))
    return np.maximum((1 - w) * Ti + w * Tr, 1e-3)

def kfilter(img, d, Th, alpha=1.0, s=1.0):
    """Apply K = T_h(|f|/s)^alpha to a slice with pixel spacing d (physical frequency)."""
    n = img.shape[0]
    f = np.fft.fftfreq(n, d)
    fr = np.hypot(*np.meshgrid(f, f)) / s
    H = np.interp(fr, FC, Th, right=Th[-1]) ** alpha
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))

def bwfilter(img, d, fc, order=BW_ORDER):
    n = img.shape[0]
    f = np.fft.fftfreq(n, d)
    H = 1.0 / (1.0 + (np.hypot(*np.meshgrid(f, f)) / fc) ** order)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))

def fmean_of(prof, FG):
    w = np.where(np.isfinite(prof), prof, 0) * FG
    return float((FG * w).sum() / w.sum())

def roi_positions(m, key):
    r = np.random.default_rng(seed_of("roi", key)); ys, xs = [], []; tries = 0
    while len(ys) < N_ROI and tries < 50 * N_ROI:
        tries += 1; y = r.integers(0, m.shape[0] - ROI); x = r.integers(0, m.shape[1] - ROI)
        if m[y:y + ROI, x:x + ROI].all():
            ys.append(y); xs.append(x)
    return ys, xs

def tissue_rois_T(store, masks, keys, TS_list=(128, 96, 64)):
    for TS in TS_list:
        sr, dr = [], []
        for k in keys:
            m = masks[k]; r = np.random.default_rng(seed_of("T", k, TS)); tries = 0; got = 0
            nD = store[k]["D45_QD"] - store[k]["D45_FD"]; nB = store[k]["B30_QD"] - store[k]["B30_FD"]
            while got < 6 and tries < 400:
                tries += 1; y = r.integers(0, m.shape[0] - TS); x = r.integers(0, m.shape[1] - TS)
                if m[y:y + TS, x:x + TS].all():
                    sr.append(nD[y:y + TS, x:x + TS]); dr.append(nB[y:y + TS, x:x + TS]); got += 1
        if len(sr) >= 40:
            break
    assert len(sr) >= 20, f"too few tissue ROIs: {len(sr)}"
    return np.asarray(sr), np.asarray(dr), TS

def ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)

def analysis(store, PX, PATIENTS, OUT):
    """store[(pt, z)] = dict B30_FD, B30_QD, D45_FD, D45_QD, D45_QDn, B30_QDn (n = adjacent slice)."""
    res = {}
    masks = {k: tissue_mask(v["B30_FD"]) for k, v in store.items()}
    keys_of = {p: [k for k in store if k[0] == p] for p in PATIENTS}
    # ---- per-patient transfer sums (image-level and tissue-ROI noise-level)
    S_img, S_roi = {}, {}
    for p in PATIENTS:
        ks = keys_of[p]; d = PX[p]
        S_img[p] = transfer_sums([store[k]["D45_FD"] for k in ks], [store[k]["B30_FD"] for k in ks], d, DF, NB)
        sr, dr, _ = tissue_rois_T(store, masks, ks)
        S_roi[p] = transfer_sums_rois(sr, dr, d, DF, NB)
    def T_from(pats):
        Ti = finalize_transfer(sum(S_img[q][0] for q in pats), sum(S_img[q][1] for q in pats), DF)[1]
        Tr = finalize_transfer(sum(S_roi[q][0] for q in pats), sum(S_roi[q][1] for q in pats), DF)[1]
        return Ti, Tr, hybrid(Ti, Tr)
    Ti_all, Tr_all, Th_all = T_from(PATIENTS)
    pd.DataFrame({"f": FC, "T_img": Ti_all, "T_roi": Tr_all, "T_h": Th_all}).to_csv(f"{OUT}/kernel_operator.csv", index=False)
    fny = min(0.5 / PX[p] for p in PATIENTS)
    FG = np.arange(0.5, int(fny / 0.02)) * 0.02
    res["FG_max"] = float(FG[-1])

    def nps_prof(rois, d):
        P = nps2d(np.asarray(rois), d, order=2)
        return radial_on_grid(P, d, FG), nps_summary(P, d)["variance"]

    # ---- Check 1 (LOPO, alpha = 1)
    rows = []
    for p in PATIENTS:
        _, _, Th = T_from([q for q in PATIENTS if q != p]); d = PX[p]
        R = {"emu": [], "B30": [], "D45": []}; rat = {"emu": [], "unf": []}
        for k in keys_of[p]:
            v, m = store[k], masks[k]
            nB = v["B30_QD"] - v["B30_FD"]; nD = v["D45_QD"] - v["D45_FD"]; sig = float(np.std(nB[m]))
            e = kfilter(v["D45_QD"], d, Th)
            rat["emu"].append(float(np.sqrt(np.mean((e - v["B30_QD"])[m] ** 2))) / sig)
            rat["unf"].append(float(np.sqrt(np.mean((v["D45_QD"] - v["B30_QD"])[m] ** 2))) / sig)
            ys, xs = roi_positions(m, k); eN = kfilter(nD, d, Th)
            for nm, img in [("emu", eN), ("B30", nB), ("D45", nD)]:
                R[nm] += [img[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
        pr = {nm: nps_prof(v_, d) for nm, v_ in R.items()}
        w_emu = nnps_distance(pr["emu"][0], pr["B30"][0], FG); w_d45 = nnps_distance(pr["D45"][0], pr["B30"][0], FG)
        row = dict(patient=p, ratio_emu_QD=float(np.median(rat["emu"])), ratio_unf_QD=float(np.median(rat["unf"])),
                   W1_emu=w_emu, W1_D45=w_d45, var_ratio=pr["emu"][1] / pr["B30"][1])
        row["C1"] = row["ratio_emu_QD"] <= 0.5 * row["ratio_unf_QD"]; row["C2"] = w_emu <= 0.2 * w_d45
        row["C3"] = abs(row["var_ratio"] - 1) <= 0.10
        rows.append(row)
    c1 = pd.DataFrame(rows); c1.to_csv(f"{OUT}/check1_lopo_Th.csv", index=False)
    res["check1"] = {"n_C1": int(c1.C1.sum()), "n_C2": int(c1.C2.sum()), "n_C3": int(c1.C3.sum()),
                     "median_var_ratio": float(c1.var_ratio.median()), "min_var_ratio": float(c1.var_ratio.min()),
                     "max_var_ratio": float(c1.var_ratio.max()), "median_ratio_emu_QD": float(c1.ratio_emu_QD.median()),
                     "median_W1_emu": float(c1.W1_emu.median()), "median_W1_D45": float(c1.W1_D45.median())}
    res["check1"]["PASS"] = bool(min(res["check1"]["n_C1"], res["check1"]["n_C2"], res["check1"]["n_C3"]) >= 9)

    # ---- reference and estimated noise NPS per patient and condition (beta = 1), with Th_all
    CONDS = [f"a{a}" for a in ALPHAS] + ["B30real"] + [f"off{s}" for s in OFF_S] + ["offBW"]
    # Butterworth cutoff: pooled f_mean of |H|^2 * NPS_D45 equal to that of |T_h^0.5|^2 * NPS_D45 (bisection)
    pD = []
    for p in PATIENTS:
        d = PX[p]; R = []
        for k in keys_of[p]:
            ys, xs = roi_positions(masks[k], k); nD = store[k]["D45_QD"] - store[k]["D45_FD"]
            R += [nD[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
        pD.append(nps_prof(R, d)[0])
    pD = np.nanmean(np.array(pD), 0)
    target = fmean_of(pD * np.interp(FG, FC, Th_all) ** (2 * 0.5), FG)
    lo, hi = 0.05, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fmean_of(pD / (1 + (FG / mid) ** BW_ORDER) ** 2, FG) < target: lo = mid
        else: hi = mid
    FC_BW = 0.5 * (lo + hi); res["butterworth_fc"] = FC_BW; res["fmean_target_alpha0.5"] = target
    AFINE = np.round(np.arange(0, 1.0001, 0.05), 2)
    ref, est, var, fine = {}, {}, {}, {}
    for p in PATIENTS:
        d = PX[p]; Rr = {c: [] for c in CONDS}; Re = {c: [] for c in CONDS}; Rf = {a: [] for a in AFINE}
        for k in keys_of[p]:
            v, m = store[k], masks[k]; ys, xs = roi_positions(m, k)
            nD = v["D45_QD"] - v["D45_FD"]; nB = v["B30_QD"] - v["B30_FD"]
            dD = (v["D45_QD"] - v["D45_QDn"]) / np.sqrt(2); dB = (v["B30_QD"] - v["B30_QDn"]) / np.sqrt(2)
            pairs = {f"a{a}": (kfilter(nD, d, Th_all, a), kfilter(dD, d, Th_all, a)) for a in ALPHAS}
            pairs["B30real"] = (nB, dB)
            for s in OFF_S:
                pairs[f"off{s}"] = (kfilter(nD, d, Th_all, 1.0, s), kfilter(dD, d, Th_all, 1.0, s))
            pairs["offBW"] = (bwfilter(nD, d, FC_BW), bwfilter(dD, d, FC_BW))
            for c, (a_ref, a_est) in pairs.items():
                Rr[c] += [a_ref[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
                Re[c] += [a_est[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
            for a in AFINE:
                g = kfilter(nD, d, Th_all, a); Rf[a] += [g[y:y + ROI, x:x + ROI] for y, x in zip(ys, xs)]
        for c in CONDS:
            ref[(p, c)], var[(p, c)] = nps_prof(Rr[c], d)
            est[(p, c)], _ = nps_prof(Re[c], d)
        for a in AFINE:
            fine[(p, a)] = nps_prof(Rf[a], d)[0]
    long = [dict(patient=p, cond=c, kind=kd, f=float(f_), nps=float(v_)) for (p, c), arr in ref.items()
            for kd, A in [("ref", arr), ("est", est[(p, c)])] for f_, v_ in zip(FG, A)]
    pd.DataFrame(long).to_csv(f"{OUT}/nps_radial_ref_est.csv", index=False)
    pd.DataFrame([dict(patient=p, cond=c, var_beta1=v_) for (p, c), v_ in var.items()]).to_csv(f"{OUT}/noise_variance_beta1.csv", index=False)

    # ---- Check 2: W1(alpha, 0) strictly increasing, every patient
    mono = {}
    for p in PATIENTS:
        w = [nnps_distance(fine[(p, a)], fine[(p, 0.0)], FG) for a in AFINE]
        mono[p] = bool(np.all(np.diff(w) > 0))
        res.setdefault("W1_alpha_vs_0_median", {})[p] = [round(x, 5) for x in w]
    res["check2"] = {"n_monotone": int(sum(mono.values())), "PASS": bool(all(mono.values()))}

    # ---- Check 3: off-family distance to nearest family member (pooled NPS over patients)
    pool = lambda D, c: np.nanmean(np.array([D[(p, c)] for p in PATIENTS]), 0)
    fam = {a: np.nanmean(np.array([fine[(p, a)] for p in PATIENTS]), 0) for a in AFINE}
    res["check3"] = {}
    for c in [f"off{s}" for s in OFF_S] + ["offBW"]:
        o = pool(ref, c)
        dmin = min(nnps_distance(o, fam[a], FG) for a in AFINE)
        res["check3"][c] = {"W1_nearest_family": float(dmin), "fmean": fmean_of(o, FG), "PASS": bool(dmin >= 0.01)}
    res["fmean_family"] = {f"a{a}": fmean_of(pool(ref, f"a{a}"), FG) for a in ALPHAS}
    res["check3"]["PASS"] = bool(all(v["PASS"] for v in res["check3"].values() if isinstance(v, dict)))
    o_b = pool(ref, "B30real")
    res["B30real_W1_to_alpha1"] = float(nnps_distance(o_b, fam[1.0], FG))

    # ---- W1 / dlogvar matrices for every fold, training condition and test (patient, condition, beta)
    mat = []
    for fo, test in FOLDS.items():
        train = [p for p in PATIENTS if p not in test and p != VAL[fo]]
        for (a, b) in TRAIN_CONDS:
            ptr = np.nanmean(np.array([ref[(p, f"a{a}")] for p in train]), 0)
            vtr = float(np.mean([var[(p, f"a{a}")] for p in train])) * b ** 2
            for p in test:
                for c in CONDS:
                    for b2 in BETAS_TEST:
                        mat.append(dict(fold=fo, train_alpha=a, train_beta=b, test_patient=p, test_cond=c, test_beta=b2,
                                        W1_ref=nnps_distance(ptr, ref[(p, c)], FG),
                                        W1_est=nnps_distance(ptr, est[(p, c)], FG),
                                        dlogvar=float(np.log(var[(p, c)] * b2 ** 2 / vtr))))
    mat = pd.DataFrame(mat); mat.to_csv(f"{OUT}/pair_matrix.csv", index=False)
    # ---- H5 (computed here because it needs no denoiser; criterion unchanged: CCC >= 0.90, CI low >= 0.80)
    h = mat[mat.test_beta == 1.0].drop_duplicates(["fold", "train_alpha", "train_beta", "test_patient", "test_cond"])
    h = h[h.train_beta == 1.0]
    c_all = ccc(h.W1_est, h.W1_ref)
    rng = np.random.default_rng(2026); bs = []
    for _ in range(2000):
        pick = rng.choice(PATIENTS, len(PATIENTS), replace=True)
        hh = pd.concat([h[h.test_patient == q] for q in pick]); bs.append(ccc(hh.W1_est, hh.W1_ref))
    res["H5"] = {"CCC": float(c_all), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 "n_points": int(len(h)), "mean_diff_est_minus_ref": float((h.W1_est - h.W1_ref).mean()),
                 "sd_diff": float((h.W1_est - h.W1_ref).std())}
    res["H5"]["criterion_met"] = bool(c_all >= 0.90 and res["H5"]["ci95"][0] >= 0.80)

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(FC, Ti_all, label="T_img"); ax[0].plot(FC, Tr_all, label="T_roi"); ax[0].plot(FC, Th_all, "k--", label="T_h")
    ax[0].set_xlim(0, 1.0); ax[0].axvline(FX, c="gray", ls=":"); ax[0].legend(); ax[0].set_title("D45 -> B30 transfer (10 patients)")
    for a in ALPHAS:
        q = pool(ref, f"a{a}"); ax[1].plot(FG, q / np.nansum(q * FG), label=f"alpha={a}")
    q = pool(ref, "B30real"); ax[1].plot(FG, q / np.nansum(q * FG), "k:", label="real B30")
    for c in [f"off{s}" for s in OFF_S] + ["offBW"]:
        q = pool(ref, c); ax[1].plot(FG, q / np.nansum(q * FG), "--", label=c)
    ax[1].legend(fontsize=7); ax[1].set_title("normalised radial NPS (pooled)")
    ax[2].scatter(h.W1_ref, h.W1_est, s=6); lim = [0, max(h.W1_ref.max(), h.W1_est.max()) * 1.05]
    ax[2].plot(lim, lim, "k:"); ax[2].set_xlabel("W1 reference (QD-FD)"); ax[2].set_ylabel("W1 estimated (adjacent slices)")
    ax[2].set_title(f"H5: CCC={c_all:.3f}")
    for a in ax[:2]: a.set_xlabel("f (1/mm)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_stage2a.png", dpi=150); plt.close(fig)
    return res, Th_all

# ====================================================================== I/O (Kaggle only)
if __name__ == "__main__" and not os.environ.get("STAGE2A_TEST"):
    import pydicom
    OUT = "/kaggle/working/stage2a"; CACHE = "/kaggle/working/cache"
    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    T0 = time.time()
    ROOT = sorted(glob.glob("/kaggle/input/**/Original Data", recursive=True), key=len)[0]
    files = sorted(p for p in glob.glob(os.path.join(ROOT, "**", "*.IMA"), recursive=True) if "1mm" in p)
    assert len(files) == 23744, len(files)
    inv = pd.DataFrame([dict(path=p, dose="FD" if "Full Dose" in p else "QD", kernel="D45" if "D45" in p else "B30",
                             patient=re.search(r"(L\d{3})", p).group(1)) for p in files])
    def hdr(p):
        d = pydicom.dcmread(p, stop_before_pixels=True)
        return dict(z=float(d.ImagePositionPatient[2]), px=float(d.PixelSpacing[0]), slope=float(d.RescaleSlope),
                    icpt=float(d.RescaleIntercept), kvp=str(d.KVP), model=str(d.ManufacturerModelName))
    with ThreadPoolExecutor(32) as ex:
        inv = pd.concat([inv, pd.DataFrame(list(ex.map(hdr, inv.path)))], axis=1)
    inv["zr"] = inv.z.round(2)
    assert (inv.slope == 1).all(), "RescaleSlope != 1 : int16 cache would not be exact"
    PATIENTS = sorted(inv.patient.unique()); assert len(PATIENTS) == 10
    assert sorted(sum(FOLDS.values(), [])) == PATIENTS
    PX = {p: float(inv[inv.patient == p].px.iloc[0]) for p in PATIENTS}
    assert all(inv[inv.patient == p].px.nunique() == 1 for p in PATIENTS)
    lut = {(r.patient, r.kernel, r.dose, r.zr): r.path for r in inv.itertuples()}
    def hu(path):
        ds = pydicom.dcmread(path); return ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    def hu16(path):
        ds = pydicom.dcmread(path); a = ds.pixel_array.astype(np.int32) + int(round(float(ds.RescaleIntercept)))
        assert a.min() >= -32768 and a.max() <= 32767; return a.astype(np.int16)
    ZS = {}
    for p in PATIENTS:
        g = inv[inv.patient == p]
        ZS[p] = sorted(set.intersection(*[set(g[(g.kernel == k) & (g.dose == d)].zr) for k in ["B30", "D45"] for d in ["FD", "QD"]]))
    # ---- analysis subset with adjacent slices
    store = {}
    for p in PATIENTS:
        zs = ZS[p]; idx = np.linspace(2, len(zs) - 3, K_SLICES).round().astype(int)
        for i in idx:
            z, zn = zs[i], zs[i + 1]
            store[(p, z)] = {"B30_FD": hu(lut[(p, "B30", "FD", z)]), "B30_QD": hu(lut[(p, "B30", "QD", z)]),
                             "D45_FD": hu(lut[(p, "D45", "FD", z)]), "D45_QD": hu(lut[(p, "D45", "QD", z)]),
                             "D45_QDn": hu(lut[(p, "D45", "QD", zn)]), "B30_QDn": hu(lut[(p, "B30", "QD", zn)])}
    dz = {p: float(np.median(np.diff(ZS[p]))) for p in PATIENTS}
    print("analysis slices", len(store), "slice spacing (mm)", dz, "t=%.0fs" % (time.time() - T0))
    res, Th = analysis(store, PX, PATIENTS, OUT)
    res["slice_spacing_mm"] = dz
    del store
    print("ANALYSIS_JSON=" + json.dumps(res, default=float))
    # ---- training cache: every 2nd slice, int16 HU (exact), D45 FD/QD; B30 FD/QD on the 50 test slices
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "folds": FOLDS, "val": VAL, "px": PX,
                "T_h": {"f": FC.tolist(), "T": Th.tolist()}, "patients": {}}
    for p in PATIENTS:
        zc = ZS[p][::2]
        ti = np.linspace(2, len(zc) - 3, 50).round().astype(int).tolist()
        vi = np.linspace(3, len(zc) - 4, 30).round().astype(int).tolist()
        with ThreadPoolExecutor(16) as ex:
            fd = np.stack(list(ex.map(lambda z: hu16(lut[(p, "D45", "FD", z)]), zc)))
            qd = np.stack(list(ex.map(lambda z: hu16(lut[(p, "D45", "QD", z)]), zc)))
            bfd = np.stack(list(ex.map(lambda i: hu16(lut[(p, "B30", "FD", zc[i])]), ti)))
            bqd = np.stack(list(ex.map(lambda i: hu16(lut[(p, "B30", "QD", zc[i])]), ti)))
        ent = {"z": [float(z) for z in zc], "test_idx": ti, "val_idx": vi,
               "kvp": inv[inv.patient == p].kvp.iloc[0], "model": inv[inv.patient == p].model.iloc[0], "sha256": {}}
        for nm, arr in [("D45_FD", fd), ("D45_QD", qd), ("B30_FD_test", bfd), ("B30_QD_test", bqd)]:
            fn = f"{CACHE}/{p}_{nm}.npy"; np.save(fn, arr)
            ent["sha256"][nm] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
        ent["n_cached"] = len(zc); manifest["patients"][p] = ent
        print(p, "cached", len(zc), "t=%.0fs" % (time.time() - T0))
    json.dump(manifest, open(f"{CACHE}/manifest.json", "w"), indent=1)
    res["runtime_s"] = round(time.time() - T0, 1)
    json.dump(res, open(f"{OUT}/stage2a_results.json", "w"), indent=1, default=float)
    print("STAGE2A_JSON=" + json.dumps(res, default=float))
