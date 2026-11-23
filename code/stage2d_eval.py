# ============================ STAGE 2d (GPU) : test-set inference of the 80 frozen models + pre-registered statistics
# Exec'd AFTER the Stage-2c library (LIB) so that models, kernel operator, cache loading, tissue mask and EvalSet are
# byte-identical to training. Evaluation is float32 only. Nothing here is tuned on test data.
#   test grid per model and test patient : family alpha in {0,.25,.5,.75,1} x beta in {.5,.75,1}  (15)
#                                          + real B30, T_h(f/0.8) (alpha=1), Butterworth (fc from Stage 2a) x beta in {.5,1} (6)
#   metrics : RMSE in the tissue mask pooled over the 50 test slices (primary, as in training), RMSE of the input,
#             SSIM (clip [-160, 240] HU, data range 400, 7x7 uniform window = skimage defaults), residual-noise NPS
#             (output - target) in 64x64 tissue ROIs (ICRU 87 / TG-233 normalisation, order-2 detrend).
import os, sys, json, time, glob, math, zlib, gzip, base64, io, subprocess, shutil
import numpy as np, pandas as pd, torch, torch.nn.functional as F

R2A_PATH = sorted(glob.glob("/kaggle/input/**/stage2a/stage2a_results.json", recursive=True)); assert len(R2A_PATH) == 1, R2A_PATH
R2A = json.load(open(R2A_PATH[0])); FC_BW = float(R2A["butterworth_fc"])
PM_PATH = os.path.join(os.path.dirname(R2A_PATH[0]), "pair_matrix.csv"); assert os.path.exists(PM_PATH)
MD = sorted(glob.glob("/kaggle/input/**/stage2c/REDCNN_f1_a0.0_b1.0.pt", recursive=True)); assert len(MD) == 1, MD
MODEL_DIR = os.path.dirname(MD[0])
EOUT = "/kaggle/working/stage2d"; os.makedirs(EOUT, exist_ok=True)
BETAS_FAM = [0.5, 0.75, 1.0]; BETAS_OFF = [0.5, 1.0]
TEST_CONDS = [(f"a{float(a)}", b) for a in ALPHAS for b in BETAS_FAM] + [(c, b) for c in ("B30real", "off0.8", "offBW") for b in BETAS_OFF]
FNY = min(0.5 / PX[p] for p in PATIENTS); FG = np.arange(0.5, int(FNY / 0.02)) * 0.02      # same grid as Stage 2a
ROI_N, ROI_PER_SLICE = 64, 4
EVAL_BS = 4


def cond_pairs(es, c, b):
    if c == "B30real":
        return es.pairs(beta=b, real_b30=True)
    if c == "off0.8":
        return es.pairs(alpha=1.0, beta=b, s=0.8)
    if c == "offBW":
        return es.pairs(beta=b, bw_fc=FC_BW)
    return es.pairs(alpha=float(c[1:]), beta=b)


def ssim_t(a, b, L=400.0):
    """skimage.metrics.structural_similarity(data_range=400) on images clipped to [-160, 240] HU, per slice (B,H,W)."""
    a = a.double().clamp(-160, 240); b = b.double().clamp(-160, 240)
    f = lambda t: F.avg_pool2d(t[:, None], 7, stride=1)[:, 0]
    cov = 49 / 48
    ux, uy = f(a), f(b); vx = cov * (f(a * a) - ux * ux); vy = cov * (f(b * b) - uy * uy); vxy = cov * (f(a * b) - ux * uy)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    return S.mean(dim=(1, 2))


def roi_list(es):
    rng = np.random.default_rng(zlib.crc32(f"roi2d_{es.p}".encode())); out = []
    mk = es.mask.cpu().numpy()
    for i in range(mk.shape[0]):
        got = tries = 0
        while got < ROI_PER_SLICE and tries < 400:
            tries += 1; y = int(rng.integers(0, 512 - ROI_N)); x = int(rng.integers(0, 512 - ROI_N))
            if mk[i, y:y + ROI_N, x:x + ROI_N].all():
                out.append((i, y, x)); got += 1
    return out


def detrend_nps(rois, d):
    """Same as ctphys.nps2d(order=2) + radial_on_grid + variance (copied to keep Stage 2d self-contained)."""
    ny = nx = rois.shape[1]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float); xx = (xx - xx.mean()) / nx; yy = (yy - yy.mean()) / ny
    A = np.stack([np.ones_like(xx), xx, yy, xx ** 2, xx * yy, yy ** 2], -1).reshape(-1, 6)
    R = rois.reshape(len(rois), -1).T.astype(float)
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    det = (R - A @ coef).T.reshape(rois.shape)
    P = np.fft.fftshift((np.abs(np.fft.fft2(det)) ** 2).mean(0)) * d * d / (nx * ny)
    df_ = 1.0 / (nx * d); var = float(P.sum() * df_ * df_)
    f = np.fft.fftshift(np.fft.fftfreq(nx, d)); fr = np.hypot(*np.meshgrid(f, f))
    edges = np.arange(nx // 2 + 1) * df_; idx = np.digitize(fr.ravel(), edges) - 1; ok = (idx >= 0) & (idx < nx // 2)
    s = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=nx // 2); c = np.bincount(idx[ok], minlength=nx // 2)
    prof = np.where(c > 0, s / np.maximum(c, 1), np.nan); fc = (edges[:-1] + edges[1:]) / 2; g = np.isfinite(prof)
    return var, np.interp(FG, fc[g], prof[g], left=np.nan, right=np.nan)


@torch.no_grad()
def eval_model(model, es, rois):
    model.eval(); rows = []
    by_slice = {}
    for (i, y, x) in rois:
        by_slice.setdefault(i, []).append((y, x))
    for (c, b) in TEST_CONDS:
        xs, ys = cond_pairs(es, c, b)
        se = se_in = 0.0; n = 0; ss = []; res = []
        for i0 in range(0, xs.shape[0], EVAL_BS):
            xb, yb = xs[i0:i0 + EVAL_BS], ys[i0:i0 + EVAL_BS]
            ob = model(xb[:, None] / 1000.0)[:, 0] * 1000.0
            for j in range(ob.shape[0]):
                m = es.mask[i0 + j]
                dd = (ob[j] - yb[j])[m]; se += float((dd.double() ** 2).sum()); n += int(dd.numel())
                di = (xb[j] - yb[j])[m]; se_in += float((di.double() ** 2).sum())
                for (y, x) in by_slice.get(i0 + j, []):
                    res.append((ob[j] - yb[j])[y:y + ROI_N, x:x + ROI_N])
            ss.append(ssim_t(ob, yb).cpu().numpy())
        rv, prof = detrend_nps(torch.stack(res).cpu().numpy(), PX[es.p])
        rows.append(dict(cond=c, beta=b, rmse=math.sqrt(se / n), rmse_input=math.sqrt(se_in / n),
                         ssim=float(np.concatenate(ss).mean()), res_var=rv, res_nps=[round(float(v), 6) for v in prof], n_px=n, n_roi=len(res)))
    return rows


def model_jobs():
    J = [j for j in all_jobs()]
    assert len(J) == 80
    return sorted(J, key=lambda j: (j["fold"], j["arch"], j["alpha"], j["beta"]))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "evalworker":
    gpu, deadline = int(sys.argv[2]), float(sys.argv[3])
    ES, ROIS = {}, {}
    for j in model_jobs():
        tag = j["tag"]; outf = f"{EOUT}/eval_{tag}.json"
        if os.path.exists(outf):
            continue
        try:
            fdesc = os.open(f"{EOUT}/{tag}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fdesc, str(gpu).encode()); os.close(fdesc)
        except FileExistsError:
            continue
        if time.time() > deadline:
            print(f"[gpu{gpu}] deadline reached before {tag}", flush=True); os.remove(f"{EOUT}/{tag}.claim"); break
        t0 = time.time()
        rec = json.load(open(f"{MODEL_DIR}/{tag}.json"))
        model = ARCHS[j["arch"]]().to(DEV)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/{tag}.pt", map_location=DEV)); model.eval()
        # integrity: the loaded checkpoint must reproduce its recorded best validation RMSE
        vrm = rmse_hu(model, EvalSet(VAL[j["fold"]], "val"), alpha=j["alpha"], beta=j["beta"]); model.eval()
        rel = abs(vrm - rec["best_val_rmse"]) / rec["best_val_rmse"]
        assert rel < 2e-3, (tag, vrm, rec["best_val_rmse"])
        rows = []
        for p in FOLDS[j["fold"]]:
            if p not in ES:
                for q in list(ES):
                    if q not in FOLDS[j["fold"]]:
                        del ES[q]
                torch.cuda.empty_cache(); ES[p] = EvalSet(p, "test"); ROIS[p] = roi_list(ES[p])
            for r in eval_model(model, ES[p], ROIS[p]):
                r.update(arch=j["arch"], fold=j["fold"], train_alpha=j["alpha"], train_beta=j["beta"], patient=p, tag=tag); rows.append(r)
        json.dump({"tag": tag, "val_recheck": vrm, "val_recorded": rec["best_val_rmse"], "val_rel_diff": rel,
                   "best_it": rec["best_it"], "amp_train": rec["amp"], "rows": rows, "sec": time.time() - t0}, open(outf, "w"))
        os.remove(f"{EOUT}/{tag}.claim")
        print(f"[gpu{gpu}] EVAL {tag} val {vrm:.3f} (rec {rec['best_val_rmse']:.3f}) "
              f"rmse a1.0/b1 {[round(r['rmse'], 2) for r in rows if r['cond'] == 'a1.0' and r['beta'] == 1.0]} "
              f"B30real/b1 {[round(r['rmse'], 2) for r in rows if r['cond'] == 'B30real' and r['beta'] == 1.0]} {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
