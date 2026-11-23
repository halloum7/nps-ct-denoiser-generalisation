# ---- Stage 2d driver (one notebook cell). LIB = Stage-2c library (unchanged), EVAL = stage2d_eval.py,
# STATS / POST = stage2d_stats.py / stage2d_post.py (tested offline on synthetic data with known truth).
import os, sys, glob, json, time, shutil, subprocess
T0 = time.time()
open("/kaggle/working/s2d.py", "w").write(LIB + "\n" + EVAL)
open("/kaggle/working/stage2d_stats.py", "w").write(STATS); open("/kaggle/working/stage2d_post.py", "w").write(POST)
sys.path.insert(0, "/kaggle/working")
exec(LIB + "\n" + EVAL)
for f in glob.glob("/kaggle/input/**/stage2d/eval_*.json", recursive=True):
    dst = os.path.join(EOUT, os.path.basename(f))
    if not os.path.exists(dst): shutil.copy(f, dst)
# ---------------- smoke tests (GPU 0), all must pass before any long computation
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
tags = {j["tag"] for j in all_jobs()}
assert all(os.path.exists(f"{MODEL_DIR}/{t}.pt") and os.path.exists(f"{MODEL_DIR}/{t}.json") for t in tags), "missing model files"
def kf_np(img, d, Th, alpha=1.0, s=1.0):                                  # Stage-2a numpy operator (copied verbatim)
    n = img.shape[0]; f = np.fft.fftfreq(n, d); fr = np.hypot(*np.meshgrid(f, f)) / s
    H = np.interp(fr, FC, Th, right=Th[-1]) ** alpha
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))
def bw_np(img, d, fc, order=8):
    n = img.shape[0]; f = np.fft.fftfreq(n, d); H = 1.0 / (1.0 + (np.hypot(*np.meshgrid(f, f)) / fc) ** order)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))
es = EvalSet("L067", "test"); d = PX["L067"]; fd0 = es.fd[0].cpu().numpy().astype(float); qd0 = es.qd[0].cpu().numpy().astype(float)
sm = {}
_, y = cond_pairs(es, "off0.8", 1.0); sm["off0.8_maxdiff_HU"] = float(np.abs(y[0].cpu().numpy() - kf_np(fd0, d, TH, 1.0, 0.8)).max())
_, y = cond_pairs(es, "offBW", 1.0); sm["offBW_maxdiff_HU"] = float(np.abs(y[0].cpu().numpy() - bw_np(fd0, d, FC_BW)).max())
x, _ = cond_pairs(es, "a0.5", 0.5); sm["a0.5_b0.5_x_maxdiff_HU"] = float(np.abs(x[0].cpu().numpy() - kf_np(fd0 + 0.5 * (qd0 - fd0), d, TH, 0.5)).max())
x, y = cond_pairs(es, "B30real", 0.5)
sm["B30real_maxdiff_HU"] = float(max((y - es.bfd).abs().max(), (x - (es.bfd + 0.5 * (es.bqd - es.bfd))).abs().max()))
a, b = es.bfd[0], es.bqd[0]
sm["ssim_torch"] = float(ssim_t(a[None], b[None])[0]); sm["ssim_skimage"] = float(sk_ssim(np.clip(a.cpu().numpy().astype(float), -160, 240), np.clip(b.cpu().numpy().astype(float), -160, 240), data_range=400))
sm["n_rois_L067"] = len(roi_list(es)); sm["FC_BW"] = FC_BW; sm["FG"] = [float(FG[0]), float(FG[-1]), len(FG)]
print("SMOKE_JSON=" + json.dumps(sm), flush=True)
assert sm["off0.8_maxdiff_HU"] < 0.01 and sm["offBW_maxdiff_HU"] < 0.01 and sm["a0.5_b0.5_x_maxdiff_HU"] < 0.01 and sm["B30real_maxdiff_HU"] == 0
assert abs(sm["ssim_torch"] - sm["ssim_skimage"]) < 1e-6 and sm["n_rois_L067"] >= 150
del es; torch.cuda.empty_cache()
# ---------------- inference on 2 GPUs
deadline = T0 + 10.5 * 3600
logs = [open(f"/kaggle/working/eval{g}.log", "w") for g in (0, 1)]
P = [subprocess.Popen([sys.executable, "/kaggle/working/s2d.py", "evalworker", str(g), str(deadline)],
                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}, stdout=logs[g], stderr=subprocess.STDOUT) for g in (0, 1)]
for p in P: p.wait()
for g in (0, 1):
    logs[g].close(); print(open(f"/kaggle/working/eval{g}.log").read()[-30000:])
n_eval = len(glob.glob(f"{EOUT}/eval_*.json"))
print("EVAL_STATUS=" + json.dumps({"n_eval": n_eval, "return_codes": [p.returncode for p in P], "hours": (time.time() - T0) / 3600}), flush=True)
assert n_eval == 80
# ---------------- statistics (pre-registered order: V0, H1, H2, H3, H4) + independent LMM check
import stage2d_post as POSTM
res, met, Lw, pm = POSTM.run(EOUT, PM_PATH, B=10000)
res["H5_from_stage2a"] = R2A["H5"]
try:
    import statsmodels.formula.api as smf
    dd = Lw.assign(sgn=np.sign(Lw.test_alpha - Lw.train_alpha), dn=(Lw.arch == "DnCNN").astype(float))
    fit = smf.mixedlm("L ~ W1 + dlogvar + sgn + dn", dd, groups=dd["patient"]).fit(reml=True)
    res["H2"]["statsmodels_check"] = {k: float(v) for k, v in fit.params.items()}
except Exception as e:
    res["H2"]["statsmodels_check"] = "failed: " + repr(e)[:200]
json.dump(res, open(f"{EOUT}/stage2d_results.json", "w"), indent=1, default=float)
print("STAGE2D_JSON=" + json.dumps(res, default=float), flush=True)
print("METRICS_B64=" + POSTM.b64gz(met, ["arch", "fold", "train_alpha", "train_beta", "patient", "cond", "beta", "rmse", "rmse_input", "ssim", "res_var"]))
print("PM_B64=" + POSTM.b64gz(pm, list(pm.columns)))
print("STAGE2D_DONE hours=%.2f" % ((time.time() - T0) / 3600))
