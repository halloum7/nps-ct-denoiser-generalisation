"""End-to-end logic test of stage1b with a synthetic Mayo-like dataset of KNOWN truth:
10 patients, different pixel spacings, B30 = physical-frequency linear filter of D45."""
import numpy as np, pandas as pd, os, sys, tempfile, json, time, zlib
import matplotlib; matplotlib.use("Agg")
from scipy import ndimage, stats
import matplotlib.pyplot as plt
exec(open('ctphys.py').read())
src = open('stage1b_main.py').read()
NONLIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0          # 1.0 = linear truth
n = 256
PXS = dict(zip([f"L{i:03d}" for i in range(10)], np.linspace(0.62, 0.86, 10)))
y, x = np.mgrid[0:n, 0:n]
body = ((x - n/2)/110)**2 + ((y - n/2)/90)**2 < 1
lung = ((x - n/2 - 50)/30)**2 + ((y - n/2)/35)**2 < 1
Tt = lambda f: np.exp(-(f/0.30)**2) * (1 + 0.3*np.exp(-((f-0.15)/0.08)**2))   # true D45->B30 in 1/mm
cache = {}
def make(pt, z):
    if (pt, z) in cache: return cache[(pt, z)]
    d = PXS[pt]; rng = np.random.default_rng(abs(hash((pt, z))) % 2**32)
    f = np.fft.fftfreq(n, d); FR = np.hypot(*np.meshgrid(f, f))
    flt = lambda a, H: np.real(np.fft.ifft2(np.fft.fft2(a) * H))
    Hs = FR * np.exp(-(FR/0.5)**2); Hs /= Hs.max()
    anat = np.where(body, 40.0, -1000.0); anat[lung] = -800
    anat += body * flt(rng.normal(0, 1, (n, n)), 1/(1+(FR/0.04)**2)) * 60
    nfd = flt(rng.normal(0, 1, (n, n)), Hs) * 30
    nqd = nfd + flt(rng.normal(0, 1, (n, n)), Hs) * 30*np.sqrt(3)
    out = {}
    for dose, nn in [("FD", nfd), ("QD", nqd)]:
        D = anat + nn; B = flt(D, Tt(FR))
        if NONLIN != 1.0: B = np.sign(B) * np.abs(B)**NONLIN
        CL = -1e9 if os.environ.get("NOCLIP") else -1024
        out[("D45", dose)] = np.maximum(D, CL).astype(np.float32)
        out[("B30", dose)] = np.maximum(B, CL).astype(np.float32)
    cache[(pt, z)] = out; return out
rows = [dict(patient=p, thick=t, kernel=k, dose=d, zr=float(z)) for p in PXS for t in ["1mm"]
        for k in ["B30", "D45"] for d in ["FD", "QD"] for z in range(26)]
inv = pd.DataFrame(rows)
PX = {(p, "1mm"): v for p, v in PXS.items()}
def load(pt, th, k, d, z):
    hu = make(pt, z)[(k, d)]; return hu, int((hu <= -1024).sum()), 0
a = src.index("def tissue_mask"); b = src.index('if __name__ == "__main__"')
exec(src[a:b])
OUT = tempfile.mkdtemp(); t0 = T0 = time.time()
res = run_thickness("1mm", inv, PX, load, PATIENTS, OUT, results)["1mm"]
print(json.dumps({k: res[k] for k in ["pixel_spacing_range", "transfer_spread_physical_vs_pixel"]}, default=float))
for tag in ["img", "noise", "roi"]:
    r = res[tag]; print(tag, {k: (round(v["median"], 4) if isinstance(v, dict) else v) for k, v in r.items()})
print("runtime %.0fs" % (time.time() - t0), sorted(os.listdir(OUT)))
pp = pd.read_csv(f"{OUT}/per_patient_1mm.csv"); print(pp[["patient","px","n_rois","fmean_B30","fmean_D45","W1_D45_vs_B30"]].round(4).to_string())
if NONLIN == 1.0:
    assert res["ACCEPT"] and res["img"]["n_C1"] == 10 and res["roi"]["n_C2"] == 10
    assert 0.97 < res["roi"]["var_ratio"]["min"] and res["roi"]["var_ratio"]["max"] < 1.03, res["roi"]["var_ratio"]
    assert res["roi"]["W1_emu_vs_B30"]["max"] < 0.1 * pp.W1_D45_vs_B30.min()
    sp, sx = res["transfer_spread_physical_vs_pixel"]; assert sp < 0.5 * sx, (sp, sx)
    assert (pp.fmean_B30 < pp.fmean_D45).all()
    print("LINEAR-TRUTH TESTS PASSED")
else:
    print("NONLINEAR CONTROL: ACCEPT =", res["ACCEPT"], "img C1 =", res["img"]["n_C1"], "roi C2 =", res["roi"]["n_C2"], "roi var", res["roi"]["var_ratio"])
