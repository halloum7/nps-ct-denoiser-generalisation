"""Synthetic-truth test of stage2a analysis(): 10 patients, mixed pixel spacing, B30 = physical
linear filter of D45 then clipped at -1024 (like real data), adjacent slices = independent noise."""
import numpy as np, pandas as pd, os, sys, tempfile, json, time, zlib
exec(open('ctphys.py').read())
src = open('stage2a_main.py').read(); exec(src[:src.index('if __name__ == "__main__"')])
n = 256
PATIENTS = [f"L{i:03d}" for i in range(10)]
PX = dict(zip(PATIENTS, np.linspace(0.62, 0.86, 10)))
y, x = np.mgrid[0:n, 0:n]
body = ((x - n/2)/110)**2 + ((y - n/2)/90)**2 < 1
lung = ((x - n/2 - 50)/30)**2 + ((y - n/2)/35)**2 < 1
Tt = lambda f: np.exp(-(f/0.30)**2) * (1 + 0.3*np.exp(-((f-0.15)/0.08)**2))
def flt(a, d, H):
    f = np.fft.fftfreq(n, d); FR = np.hypot(*np.meshgrid(f, f)); return np.real(np.fft.ifft2(np.fft.fft2(a) * H(FR)))
store = {}
for p in PATIENTS:
    d = PX[p]; rng = np.random.default_rng(zlib.crc32(p.encode()))
    Hs = lambda FR: FR * np.exp(-(FR/0.5)**2) / (0.5/np.sqrt(2)*np.exp(-0.5))
    for z in range(20):
        anat = np.where(body, 40.0, -1000.0); anat[lung] = -800
        anat += body * flt(rng.normal(0, 1, (n, n)), d, lambda FR: 1/(1+(FR/0.04)**2)) * 60
        nfd = flt(rng.normal(0, 1, (n, n)), d, Hs) * 30
        nqd = nfd + flt(rng.normal(0, 1, (n, n)), d, Hs) * 30*np.sqrt(3)
        nqd_n = flt(rng.normal(0, 1, (n, n)), d, Hs) * 30 + flt(rng.normal(0, 1, (n, n)), d, Hs) * 30*np.sqrt(3)
        anat_n = anat + body * flt(rng.normal(0, 1, (n, n)), d, lambda FR: 1/(1+(FR/0.02)**2)) * 5   # small anatomical change
        D = {"FD": anat + nfd, "QD": anat + nqd, "QDn": anat_n + nqd_n}
        v = {}
        for k_, a_ in D.items():
            v[f"D45_{k_}"] = np.maximum(a_, -1024); v[f"B30_{k_}"] = np.maximum(flt(a_, d, Tt), -1024)
        store[(p, float(z))] = v
OUT = tempfile.mkdtemp(); t0 = time.time()
FOLDS = {1: PATIENTS[0:2], 2: PATIENTS[2:4], 3: PATIENTS[4:6], 4: PATIENTS[6:8], 5: PATIENTS[8:10]}
VAL = {1: PATIENTS[3], 2: PATIENTS[5], 3: PATIENTS[7], 4: PATIENTS[9], 5: PATIENTS[1]}
res, Th = analysis(store, PX, PATIENTS, OUT)
print(json.dumps({k: res[k] for k in ["check1", "check2", "check3", "H5", "B30real_W1_to_alpha1", "FG_max", "butterworth_fc"]}, indent=1, default=float))
f = (np.arange(len(Th)) + 0.5) * DF; m = f < 0.6
print("max |T_h - T_true| f<0.6:", float(np.abs(Th[m] - Tt(f[m])).max()), " runtime %.0fs" % (time.time() - t0))
pm = pd.read_csv(f"{OUT}/pair_matrix.csv"); print("pair rows", len(pm), pm.groupby("test_cond").W1_ref.median().round(4).to_dict())
assert res["check1"]["PASS"] and res["check2"]["PASS"], "family checks must pass under linear truth"
assert res["B30real_W1_to_alpha1"] < 0.005
assert np.abs(Th[m] - Tt(f[m])).max() < 0.02, "hybrid transfer error"
assert 0.97 < res["check1"]["median_var_ratio"] < 1.03
assert res["check3"]["PASS"], res["check3"]
print("fmean family", res["fmean_family"], "BW fc", res["butterworth_fc"])
print("STAGE2A SYNTHETIC TESTS PASSED")
