import numpy as np, pandas as pd, os, tempfile, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
exec(open('ctphys.py').read())
src=open('stage1_main.py').read()
a=src.index("def body_mask"); b=src.index("PATIENTS = sorted")
exec(src[a:b])                                    # body_mask, seed_of, analyze_thickness
rng=np.random.default_rng(1); n=256; px=0.7
f=np.fft.fftfreq(n,px); FX,FY=np.meshgrid(f,f); FR=np.hypot(FX,FY)
Hs=FR*np.exp(-(FR/0.45)**2); Hs/=Hs.max()            # 'sharp' noise texture
Tt=np.exp(-(FR/0.22)**2)                              # true D45 -> B30 transfer
filt=lambda x,H: np.real(np.fft.ifft2(np.fft.fft2(x)*H))
y,x=np.mgrid[0:n,0:n]; body=((x-n/2)/100)**2+((y-n/2)/80)**2<1
store={}
for p in range(10):
    for z in range(6):
        anat=np.where(body,40.0,-1000.0)+body*filt(rng.normal(0,1,(n,n)),1/(1+(FR/0.04)**2))*80
        nfd=filt(rng.normal(0,1,(n,n)),Hs)*40; nqd=nfd+filt(rng.normal(0,1,(n,n)),Hs)*40*np.sqrt(3)
        D45={"FD":anat+nfd,"QD":anat+nqd}
        store[(f"L{p:03d}",float(z))]={("D45","FD"):D45["FD"],("D45","QD"):D45["QD"],("B30","FD"):filt(D45["FD"],Tt),("B30","QD"):filt(D45["QD"],Tt)}
P=sorted({k[0] for k in store}); out=tempfile.mkdtemp()
res=analyze_thickness(store,px,"1mm",P[:5],P[5:],out,K_ROI=48,N_ROI=10)
import json; print(json.dumps(res,indent=1,default=float))
e=res["1mm_emulation"]
assert e["median_ratio_emulated_FD"]<0.05 and e["median_ratio_unfiltered_FD"]>0.5, "emulation logic failed"
assert e["W1_emulatedNoise_vs_realB30"]<0.2*e["W1_D45noise_vs_realB30"]
assert res["1mm_B30_nps"]["f_mean"]<res["1mm_D45_nps"]["f_mean"]
print("files:",sorted(os.listdir(out)))
# negative control: make B30 NON-linear (clip + gamma) -> ratio must rise
for k in store:
    for d in ["FD","QD"]:
        store[k][("B30",d)]=np.sign(store[k][("B30",d)])*np.abs(store[k][("B30",d)])**0.97
res2=analyze_thickness(store,px,"1mm",P[:5],P[5:],out,K_ROI=48,N_ROI=10,make_fig=False)
print("negative control ratio FD:",res2["1mm_emulation"]["median_ratio_emulated_FD"])
assert res2["1mm_emulation"]["median_ratio_emulated_FD"]>e["median_ratio_emulated_FD"]*3
print("STAGE1 LOGIC TESTS PASSED")
