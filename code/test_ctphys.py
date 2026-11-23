import numpy as np
from ctphys import *
rng=np.random.default_rng(0); dx=0.7; N=64
# 1. white noise: flat NPS = s2*dx^2, integral = variance
s=12.0; rois=rng.normal(0,s,(400,N,N))
P=nps2d(rois,dx,order=0); sm=nps_summary(P,dx)
print("T1 white: var_int=%.2f expected=%.2f | mean NPS=%.3f expected=%.3f"%(sm['variance'],s**2,P.mean(),s**2*dx*dx))
assert abs(sm['variance']/(s**2*(1-1/N**2))-1)<0.02
# 2. coloured noise with known |H|^2 (ramp*gaussian, CT-like band-pass)
f=np.fft.fftfreq(N,dx); FX,FY=np.meshgrid(f,f); FR=np.hypot(FX,FY)
H=FR*np.exp(-(FR/0.35)**2); H/=H.max()
rois=np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0,s,(600,N,N)),axes=(1,2))*H,axes=(1,2)))
P=nps2d(rois,dx,order=0); fr,pr=radial_profile(P,dx)
_,pa=radial_profile(np.fft.fftshift(s**2*dx*dx*H**2),dx)
m=(fr>0.05)&(fr<0.6); rel=np.abs(pr[m]/pa[m]-1)
print("T2 coloured: median rel err=%.4f max=%.4f"%(np.median(rel),rel.max()))
assert np.median(rel)<0.03
sm=nps_summary(P,dx); print("   f_peak=%.3f analytic peak=%.3f"%(sm['f_peak'],0.35/np.sqrt(2)))
# 2b. detrend order 2 must not bias a flat spectrum except near DC
rois=rng.normal(0,s,(400,N,N))+np.linspace(0,200,N)[None,None,:]   # strong ramp
P2=nps2d(rois,dx,order=2); fr,pr2=radial_profile(P2,dx)
print("T2b ramp removed: NPS(f>0.1)/expected = %.3f"%(np.nanmean(pr2[fr>0.1])/(s*s*dx*dx)))
assert abs(np.nanmean(pr2[fr>0.1])/(s*s*dx*dx)-1)<0.03
# 3. transfer function recovery (kernel emulation logic)
n=128; d=0.7
f=np.fft.fftfreq(n,d); FX,FY=np.meshgrid(f,f); FR=np.hypot(FX,FY)
Tt=lambda q: np.exp(-(q/0.30)**2)*(1+0.5*q)            # 'sharp->soft' smooth transfer
srcs=[np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0,1,(n,n)))*(1/(1+(FR/0.05)**2))))*300+rng.normal(0,20,(n,n)) for _ in range(40)]
dsts=[np.real(np.fft.ifft2(np.fft.fft2(x)*Tt(FR))) for x in srcs]
fc,T=estimate_transfer(srcs[:30],dsts[:30],d)
ok=fc<0.65; err=np.abs(T[ok]-Tt(fc[ok]))
print("T3 transfer: max |T-Ttrue| on f<0.65 = %.4f (note ring-average of a radial fn)"%err.max())
emu=[apply_radial_filter(x,fc,T,d) for x in srcs[30:]]
r=[np.sqrt(np.mean((a-b)**2))/np.std(b) for a,b in zip(emu,dsts[30:])]
print("   held-out relative RMSE = %.4f"%np.mean(r)); assert np.mean(r)<0.02
# 4. distance properties
fc2,pa=radial_profile(np.fft.fftshift(H**2),dx)
print("T4 dist(self)=%.4f dist(scaled x10)=%.4f"%(nnps_distance(pa,pa,fc2),nnps_distance(pa,10*pa,fc2)))
fN=np.fft.fftfreq(N,dx); FXN,FYN=np.meshgrid(fN,fN); FRN=np.hypot(FXN,FYN); H2=FRN*np.exp(-(FRN/0.20)**2); _,pb=radial_profile(np.fft.fftshift(H2**2),dx)
print("   dist(peak .25 vs .14)=%.4f 1/mm"%nnps_distance(pa,pb,fc2))
assert nnps_distance(pa,10*pa,fc2)<1e-9 and nnps_distance(pa,pb,fc2)>0.05
print("ALL TESTS PASSED")
