# ============================ STAGE 2b (GPU) : smoke tests + pilot (fold 1, alpha=1, beta=1, 3 seeds x 2 archs)
# Protocol (pre-registered doc 2026-11-14): MSE, Adam 1e-4 -> 1e-5 cosine, batch 32 patches 64x64, HU/1000,
# flips/rot90 only, on-the-fly kernel emulation K_alpha = T_h^alpha in physical frequency (patient pixel size),
# checkpoint = best validation-patient RMSE every 2000 it. Pilot fixes N (rule below), seed variability, V0 first look.
# N rule (fixed before the pilot): N = smallest multiple of 10k at which the seed-mean validation RMSE improves by
# < 0.5 % over the next 10k iterations; capped at N_MAX.
import os, json, time, glob, math, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/kaggle/working/stage2b"; os.makedirs(OUT, exist_ok=True)
MAN = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
assert len(MAN) == 1, MAN
CACHE = os.path.dirname(MAN[0]); M = json.load(open(MAN[0]))
FC = np.asarray(M["T_h"]["f"]); TH = np.asarray(M["T_h"]["T"]); PX = M["px"]
FOLDS = {int(k): v for k, v in M["folds"].items()}; VAL = {int(k): v for k, v in M["val"].items()}
PATIENTS = sorted(PX)
PATCH, BATCH, SL_PER_IT, LR0, LR1, EVAL_EVERY = 64, 32, 8, 1e-4, 1e-5, 2000

# ---------------------------------------------------------------- models
class REDCNN(nn.Module):
    """Chen et al. 2017: 5 conv + 5 deconv, 96 filters 5x5, stride 1, no padding, 3 shortcuts (reference impl.).
    No ReLU before the summations; no final ReLU (inputs are HU/1000, can be negative)."""
    def __init__(self, nf=96):
        super().__init__()
        self.c = nn.ModuleList([nn.Conv2d(1 if i == 0 else nf, nf, 5) for i in range(5)])
        self.d = nn.ModuleList([nn.ConvTranspose2d(nf, 1 if i == 4 else nf, 5) for i in range(5)])
        for m in list(self.c) + list(self.d):
            nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
    def forward(self, x):
        r1 = x
        o = F.relu(self.c[0](x)); o = F.relu(self.c[1](o)); r2 = o
        o = F.relu(self.c[2](o)); o = F.relu(self.c[3](o)); r3 = o
        o = F.relu(self.c[4](o))
        o = self.d[0](o) + r3
        o = self.d[1](F.relu(o)); o = self.d[2](F.relu(o)) + r2
        o = self.d[3](F.relu(o)); o = self.d[4](F.relu(o)) + r1
        return o

class DnCNN(nn.Module):
    """Zhang et al. 2017: 17 layers, 64 filters 3x3, BatchNorm, residual output (x - noise)."""
    def __init__(self, depth=17, nf=64):
        super().__init__()
        L = [nn.Conv2d(1, nf, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            L += [nn.Conv2d(nf, nf, 3, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True)]
        L += [nn.Conv2d(nf, 1, 3, padding=1)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return x - self.net(x)

ARCHS = {"REDCNN": REDCNN, "DnCNN": DnCNN}

# ---------------------------------------------------------------- data
def kernel_rfft(p, alpha, n=512, s=1.0, bw_fc=None):
    """Same operator as stage2a.kfilter / bwfilter, on the rfft2 grid (real input)."""
    d = PX[p]; fy = np.fft.fftfreq(n, d); fx = np.fft.rfftfreq(n, d)
    fr = np.hypot(*np.meshgrid(fx, fy))
    if bw_fc is not None:
        H = 1.0 / (1.0 + (fr / bw_fc) ** 8)
    else:
        H = np.interp(fr / s, FC, TH, right=TH[-1]) ** alpha
    return torch.tensor(H, dtype=torch.float32, device=DEV)

def apply_k(img, H):
    return torch.fft.irfft2(torch.fft.rfft2(img) * H, s=img.shape[-2:])

def load_patient(p, test_b30=False):
    e = M["patients"][p]; out = {}
    for nm in ["D45_FD", "D45_QD"] + (["B30_FD_test", "B30_QD_test"] if test_b30 else []):
        fn = f"{CACHE}/{p}_{nm}.npy"
        assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == e["sha256"][nm], f"SHA mismatch {fn}"
        out[nm] = np.load(fn)
    return out

def tissue_mask(img, erode_px=4):
    m = img > -300
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    body = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    body = ndimage.binary_fill_holes(body)
    tis = body & (ndimage.uniform_filter(m.astype(float), 9) > 0.999)
    return ndimage.binary_erosion(tis, iterations=erode_px)

class Sampler:
    """Training patches for one (fold, alpha, beta): slices of the training patients held on GPU as int16."""
    def __init__(self, pats, alpha, beta, seed):
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.fd, self.qd, self.H, self.idx = [], [], [], []
        for k, p in enumerate(pats):
            d = load_patient(p)
            self.fd.append(torch.tensor(d["D45_FD"], device=DEV)); self.qd.append(torch.tensor(d["D45_QD"], device=DEV))
            self.H.append(kernel_rfft(p, alpha)); self.idx += [(k, i) for i in range(len(d["D45_FD"]))]
        self.beta = beta
    def batch(self):
        X, Y = [], []
        pick = torch.randint(len(self.idx), (SL_PER_IT,), generator=self.g).tolist()
        per = BATCH // SL_PER_IT
        for j in pick:
            k, i = self.idx[j]
            fd = self.fd[k][i].float(); qd = self.qd[k][i].float()
            x = apply_k(fd + self.beta * (qd - fd), self.H[k]); y = apply_k(fd, self.H[k])
            got = 0; tries = 0
            while got < per and tries < 64:
                tries += 1
                r = torch.randint(512 - PATCH, (2,), generator=self.g).tolist()
                yp = y[r[0]:r[0] + PATCH, r[1]:r[1] + PATCH]
                if (yp > -300).float().mean() >= 0.5 or tries > 48:
                    xp = x[r[0]:r[0] + PATCH, r[1]:r[1] + PATCH]
                    t = int(torch.randint(8, (1,), generator=self.g))
                    if t & 4: xp, yp = xp.flip(-1), yp.flip(-1)
                    xp, yp = torch.rot90(xp, t & 3), torch.rot90(yp, t & 3)
                    X.append(xp); Y.append(yp); got += 1
        return torch.stack(X)[:, None] / 1000.0, torch.stack(Y)[:, None] / 1000.0

class EvalSet:
    """Fixed slices of one patient (val or test) with tissue masks from raw D45 FD (alpha-independent)."""
    def __init__(self, p, which):
        d = load_patient(p, test_b30=(which == "test")); e = M["patients"][p]
        ii = e["val_idx"] if which == "val" else e["test_idx"]
        self.p = p; self.fd = torch.tensor(d["D45_FD"][ii], device=DEV).float(); self.qd = torch.tensor(d["D45_QD"][ii], device=DEV).float()
        self.mask = torch.tensor(np.stack([tissue_mask(d["D45_FD"][i].astype(np.float32)) for i in ii]), device=DEV)
        if which == "test":
            self.bfd = torch.tensor(d["B30_FD_test"], device=DEV).float(); self.bqd = torch.tensor(d["B30_QD_test"], device=DEV).float()
    def pairs(self, alpha=None, beta=1.0, real_b30=False, s=1.0, bw_fc=None):
        if real_b30:
            return self.bfd + beta * (self.bqd - self.bfd), self.bfd
        H = kernel_rfft(self.p, alpha if alpha is not None else 1.0, s=s, bw_fc=bw_fc)
        return apply_k(self.fd + beta * (self.qd - self.fd), H), apply_k(self.fd, H)

@torch.no_grad()
def rmse_hu(model, es, **kw):
    model.eval(); x, y = es.pairs(**kw); se, n = 0.0, 0
    for i in range(len(x)):
        o = model(x[i][None, None] / 1000.0)[0, 0] * 1000.0
        dd = (o - y[i])[es.mask[i]]; se += float((dd ** 2).sum()); n += int(dd.numel())
    model.train(); return math.sqrt(se / n)

def train(arch, fold, alpha, beta, seed, n_iter, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    test = FOLDS[fold]; val = VAL[fold]; pats = [p for p in PATIENTS if p not in test and p != val]
    smp = Sampler(pats, alpha, beta, seed); ev = EvalSet(val, "val")
    model = ARCHS[arch]().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=LR1)
    curve, best, t0 = [], (1e9, 0), time.time()
    for it in range(1, n_iter + 1):
        x, y = smp.batch(); loss = F.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss at it {it}")
        if it % EVAL_EVERY == 0:
            v = rmse_hu(model, ev, alpha=alpha, beta=beta); curve.append((it, v, float(loss)))
            if v < best[0]:
                best = (v, it); torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    sec = time.time() - t0
    del smp; torch.cuda.empty_cache()
    return {"arch": arch, "fold": fold, "alpha": alpha, "beta": beta, "seed": seed, "n_iter": n_iter,
            "best_val_rmse": best[0], "best_it": best[1], "curve": curve, "train_seconds": sec, "sec_per_it": sec / n_iter}

class Identity(nn.Module):
    def forward(self, x): return x

# ---------------------------------------------------------------- smoke tests (must all pass before the pilot)
def smoke():
    rep = {"device": str(DEV), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__}
    for nm, A in ARCHS.items():
        m = A().to(DEV)
        for s in (64, 512):
            o = m(torch.zeros(2, 1, s, s, device=DEV)); assert o.shape == (2, 1, s, s), (nm, o.shape)
        rep[f"{nm}_params"] = sum(q.numel() for q in m.parameters())
    # GPU filter == numpy reference (stage2a.kfilter, full fft2) on a real slice
    p = PATIENTS[0]; d = load_patient(p); img = d["D45_QD"][len(d["D45_QD"]) // 2].astype(np.float64)
    f = np.fft.fftfreq(512, PX[p]); H = np.interp(np.hypot(*np.meshgrid(f, f)), FC, TH, right=TH[-1]) ** 0.5
    ref = np.real(np.fft.ifft2(np.fft.fft2(img) * H))
    gpu = apply_k(torch.tensor(img, dtype=torch.float32, device=DEV), kernel_rfft(p, 0.5)).cpu().numpy()
    rep["filter_max_abs_diff_HU"] = float(np.abs(gpu - ref).max()); assert rep["filter_max_abs_diff_HU"] < 0.05
    # identity at alpha = 0
    g0 = apply_k(torch.tensor(img, dtype=torch.float32, device=DEV), kernel_rfft(p, 0.0)).cpu().numpy()
    rep["alpha0_identity_max_abs_HU"] = float(np.abs(g0 - img).max()); assert rep["alpha0_identity_max_abs_HU"] < 0.05
    # sampler + one-batch overfit for each architecture (loss must fall by > 50 % in 300 steps)
    smp = Sampler([PATIENTS[1]], 1.0, 1.0, 0); x, y = smp.batch()
    assert x.shape == (BATCH, 1, PATCH, PATCH) and y.shape == x.shape
    rep["batch_noise_sd_HU"] = float(((x - y) * 1000).std())
    for nm, A in ARCHS.items():
        torch.manual_seed(0); m = A().to(DEV); o = torch.optim.Adam(m.parameters(), 1e-4)
        l0 = None
        for _ in range(300):
            l = F.mse_loss(m(x), y); o.zero_grad(); l.backward(); o.step(); l0 = l0 or float(l)
        rep[f"{nm}_overfit_loss_ratio"] = float(l) / l0; assert rep[f"{nm}_overfit_loss_ratio"] < 0.5, rep
    # mask sanity
    ev = EvalSet(PATIENTS[2], "val"); rep["val_mask_fraction"] = float(ev.mask.float().mean())
    assert 0.05 < rep["val_mask_fraction"] < 0.8
    # identity model RMSE == input noise RMSE (metric sanity)
    Id = Identity
    x_, y_ = ev.pairs(alpha=1.0, beta=1.0)
    direct = math.sqrt(float(sum(((x_[i] - y_[i])[ev.mask[i]] ** 2).sum() for i in range(len(x_)))) /
                       float(ev.mask.sum()))
    rep["identity_rmse_check"] = [rmse_hu(Id(), ev, alpha=1.0, beta=1.0), direct]
    assert abs(rep["identity_rmse_check"][0] - direct) < 1e-3 * direct
    del smp, ev; torch.cuda.empty_cache()
    return rep

if __name__ == "__main__" and not os.environ.get("STAGE2B_NO_RUN"):
    T0 = time.time()
    S = smoke(); print("SMOKE_JSON=" + json.dumps(S)); json.dump(S, open(f"{OUT}/smoke.json", "w"), indent=1)
    # timing guard: pilot (6 runs) must fit in ~9 h -> <= 90 min per run; N_MAX per arch = min(60k, budget/sec_per_it),
    # rounded down to a multiple of 10k (>= 20k). Timing runs (300 it) are discarded.
    timing = {}
    for arch in ["REDCNN", "DnCNN"]:
        r = train(arch, 1, 1.0, 1.0, 999, 300, f"timing_{arch}"); timing[arch] = r["sec_per_it"]
    NMAX = {a: max(20000, min(60000, int(5400 / t) // 10000 * 10000)) for a, t in timing.items()}
    print("TIMING_JSON=" + json.dumps({"sec_per_it": timing, "N_MAX": NMAX}), flush=True)
    json.dump({"sec_per_it": timing, "N_MAX": NMAX}, open(f"{OUT}/timing.json", "w"))
    res = []
    for arch in ["REDCNN", "DnCNN"]:
        N_MAX = NMAX[arch]
        for seed in [11, 22, 33]:
            tag = f"pilot_{arch}_f1_a1.0_b1.0_s{seed}"
            r = train(arch, 1, 1.0, 1.0, seed, N_MAX, tag)
            # V0 first look on the 2 test patients of fold 1 with the best checkpoint
            m = ARCHS[arch]().to(DEV); m.load_state_dict(torch.load(f"{OUT}/{tag}.pt", map_location=DEV))
            r["test"] = {}
            for p in FOLDS[1]:
                es = EvalSet(p, "test")
                r["test"][p] = {"emu_B30": rmse_hu(m, es, alpha=1.0, beta=1.0), "real_B30": rmse_hu(m, es, real_b30=True, beta=1.0),
                                "input_emu_B30": rmse_hu(Identity(), es, alpha=1.0, beta=1.0),
                                "input_real_B30": rmse_hu(Identity(), es, real_b30=True, beta=1.0)}
                del es
            res.append(r); json.dump(res, open(f"{OUT}/pilot_results.json", "w"), indent=1)
            print("PILOT_RUN=" + json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
    print("PILOT_DONE runtime_s=%.0f" % (time.time() - T0))
