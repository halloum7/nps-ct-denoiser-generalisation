
# ============================ STAGE 2c (GPU) : full training, 2 x T4 in parallel, resumable
# Same protocol as the pilot; only the data pipeline is made faster, without changing what is sampled:
#  - x = K[FD + beta (QD - FD)] and y = K[FD] are computed ONCE per training run (identical arithmetic to the pilot,
#    float32, on GPU) instead of at every iteration;
#  - a patch corner is drawn uniformly among the corners whose patch has >= 50 % of pixels > -300 HU in y
#    (the pilot drew the same distribution by rejection; its rare fallback after 48 failed tries is replaced by a
#    uniform draw over all corners when a slice has no valid corner);
#  - 8 slices x 4 patches per batch, flips / rot90 as in the pilot.
# ============================ STAGE 2b (GPU) : smoke tests + pilot (fold 1, alpha=1, beta=1, 3 seeds x 2 archs)
# Protocol (pre-registered doc 2026-11-14): MSE, Adam 1e-4 -> 1e-5 cosine, batch 32 patches 64x64, HU/1000,
# flips/rot90 only, on-the-fly kernel emulation K_alpha = T_h^alpha in physical frequency (patient pixel size),
# checkpoint = best validation-patient RMSE every 2000 it. Pilot fixes N (rule below), seed variability, V0 first look.
# N rule (fixed before the pilot): N = smallest multiple of 10k at which the seed-mean validation RMSE improves by
# < 0.5 % over the next 10k iterations; capped at N_MAX.
import os, sys, json, time, glob, math, hashlib, shutil, subprocess
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # fixed input sizes: let cuDNN pick its fastest algorithms
EST_SEC = json.loads(os.environ.get("EST_SEC", "{}"))
USE_AMP = json.loads(os.environ.get("USE_AMP", "{}"))
OUT = "/kaggle/working/stage2c"; os.makedirs(OUT, exist_ok=True)
MAN = sorted(p for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True) if "cache" in p)
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

class Identity(nn.Module):
    def forward(self, x): return x

class FastSampler:
    def __init__(self, pats, alpha, beta, seed):
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        X, Y, V = [], [], []
        for p in pats:
            d = load_patient(p); H = kernel_rfft(p, alpha)
            fd_all = torch.tensor(d["D45_FD"], device=DEV); qd_all = torch.tensor(d["D45_QD"], device=DEV)
            for i0 in range(0, len(fd_all), 16):
                fd = fd_all[i0:i0 + 16].float(); qd = qd_all[i0:i0 + 16].float()
                x = apply_k(fd + beta * (qd - fd), H); y = apply_k(fd, H)
                ok = F.avg_pool2d((y > -300).float()[:, None], PATCH, stride=1)[:, 0] >= 0.5     # (n, 449, 449)
                X.append(x); Y.append(y); V += [torch.nonzero(o.flatten()).flatten() for o in ok]
            del fd_all, qd_all
        self.x = torch.cat(X); self.y = torch.cat(Y); self.valid = V; self.side = 512 - PATCH + 1
    def batch(self):
        n = self.x.shape[0]; per = BATCH // SL_PER_IT
        sl = torch.randint(n, (SL_PER_IT,), generator=self.g).tolist()
        XS, YS = [], []
        for s in sl:
            v = self.valid[s]
            if len(v) > 0:
                c = v[torch.randint(len(v), (per,), generator=self.g)].tolist()
            else:
                c = torch.randint(self.side * self.side, (per,), generator=self.g).tolist()
            t = torch.randint(8, (per,), generator=self.g).tolist()
            for cc, tt in zip(c, t):
                r0, c0 = divmod(cc, self.side)
                xp = self.x[s, r0:r0 + PATCH, c0:c0 + PATCH]; yp = self.y[s, r0:r0 + PATCH, c0:c0 + PATCH]
                if tt & 4: xp, yp = xp.flip(-1), yp.flip(-1)
                XS.append(torch.rot90(xp, tt & 3)); YS.append(torch.rot90(yp, tt & 3))
        return torch.stack(XS)[:, None] / 1000.0, torch.stack(YS)[:, None] / 1000.0

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


def train(arch, fold, alpha, beta, seed, n_iter, tag, amp=False):
    """amp=True: forward/backward under float16 autocast with dynamic loss scaling (T4 tensor cores);
    evaluation always in float32."""
    torch.manual_seed(seed); np.random.seed(seed)
    test = FOLDS[fold]; val = VAL[fold]; pats = [p for p in PATIENTS if p not in test and p != val]
    smp = FastSampler(pats, alpha, beta, seed); ev = EvalSet(val, "val")
    model = ARCHS[arch]().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=LR1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    curve, best, t0 = [], (1e9, 0), time.time()
    for it in range(1, n_iter + 1):
        x, y = smp.batch()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.mse_loss(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % EVAL_EVERY == 0:
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite loss at it {it}")
            v = rmse_hu(model, ev, alpha=alpha, beta=beta); curve.append((it, v, lv))
            if v < best[0]:
                best = (v, it); torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    sec = time.time() - t0
    del smp; torch.cuda.empty_cache()
    return {"arch": arch, "fold": fold, "alpha": alpha, "beta": beta, "seed": seed, "n_iter": n_iter, "tag": tag,
            "best_val_rmse": best[0], "best_it": best[1], "curve": curve, "train_seconds": sec, "sec_per_it": sec / n_iter, "amp": amp}

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_CONDS = [(a, 1.0) for a in ALPHAS] + [(a, 0.5) for a in (0.0, 0.5, 1.0)]
def all_jobs():
    J = []
    for fold in sorted(FOLDS):
        for arch in ["REDCNN", "DnCNN"]:
            for (a, b) in TRAIN_CONDS:
                tag = f"{arch}_f{fold}_a{a}_b{b}"
                J.append(dict(arch=arch, fold=fold, alpha=a, beta=b, seed=1000 + zlib_seed(tag), tag=tag))
    return J
def zlib_seed(tag):
    import zlib; return zlib.crc32(tag.encode()) % 100000

def equivalence_smoke():
    """FastSampler slices == pilot computation (apply_k on the slice) ; timing of 200 iterations."""
    p = PATIENTS[0]; d = load_patient(p); H = kernel_rfft(p, 0.5)
    fs = FastSampler([p], 0.5, 0.75, 0)
    i = len(d["D45_FD"]) // 2
    fd = torch.tensor(d["D45_FD"][i], device=DEV).float(); qd = torch.tensor(d["D45_QD"][i], device=DEV).float()
    ref_x = apply_k(fd + 0.75 * (qd - fd), H); ref_y = apply_k(fd, H)
    rep = {"max_abs_x_HU": float((fs.x[i] - ref_x).abs().max()), "max_abs_y_HU": float((fs.y[i] - ref_y).abs().max())}
    assert rep["max_abs_x_HU"] < 1e-2 and rep["max_abs_y_HU"] < 1e-2, rep
    x, y = fs.batch(); assert x.shape == (BATCH, 1, PATCH, PATCH)
    rep["frac_patch_tissue_ok"] = float(((y * 1000 > -300).float().mean(dim=(1, 2, 3)) >= 0.5).float().mean())
    for amp in (False, True):
        for nm, A in ARCHS.items():
            m = A().to(DEV); o = torch.optim.Adam(m.parameters(), 1e-4); sc = torch.amp.GradScaler("cuda", enabled=amp)
            for k in range(220):
                if k == 20: torch.cuda.synchronize(); t0 = time.time()
                x, y = fs.batch()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    l = F.mse_loss(m(x).float(), y)
                o.zero_grad(); sc.scale(l).backward(); sc.step(o); sc.update()
            torch.cuda.synchronize(); rep[f"{nm}_sec_per_it_{'amp' if amp else 'fp32'}"] = (time.time() - t0) / 200
            del m, o
    del fs; torch.cuda.empty_cache(); return rep

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    # precision check: same job, same seed, fp32 (gpu 0) vs amp (gpu 1)
    gpu, n_iter, arch = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; amp = gpu == 1
    j = [j for j in all_jobs() if j["tag"] == f"{arch}_f1_a1.0_b1.0"][0]
    tag = f"check_{'amp' if amp else 'fp32'}_" + j["tag"]
    r = train(j["arch"], j["fold"], j["alpha"], j["beta"], j["seed"], n_iter, tag, amp=amp)
    json.dump(r, open(f"{OUT}/{tag}.check", "w")); print(f"[check gpu{gpu}] " + json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "worker":
    # Session >= 2 (operational change only; job definitions, seeds, N, precision unchanged): jobs are claimed
    # dynamically by the two GPUs (atomic O_EXCL claim file) in longest-first order, so the load is balanced; a job that
    # no longer fits before the deadline is skipped (a shorter one may still fit) instead of stopping the worker.
    gpu, n_iter, deadline = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    jobs = sorted(all_jobs(), key=lambda j: -EST_SEC.get(j["arch"], 3600))
    for j in jobs:
        if os.path.exists(f"{OUT}/{j['tag']}.json"):
            continue
        if time.time() + 1.3 * EST_SEC.get(j["arch"], 3600) > deadline:
            print(f"[gpu{gpu}] skip (time): {j['tag']}", flush=True); continue
        try:
            fd = os.open(f"{OUT}/{j['tag']}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(gpu).encode()); os.close(fd)
        except FileExistsError:
            continue
        r = train(j["arch"], j["fold"], j["alpha"], j["beta"], j["seed"], n_iter, j["tag"], amp=USE_AMP.get(j["arch"], False))
        r["gpu"] = gpu; json.dump(r, open(f"{OUT}/{j['tag']}.json", "w"))
        os.remove(f"{OUT}/{j['tag']}.claim")
        print(f"[gpu{gpu}] DONE " + json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
