# ============================ STAGE 2d (CPU) : pre-registered statistics (protocol doc 2026-11-14, rev 25)
# Inputs : metrics table (one row per model x test patient x test condition x test beta) + Stage-2a pair_matrix.
# Order  : V0, then H1 (confirmatory), H2-H4 (descriptive, effects + CI). H5 is taken from Stage 2a (no denoiser).
# Pure numpy/scipy so that it can be re-run and checked outside Kaggle.
#
# Operational definitions fixed BEFORE any test-set inference (2026-11-21), where the protocol text left a choice:
#  * V0     : per architecture, model (alpha=1, beta=1) tested at beta=1 ; r_p = RMSE(real B30) / RMSE(emulated a1.0).
#  * H1     : two-sided exact Wilcoxon signed-rank on the 10 per-patient Spearman rho (vs 0); Holm over 2 archs.
#  * H2     : random-intercept LMM fitted by REML (profiled, closed form for compound symmetry) on the 56 unmatched
#             pairs x 2 archs x 10 patients; sign(dalpha) = sign(alpha_test - alpha_train) (0 for beta-only pairs);
#             patient-cluster bootstrap (B = 10000, resampled patients relabelled as distinct groups), percentile CI.
#  * H3     : per patient and architecture, mean L over symmetric pairs at equal beta (beta=1: 10 pairs ;
#             beta=0.5: 3 pairs), fine = smaller alpha ; two-sided exact paired Wilcoxon, reported per architecture.
#  * H4     : per architecture, OLS log(RMSE) ~ 1 + W1 + dlogvar on all family test conditions (15) x 8 models x
#             10 patients ; prediction for real B30, T_h(f/0.8), Butterworth at beta in {0.5, 1}.
#             Naive predictor = RMSE of the SAME model on the SAME patient at the nearest family condition
#             (alpha* = 1 for real B30 and T_h(f/0.8), alpha* = 0.5 for the Butterworth, same beta).
import json, math
import numpy as np, pandas as pd
from scipy import stats, optimize

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TRAIN_CONDS = [(a, 1.0) for a in ALPHAS] + [(a, 0.5) for a in (0.0, 0.5, 1.0)]
FAMILY_BETAS = [0.5, 0.75, 1.0]
OFF_CONDS = {"B30real": 1.0, "off0.8": 1.0, "offBW": 0.5}      # -> alpha* of the naive predictor
ARCHS = ["REDCNN", "DnCNN"]


def cname(a):
    return f"a{float(a)}"


# ------------------------------------------------------------------ data assembly
def build_L(met):
    """All (train c -> test c') pairs with c' in TRAIN_CONDS ; L = RMSE(f_c, c') / RMSE(f_c', c') - 1."""
    m = met.set_index(["arch", "patient", "train_alpha", "train_beta", "cond", "beta"])["rmse"]
    rows = []
    for arch in ARCHS:
        for p in sorted(met.patient.unique()):
            for (a2, b2) in TRAIN_CONDS:
                ref = m.loc[(arch, p, a2, b2, cname(a2), b2)]
                for (a1, b1) in TRAIN_CONDS:
                    if (a1, b1) == (a2, b2):
                        continue
                    v = m.loc[(arch, p, a1, b1, cname(a2), b2)]
                    rows.append(dict(arch=arch, patient=p, train_alpha=a1, train_beta=b1, test_alpha=a2, test_beta=b2,
                                     rmse=v, rmse_matched=ref, L=v / ref - 1))
    return pd.DataFrame(rows)


def attach_pairs(df, pm, fold_of):
    """Add W1 (reference, pair_matrix.W1_ref) and dlogvar for each row (train_alpha, train_beta, patient, test cond, beta)."""
    key = pm.set_index(["fold", "train_alpha", "train_beta", "test_patient", "test_cond", "test_beta"])
    W, D = [], []
    for r in df.itertuples():
        tc = r.test_cond if hasattr(r, "test_cond") else cname(r.test_alpha)
        k = key.loc[(fold_of[r.patient], r.train_alpha, r.train_beta, r.patient, tc, r.test_beta)]
        W.append(float(k.W1_ref)); D.append(float(k.dlogvar))
    df = df.copy(); df["W1"] = W; df["dlogvar"] = D
    return df


# ------------------------------------------------------------------ V0
def v0(met, arch):
    m = met[(met.arch == arch) & (met.train_alpha == 1.0) & (met.train_beta == 1.0) & (met.beta == 1.0)]
    real = m[m.cond == "B30real"].set_index("patient").rmse
    emu = m[m.cond == "a1.0"].set_index("patient").rmse
    lr = np.log(real / emu.loc[real.index]).values
    n = len(lr); mu = lr.mean(); se = lr.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.95, n - 1)
    ci = (math.exp(mu - t * se), math.exp(mu + t * se))
    return {"n": n, "ratio_geomean": math.exp(mu), "ci90": ci, "per_patient": dict(zip(real.index, np.exp(lr).round(5))),
            "PASS": bool(ci[0] >= 0.95 and ci[1] <= 1.05)}


# ------------------------------------------------------------------ H1
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (k - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def h1(Lw):
    out, ps = {}, []
    for arch in ARCHS:
        d = Lw[(Lw.arch == arch) & (Lw.train_beta == 1.0) & (Lw.test_beta == 1.0)]
        rhos = {}
        for p, g in d.groupby("patient"):
            assert len(g) == 20, (arch, p, len(g))
            rhos[p] = float(stats.spearmanr(g.W1, g.L).statistic)
        r = np.array(list(rhos.values()))
        w = stats.wilcoxon(r, alternative="two-sided", method="exact") if np.any(r != 0) else None
        out[arch] = {"rho_per_patient": {k: round(v, 4) for k, v in rhos.items()}, "rho_median": float(np.median(r)),
                     "n_positive": int((r > 0).sum()), "wilcoxon_p": float(w.pvalue) if w else 1.0}
        ps.append(out[arch]["wilcoxon_p"])
    adj = holm(ps)
    for arch, pa in zip(ARCHS, adj):
        out[arch]["p_holm"] = float(pa)
        out[arch]["PASS"] = bool(pa < 0.05 and out[arch]["rho_median"] >= 0.5)
    out["PASS_both"] = bool(all(out[a]["PASS"] for a in ARCHS))
    return out


# ------------------------------------------------------------------ H2 : random-intercept LMM (REML)
def lmm_ri(y, X, g):
    """y = X b + u_g + e, u ~ N(0, s2u), e ~ N(0, s2e). Profiled REML over lam = s2u / s2e (closed-form
    inverse of I + lam J per group). Returns (beta, s2e, s2u)."""
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    groups = [np.flatnonzero(g == k) for k in np.unique(g)]
    N, P = X.shape
    Xs = [(X[i], y[i]) for i in groups]
    sx = [(Xi.sum(0), yi.sum(), len(yi)) for Xi, yi in Xs]
    XtX = X.T @ X; Xty = X.T @ y; yty = y @ y

    def parts(lam):
        A = XtX.copy(); b = Xty.copy(); c = yty; logdet = 0.0
        for (sxi, syi, n) in sx:
            k = lam / (1 + n * lam)
            A -= k * np.outer(sxi, sxi); b -= k * sxi * syi; c -= k * syi * syi; logdet += math.log1p(n * lam)
        beta = np.linalg.solve(A, b)
        rss = c - b @ beta
        return A, beta, rss, logdet

    def nll(t):
        A, beta, rss, logdet = parts(math.exp(t))
        s2 = rss / (N - P)
        return (N - P) * math.log(s2) + logdet + np.linalg.slogdet(A)[1]

    res = optimize.minimize_scalar(nll, bounds=(-20, 8), method="bounded", options={"xatol": 1e-6})
    lam = math.exp(res.x)
    A, beta, rss, _ = parts(lam)
    s2e = rss / (N - P)
    return beta, s2e, lam * s2e


def h2_design(d):
    X = np.column_stack([np.ones(len(d)), d.W1, d.dlogvar, np.sign(d.test_alpha - d.train_alpha),
                         (d.arch == "DnCNN").astype(float)])
    return X, ["intercept", "W1", "dlogvar", "sign_dalpha", "arch_DnCNN"]


def h2(Lw, B=10000, seed=2026):
    d = Lw.reset_index(drop=True)
    X, names = h2_design(d)
    beta, s2e, s2u = lmm_ri(d.L.values, X, d.patient.values)
    pats = np.array(sorted(d.patient.unique()))
    idx = {p: np.flatnonzero(d.patient.values == p) for p in pats}
    rng = np.random.default_rng(seed); bw = np.empty(B)
    for b in range(B):
        pick = rng.choice(pats, len(pats), replace=True)
        ii = np.concatenate([idx[p] for p in pick]); gg = np.concatenate([np.full(len(idx[p]), k) for k, p in enumerate(pick)])
        bw[b] = lmm_ri(d.L.values[ii], X[ii], gg)[0][1]
    ci = (float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5)))
    return {"n_rows": int(len(d)), "coef": dict(zip(names, map(float, beta))), "s2_resid": float(s2e), "s2_patient": float(s2u),
            "W1_ci95_bootstrap": ci, "B": B, "PASS": bool(ci[0] > 0 or ci[1] < 0)}


# ------------------------------------------------------------------ H3
def h3(Lw):
    out = {}
    for arch in ARCHS:
        d = Lw[(Lw.arch == arch) & (Lw.train_beta == Lw.test_beta)]
        fs, sf = {}, {}
        for p, g in d.groupby("patient"):
            a = g[g.test_alpha > g.train_alpha].L; b = g[g.test_alpha < g.train_alpha].L
            assert len(a) == len(b) == 13, (arch, p, len(a), len(b))
            fs[p] = float(a.mean()); sf[p] = float(b.mean())
        x = np.array([fs[p] for p in sorted(fs)]); y = np.array([sf[p] for p in sorted(fs)])
        w = stats.wilcoxon(x, y, alternative="two-sided", method="exact")
        out[arch] = {"L_fine_to_smooth_median": float(np.median(x)), "L_smooth_to_fine_median": float(np.median(y)),
                     "n_patients_fs_lt_sf": int((x < y).sum()), "wilcoxon_p": float(w.pvalue),
                     "PASS": bool(w.pvalue < 0.05 and np.median(x - y) < 0)}
    return out


# ------------------------------------------------------------------ H4
def h4(met_pairs):
    out = {}
    for arch in ARCHS:
        d = met_pairs[met_pairs.arch == arch]
        fam = d[d.cond.str.match(r"^a\d") & d.beta.isin(FAMILY_BETAS)]
        Xf = np.column_stack([np.ones(len(fam)), fam.W1, fam.dlogvar]); yf = np.log(fam.rmse.values)
        coef, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        off = d[d.cond.isin(list(OFF_CONDS)) & d.beta.isin([0.5, 1.0])].copy()
        off["pred"] = np.exp(np.column_stack([np.ones(len(off)), off.W1, off.dlogvar]) @ coef)
        key = d.set_index(["patient", "train_alpha", "train_beta", "cond", "beta"]).rmse
        off["naive"] = [key.loc[(r.patient, r.train_alpha, r.train_beta, cname(OFF_CONDS[r.cond]), r.beta)] for r in off.itertuples()]
        off["err_model"] = (off.pred - off.rmse).abs() / off.rmse
        off["err_naive"] = (off.naive - off.rmse).abs() / off.rmse
        per = off.groupby("cond")[["err_model", "err_naive"]].median().round(4).to_dict("index")
        em, en = float(off.err_model.median()), float(off.err_naive.median())
        out[arch] = {"coef": dict(zip(["intercept", "W1", "dlogvar"], map(float, coef))), "n_fit": int(len(fam)),
                     "n_pred": int(len(off)), "median_rel_err_model": em, "median_rel_err_naive": en, "per_condition": per,
                     "PASS": bool(em <= 0.10 and em < en)}
    return out


# ------------------------------------------------------------------ driver
def run_all(met, pm, fold_of, B=10000):
    res = {"V0": {a: v0(met, a) for a in ARCHS}}
    res["V0"]["PASS_both"] = bool(all(res["V0"][a]["PASS"] for a in ARCHS))
    L = build_L(met)
    Lw = attach_pairs(L, pm, fold_of)
    res["H1"] = h1(Lw)
    res["H2"] = h2(Lw, B=B)
    res["H3"] = h3(Lw)
    mp = attach_pairs(met.rename(columns={"patient": "patient"}).assign(test_beta=met.beta, test_cond=met.cond), pm, fold_of)
    res["H4"] = h4(mp)
    return res, Lw, mp
