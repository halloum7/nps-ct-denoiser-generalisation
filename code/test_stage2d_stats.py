import itertools, math, time
import numpy as np, pandas as pd
from scipy import stats
import stage2d_stats as S

FOLDS = {1: ["L067", "L291"], 2: ["L109", "L143"], 3: ["L192", "L310"], 4: ["L333", "L096"], 5: ["L506", "L286"]}
FOLD_OF = {p: f for f, ps in FOLDS.items() for p in ps}
PATS = sorted(FOLD_OF)
CONDS = [S.cname(a) for a in S.ALPHAS] + ["B30real", "off0.8", "offBW"]
GPOS = {"a0.0": 0.0, "a0.25": 0.05, "a0.5": 0.09, "a0.75": 0.12, "a1.0": 0.14, "B30real": 0.141, "off0.8": 0.19, "offBW": 0.10}


def synth(effect=True, v0_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pm, met = [], []
    base = {p: rng.uniform(40, 65) for p in PATS}
    u = {p: rng.normal(0, 0.02) for p in PATS}
    for p in PATS:
        fo = FOLD_OF[p]
        for (a, b) in S.TRAIN_CONDS:
            for c in CONDS:
                w1 = abs(GPOS[S.cname(a)] - GPOS[c]) + abs(rng.normal(0, 0.002))
                for b2 in [0.5, 0.75, 1.0]:
                    dl = math.log((b2 / b) ** 2) + rng.normal(0, 0.01) + 0.4 * (GPOS[c] - GPOS[S.cname(a)])
                    pm.append(dict(fold=fo, train_alpha=a, train_beta=b, test_patient=p, test_cond=c, test_beta=b2,
                                   W1_ref=w1, W1_est=w1 + 0.02, dlogvar=dl))
                    level = base[p] * (0.55 + 0.45 * b2) * (1 - 0.3 * GPOS[c] / 0.14)
                    pen = (1 + (3.0 * w1 if effect else 0) + 0.05 * abs(dl) + u[p]) * math.exp(rng.normal(0, 0.003 if effect else 0.05))
                    if c == "B30real":
                        level *= (1 + v0_shift)
                    for arch in S.ARCHS:
                        met.append(dict(arch=arch, fold=fo, train_alpha=a, train_beta=b, patient=p, cond=c, beta=b2,
                                        rmse=level * pen * (1.01 if arch == "DnCNN" else 1.0)))
    return pd.DataFrame(met), pd.DataFrame(pm)


def brute_reml(y, X, g, s2e, s2u):
    N, P = X.shape
    V = s2e * np.eye(N) + s2u * (g[:, None] == g[None, :])
    Vi = np.linalg.inv(V); A = X.T @ Vi @ X; b = np.linalg.solve(A, X.T @ Vi @ y); r = y - X @ b
    return np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1] + r @ Vi @ r, b


def test_lmm():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(8), 15); X = np.column_stack([np.ones(120), rng.normal(size=120), rng.normal(size=120)])
    y = X @ [1.0, 2.0, -0.5] + rng.normal(0, 0.7, 8)[g] + rng.normal(0, 0.3, 120)
    beta, s2e, s2u = S.lmm_ri(y, X, g)
    # optimum of the brute-force REML criterion must be at (s2e, s2u) and give the same beta
    f0, b0 = brute_reml(y, X, g, s2e, s2u)
    for de, du in [(1.05, 1), (0.95, 1), (1, 1.1), (1, 0.9)]:
        assert brute_reml(y, X, g, s2e * de, s2u * du)[0] > f0 - 1e-9
    assert np.allclose(b0, beta, atol=1e-8), (b0, beta)
    assert abs(beta[1] - 2.0) < 0.1 and 0.15 < s2u < 1.5 and 0.05 < s2e < 0.15, (beta, s2e, s2u)
    print("lmm ok", beta.round(3), round(s2e, 4), round(s2u, 4))


def test_positive():
    met, pm = synth(effect=True, v0_shift=0.005)
    t = time.time(); res, Lw, mp = S.run_all(met, pm, FOLD_OF, B=300); dt = time.time() - t
    assert len(Lw) == 2 * 10 * 56
    assert res["V0"]["PASS_both"], res["V0"]
    assert res["H1"]["PASS_both"], res["H1"]
    for a in S.ARCHS:
        assert res["H1"][a]["rho_median"] > 0.9 and res["H1"][a]["n_positive"] == 10
        assert abs(res["H1"][a]["wilcoxon_p"] - 0.001953125) < 1e-9          # exact minimum two-sided p, n = 10
    assert res["H2"]["PASS"] and res["H2"]["coef"]["W1"] > 0, res["H2"]
    print("positive ok (%.1fs, B=300)" % dt, {a: res["H1"][a]["rho_median"] for a in S.ARCHS}, res["H2"]["coef"]["W1"],
          res["H2"]["W1_ci95_bootstrap"], {a: (res["H4"][a]["median_rel_err_model"], res["H4"][a]["median_rel_err_naive"]) for a in S.ARCHS},
          {a: res["H3"][a]["wilcoxon_p"] for a in S.ARCHS})


def test_null_and_v0_fail():
    met, pm = synth(effect=False, v0_shift=0.10, seed=3)
    res, Lw, mp = S.run_all(met, pm, FOLD_OF, B=200)
    assert not res["V0"]["PASS_both"], res["V0"]
    assert all(res["V0"][a]["ratio_geomean"] > 1.08 for a in S.ARCHS)
    assert not res["H1"]["PASS_both"], res["H1"]
    print("null ok", {a: (round(res["H1"][a]["rho_median"], 3), round(res["H1"][a]["p_holm"], 3)) for a in S.ARCHS})


def test_L_definition():
    met, pm = synth(effect=True, seed=5)
    L = S.build_L(met)
    r = L[(L.arch == "REDCNN") & (L.patient == "L291") & (L.train_alpha == 0.0) & (L.train_beta == 1.0) & (L.test_alpha == 1.0) & (L.test_beta == 1.0)].iloc[0]
    m = met[(met.arch == "REDCNN") & (met.patient == "L291") & (met.cond == "a1.0") & (met.beta == 1.0)]
    num = m[(m.train_alpha == 0.0) & (m.train_beta == 1.0)].rmse.item(); den = m[(m.train_alpha == 1.0) & (m.train_beta == 1.0)].rmse.item()
    assert abs(r.L - (num / den - 1)) < 1e-12
    print("L definition ok")


def test_holm():
    assert np.allclose(S.holm([0.01, 0.04]), [0.02, 0.04]); assert np.allclose(S.holm([0.03, 0.02]), [0.04, 0.04])
    print("holm ok")


if __name__ == "__main__":
    test_holm(); test_lmm(); test_L_definition(); test_positive(); test_null_and_v0_fail()
    print("ALL STATS TESTS PASSED")
