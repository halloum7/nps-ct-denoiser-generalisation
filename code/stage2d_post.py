# ============================ STAGE 2d (CPU) : assemble per-model evaluation files, run the statistics, figures
import os, glob, json, gzip, base64, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage2d_stats as S

FOLDS = {1: ["L067", "L291"], 2: ["L109", "L143"], 3: ["L192", "L310"], 4: ["L333", "L096"], 5: ["L506", "L286"]}
FOLD_OF = {p: f for f, ps in FOLDS.items() for p in ps}


def assemble(eout):
    files = sorted(glob.glob(f"{eout}/eval_*.json"))
    assert len(files) == 80, len(files)
    rows, chk = [], []
    for fn in files:
        e = json.load(open(fn))
        chk.append(dict(tag=e["tag"], val_recheck=e["val_recheck"], val_recorded=e["val_recorded"], rel=e["val_rel_diff"]))
        rows += e["rows"]
    met = pd.DataFrame(rows)
    assert len(met) == 80 * 2 * 21, len(met)
    assert not met.duplicated(["arch", "patient", "train_alpha", "train_beta", "cond", "beta"]).any()
    assert np.isfinite(met.rmse).all() and (met.rmse > 0).all()
    return met, pd.DataFrame(chk)


def b64gz(df, cols, fmt="%.6g"):
    buf = io.StringIO(); df[cols].to_csv(buf, index=False, float_format=fmt)
    return base64.b64encode(gzip.compress(buf.getvalue().encode(), 9)).decode()


def figures(met, Lw, mp, res, eout):
    # F1: L vs W1 (texture series), per architecture
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, arch in enumerate(S.ARCHS):
        d = Lw[(Lw.arch == arch) & (Lw.train_beta == 1.0) & (Lw.test_beta == 1.0)]
        for p, g in d.groupby("patient"):
            ax[k].scatter(g.W1, 100 * g.L, s=12, label=p)
        r = res["H1"][arch]
        ax[k].set_title(f"{arch}: median rho = {r['rho_median']:.2f}, Holm p = {r['p_holm']:.3g}")
        ax[k].set_xlabel("W1 between training and test noise texture (1/mm)"); ax[k].axhline(0, c="gray", lw=0.6)
    ax[0].set_ylabel("generalisation loss L (%)"); ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{eout}/fig_H1_L_vs_W1.png", dpi=200); plt.close(fig)
    # F2: mean RMSE heat map, train alpha x test alpha, beta = 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, arch in enumerate(S.ARCHS):
        d = met[(met.arch == arch) & (met.train_beta == 1.0) & (met.beta == 1.0) & met.cond.str.match(r"^a\d")]
        t = d.assign(test_alpha=d.cond.str[1:].astype(float)).pivot_table(index="train_alpha", columns="test_alpha", values="rmse", aggfunc="mean")
        im = ax[k].imshow(t.values, cmap="viridis"); ax[k].set_xticks(range(5), t.columns); ax[k].set_yticks(range(5), t.index)
        for i in range(5):
            for j in range(5):
                ax[k].text(j, i, f"{t.values[i, j]:.1f}", ha="center", va="center", color="w", fontsize=7)
        ax[k].set_xlabel("test alpha"); ax[k].set_ylabel("training alpha"); ax[k].set_title(f"{arch}: mean RMSE (HU), beta = 1")
        fig.colorbar(im, ax=ax[k], shrink=0.8)
    fig.tight_layout(); fig.savefig(f"{eout}/fig_rmse_matrix.png", dpi=200); plt.close(fig)
    # F3: V0
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, arch in enumerate(S.ARCHS):
        v = res["V0"][arch]; pp = list(v["per_patient"])
        ax.scatter(np.arange(len(pp)) + 0.15 * k, list(v["per_patient"].values()), label=f"{arch} (90% CI {v['ci90'][0]:.3f}-{v['ci90'][1]:.3f})")
    ax.axhspan(0.95, 1.05, color="g", alpha=0.08); ax.axhline(1, c="gray", lw=0.6)
    ax.set_xticks(range(len(pp)), pp, rotation=45, fontsize=7); ax.set_ylabel("RMSE real B30 / emulated B30"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{eout}/fig_V0.png", dpi=200); plt.close(fig)


def run(eout, pm_path, B=10000):
    met, chk = assemble(eout)
    pm = pd.read_csv(pm_path)
    assert len(pm) == 5 * 8 * 2 * 8 * 3, len(pm)
    res, Lw, mp = S.run_all(met, pm, FOLD_OF, B=B)
    res["val_recheck"] = {"max_rel_diff": float(chk.rel.max()), "n": int(len(chk))}
    met.drop(columns=["res_nps"]).to_csv(f"{eout}/metrics.csv", index=False)
    met[["arch", "patient", "train_alpha", "train_beta", "cond", "beta", "res_nps"]].to_json(f"{eout}/residual_nps.json", orient="records")
    Lw.to_csv(f"{eout}/L_pairs.csv", index=False)
    figures(met, Lw, mp, res, eout)
    json.dump(res, open(f"{eout}/stage2d_results.json", "w"), indent=1, default=float)
    return res, met, Lw, pm
