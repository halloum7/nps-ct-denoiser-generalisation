"""Figures 2 and 5 of the article, from results/stage2d_results.json and results/metrics.csv.
Fig. 2: V0 equivalence of emulated and real B30, per patient.
Fig. 5: median absolute RMSE of every training condition on every beta = 1 test kernel, including off-family kernels."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("../results/stage2d_results.json"))
M = pd.read_csv("../results/metrics.csv")
ARCHS = [("REDCNN", BLUE, "o"), ("DnCNN", ORANGE, "s")]

# ---------------- Figure 2: V0, ratio RMSE(real B30) / RMSE(emulated B30) per patient
pats = sorted(R["V0"]["REDCNN"]["per_patient"])
fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
ax.axhspan(0.95, 1.05, color="#e5f4e4", zorder=0, label="Equivalence margin [0.95, 1.05]")
ax.axhline(1.0, color=INK2, lw=0.8)
for k, (arch, col, mk) in enumerate(ARCHS):
    v = R["V0"][arch]; x = np.arange(len(pats)) + (k - 0.5) * 0.22
    ax.scatter(x, [v["per_patient"][p] for p in pats], color=col, marker=mk, s=30, edgecolor="white", lw=0.5,
               label=f"{arch}: geometric mean {v['ratio_geomean']:.3f}, 90% CI {v['ci90'][0]:.3f}–{v['ci90'][1]:.3f}", zorder=3)
    xr = len(pats) + 0.3 + k * 0.35
    ax.errorbar([xr], [v["ratio_geomean"]], yerr=[[v["ratio_geomean"] - v["ci90"][0]], [v["ci90"][1] - v["ratio_geomean"]]],
                fmt=mk, color=col, capsize=3, ms=5, zorder=3)
ax.set_xticks(list(range(len(pats))) + [len(pats) + 0.475], pats + ["Mean\n(90% CI)"], fontsize=8)
ax.set_ylim(0.94, 1.06); ax.set_ylabel("RMSE ratio, real / emulated B30")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
for ext in ("png", "pdf"):
    fig.savefig(f"Fig2_V0_equivalence.{ext}", dpi=300)

# ---------------- Figure 5: median RMSE, 8 training conditions x 8 test kernels at beta = 1
TRAIN = [(a, 1.0) for a in (0.0, 0.25, 0.5, 0.75, 1.0)] + [(a, 0.5) for a in (0.0, 0.5, 1.0)]
TEST = ["a0.0", "a0.25", "a0.5", "a0.75", "a1.0", "B30real", "off0.8", "offBW"]
TLAB = ["α=0 (D45)", "α=0.25", "α=0.5", "α=0.75", "α=1", "Real B30", "T_h(f/0.8)", "Butterworth"]
cmap = LinearSegmentedColormap.from_list("seq", ["#f6f5f2", BLUE])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.3), constrained_layout=True)
mats = {}
for arch, _, _ in ARCHS:
    mat = np.zeros((len(TRAIN), len(TEST)))
    for i, (a, b) in enumerate(TRAIN):
        for j, c in enumerate(TEST):
            d = M[(M.arch == arch) & (M.train_alpha == a) & (M.train_beta == b) & (M.cond == c) & (M.beta == 1.0)]
            assert len(d) == 10, (arch, a, b, c, len(d))
            mat[i, j] = d.rmse.median()
    mats[arch] = mat
vmax = max(m.max() for m in mats.values())
for ax, (arch, _, _) in zip(axs, ARCHS):
    mat = mats[arch]; im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            matched = TEST[j] == f"a{TRAIN[i][0]}" and TRAIN[i][1] == 1.0
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if mat[i, j] > 0.6 * vmax else INK, fontweight="bold" if matched else "normal")
    ax.axvline(4.5, color=INK, lw=0.8); ax.axhline(4.5, color=INK, lw=0.8, ls=":")
    ax.set_xticks(range(len(TEST)), TLAB, fontsize=7, rotation=45, ha="right", rotation_mode="anchor"); ax.set_yticks(range(len(TRAIN)), [f"α={a}, β={b}" for a, b in TRAIN], fontsize=7.5)
    ax.set_xlabel("Test kernel (β = 1)"); ax.set_title(arch, fontsize=9.5, color=INK, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel("Training condition")
cb = fig.colorbar(im, ax=axs, shrink=0.8); cb.set_label("Median RMSE (HU), 10 test patients")
for ext in ("png", "pdf"):
    fig.savefig(f"Fig5_RMSE_matrix.{ext}", dpi=300)
print({k: np.round(v, 1).tolist() for k, v in mats.items()})
