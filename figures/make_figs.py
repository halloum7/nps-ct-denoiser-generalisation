import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
import h1_data as h

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
ARCH = [("R", "REDCNN"), ("Dn", "DnCNN")]
fine2smooth = np.array([a < b for a, b in h.PAIRS])   # train sharper -> test smoother

# ---------------- Figure 2: L vs W1
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
for ax, (k, name) in zip(axs, ARCH):
    W = np.concatenate([v["W"] for v in h.D.values()]); L = 100 * np.concatenate([v[k] for v in h.D.values()])
    fs = np.tile(fine2smooth, len(h.D))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(W[fs], L[fs], s=16, marker="o", color=BLUE, alpha=0.75, edgecolor="white", lw=0.4, label="sharper → smoother kernel")
    ax.scatter(W[~fs], L[~fs], s=18, marker="^", color=ORANGE, alpha=0.75, edgecolor="white", lw=0.4, label="smoother → sharper kernel")
    rho = [stats.spearmanr(v["W"], v[k]).correlation for v in h.D.values()]
    ax.set_title(f"{name}  (median per-patient ρ = {np.median(rho):.2f})", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("W1 between training and test NPS (cycles/mm)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel("Generalisation loss L (%)")
h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8.5, loc="outside lower center", ncol=2)
for ext in ("png", "pdf"):
    fig.savefig(f"Fig2_L_vs_W1.{ext}", dpi=300)

# ---------------- Figure 3: median L heatmap (train alpha x test alpha, beta = 1)
A = h.A; cmap = LinearSegmentedColormap.from_list("seq", ["#f6f5f2", ORANGE])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
mats = {}
for k, name in ARCH:
    m = np.zeros((5, 5))
    for i, (a, b) in enumerate(h.PAIRS):
        m[A.index(a), A.index(b)] = 100 * np.median([v[k][i] for v in h.D.values()])
    mats[k] = m
vmax = max(abs(m).max() for m in mats.values())
for ax, (k, name) in zip(axs, ARCH):
    m = mats[k]
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(5):
        for j in range(5):
            txt = "—" if i == j else f"{m[i, j]:+.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(5), [str(a) for a in A]); ax.set_yticks(range(5), [str(a) for a in A])
    ax.set_xlabel("Test α (0 = D45, 1 ≈ B30)"); ax.set_ylabel("Training α")
    ax.set_title(name, fontsize=9.5, color=INK, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axs, shrink=0.8); cb.set_label("Median generalisation loss L (%)\n(values < 0 shown as 0)")
for ext in ("png", "pdf"):
    fig.savefig(f"Fig3_L_matrix.{ext}", dpi=300)
print({k: np.round(m, 1).tolist() for k, m in mats.items()})
