"""
Schematic figure explaining the lesion area L and the outside-lesion reference region R.
The data is synthetic (illustration only).

Input:
    -o / --output_dir: folder where fig_outside_lesion.pdf and fig_outside_lesion.png are saved
        (default: current folder)

Author: Pierre-Louis Benveniste
"""
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Schematic figure of the lesion area L and the outside-lesion region R.")
parser.add_argument("-o", "--output_dir", default=".", help="Output folder. Default: current folder.")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------- settings
SUPERIOR_ON_LEFT = True        # set False if your other figures plot PAM50 slices increasing to the right
OUT = os.path.join(args.output_dir, "fig_outside_lesion")     # writes <OUT>.pdf and <OUT>.png

RED, RED_FILL = "#c8322f", "#f7d9d8"
BLUE, BLUE_FILL = "#2a6db0", "#dce8f5"
T0_COL, T_COL = "#d4a706", "#440154"     # baseline yellow -> follow-up purple, as in Fig. 2b
INK, MUTED = "#222222", "#7a7a7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "ytick.labelsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "ytick.major.width": 0.6, "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
})

# ---------------------------------------------------------------- synthetic data
z = np.arange(0, 121)                                   # PAM50 slices (higher = superior)
L_start, L_end = 51, 71                                 # lesion area (union across sessions)
half = math.ceil((L_end - L_start + 1) / 2)             # same rule as get_flank_slices()
R_below = (L_start - half, L_start - 1)
R_above = (L_end + 1, L_end + half)

base = 70 + 7 * np.exp(-((z - 15) / 30) ** 2) - 3 * np.exp(-((z - 95) / 25) ** 2)
csa_t0 = base + 1.0 * np.sin(z / 9)
csa_t = 0.965 * csa_t0 - 9 * np.exp(-((z - 61) / 8) ** 2)   # global loss + faster focal loss at the lesion


def seg_mean(csa, a, b):
    return csa[(z >= a) & (z <= b)].mean()


# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(3.45, 2.15))
Y_MIN, Y_MAX = 57, 84

# regions: shaded bands + labels at the top
regions = [((L_start, L_end), RED, RED_FILL, r"$\mathcal{L}$", r"$|\mathcal{L}|$"),
           (R_below, BLUE, BLUE_FILL, r"$\mathcal{R}$", r"$\lceil|\mathcal{L}|/2\rceil$"),
           (R_above, BLUE, BLUE_FILL, r"$\mathcal{R}$", r"$\lceil|\mathcal{L}|/2\rceil$")]
for (a, b), col, fill, name, width in regions:
    ax.axvspan(a - 0.5, b + 0.5, color=fill, lw=0, zorder=0)
    xc = (a + b) / 2
    ax.text(xc, Y_MAX - 0.6, name, color=col, ha="center", va="top", fontsize=10)
    ax.text(xc, Y_MAX - 4.2, width, color=col, ha="center", va="top", fontsize=6.5)
for x in (R_below[0] - 0.5, L_start - 0.5, L_end + 0.5, R_above[1] + 0.5):
    ax.axvline(x, color="white", lw=0.8, zorder=0.5)

# CSA profiles
ax.plot(z, csa_t0, color=T0_COL, lw=1.4, solid_capstyle="round", zorder=3)
ax.plot(z, csa_t, color=T_COL, lw=1.4, solid_capstyle="round", zorder=3)

# mean CSA over each region, per session (same slices for both sessions)
for (a, b), *_ in regions:
    for csa, col in [(csa_t0, T0_COL), (csa_t, T_COL)]:
        m = seg_mean(csa, a, b)
        ax.plot([a - 0.5, b + 0.5], [m, m], color=col, lw=1.0, ls=(0, (3, 1.5)), zorder=2)

# Delta arrows from baseline mean to follow-up mean
for (a, b), col, txt in [((L_start, L_end), RED, r"$\Delta_{\mathcal{L}}$"),
                         (R_above, BLUE, r"$\Delta_{\mathcal{R}}$")]:
    m0, m1 = seg_mean(csa_t0, a, b), seg_mean(csa_t, a, b)
    xc = (a + b) / 2
    ax.annotate("", xy=(xc, m1), xytext=(xc, m0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.annotate(txt, xy=(xc, m0), xytext=(0, 3), textcoords="offset points",
                color=col, ha="center", va="bottom", fontsize=8.5, zorder=4)

# direct labels at the (display) right end of each curve
x_end = z[0] if SUPERIOR_ON_LEFT else z[-1]
for csa, col, lab in [(csa_t0, T0_COL, r"$t_0$"), (csa_t, T_COL, r"$t$")]:
    ax.annotate(lab, xy=(x_end, csa[np.argmin(abs(z - x_end))]), xytext=(3, 0), textcoords="offset points",
                color=col, ha="left", va="center", fontsize=8.5, annotation_clip=False)

# axes
ax.set_xlim(z[0], z[-1])
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_yticks([60, 65, 70, 75, 80])
ax.set_ylabel(r"CSA (mm$^2$)")
ax.set_xticks([])
ax.set_xlabel("PAM50 axial slice", labelpad=3)
ax.spines[["top", "right"]].set_visible(False)
if SUPERIOR_ON_LEFT:
    ax.invert_xaxis()

left, right = ("superior", "inferior") if SUPERIOR_ON_LEFT else ("inferior", "superior")
ax.text(0.0, -0.035, f"← {left}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)
ax.text(1.0, -0.035, f"{right} →", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)

fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {OUT}.pdf / {OUT}.png  (L = {L_start}-{L_end}, R_below = {R_below}, R_above = {R_above})")