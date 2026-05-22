import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

GAMMA_MIN = 0
GAMMA_MAX = 1.0
LOSS_CAP  = 0.3233   # p99
SIGMA_MED = 0.0378   # p50
#SIGMA_MED = 0.3233
SIGMA_SCALE = 0.01   # tight — most train data is in a narrow band

QUANTILES = {
    "p25": 0.0204,
    "p50": 0.0378,
    "p75": 0.0744,
    "p99": 0.3233,
}

def compute_linear_gamma(loss, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, loss_cap=LOSS_CAP):
    t = np.clip(loss / loss_cap, 0.0, 1.0)
    return gamma_max - t * (gamma_max - gamma_min)

def compute_sigmoid_gamma(loss, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX,
                          sigma_med=SIGMA_MED, sigma_scale=SIGMA_SCALE):
    z = (loss - sigma_med) / sigma_scale
    s = 1.0 / (1.0 + np.exp(z))
    return gamma_min + s * (gamma_max - gamma_min)

# log-spaced losses: in-distribution range + OOD tail
#loss_vals = np.logspace(np.log10(0.005), np.log10(5.0), 500)
loss_vals = np.linspace(0.0, 5.0, 500)  # or push to 3.0 if you want to see deeper OOD

linear_gammas  = compute_linear_gamma(loss_vals)
sigmoid_gammas = compute_sigmoid_gamma(loss_vals)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(loss_vals, linear_gammas,  color="#3266ad", lw=2.5, label=f"linear  (loss_cap=p99={LOSS_CAP})")
ax.plot(loss_vals, sigmoid_gammas, color="#c0392b", lw=2.5, linestyle="--",
        label=f"sigmoid (sigma_med=p50={SIGMA_MED}, scale={SIGMA_SCALE})")

for name, val in QUANTILES.items():
    if name == "p25" or name == "p50":
        break
    ax.axvline(val, color="#888780", lw=1.2, linestyle=":", alpha=0.8)
    ax.text(val * 1.04, 0.92, name, fontsize=6, color="#888780", va="top")

ax.axhline(GAMMA_MIN, color="#aaa", lw=0.8, linestyle="--", alpha=0.5)
ax.axhline(GAMMA_MAX, color="#aaa", lw=0.8, linestyle="--", alpha=0.5)
ax.text(0.0052, GAMMA_MIN + 0.01, "γ_min", fontsize=9, color="#aaa")
ax.text(0.0052, GAMMA_MAX + 0.01, "γ_max", fontsize=9, color="#aaa")

ax.set_xlim(0, 5.0)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0, 3.0])
ax.tick_params(axis='x', rotation=45, labelsize=5)
ax.set_xlabel("nb_loss", fontsize=12)
ax.set_ylabel("gamma (γ)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.15)
ax.set_title("loss → gamma mapping", fontsize=13)

plt.tight_layout()
plt.savefig(f"linear_sigmoid_gamma_shapes_sigma_{SIGMA_SCALE}.png", dpi=150)
plt.show()