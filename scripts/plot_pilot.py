"""Plot the CartPole pilot: ES training curve + test on new random initialisations.

    python -m scripts.plot_pilot
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

res = Path("results")
gens, mean_r, best_r = [], [], []
with open(res / "pilot_cartpole_log.csv") as f:
    for row in csv.DictReader(f):
        gens.append(int(row["generation"]))
        mean_r.append(float(row["mean_return"]))
        best_r.append(float(row["best_return"]))
test = {}
with open(res / "pilot_cartpole_test.csv") as f:
    for row in csv.DictReader(f):
        test[row["condition"]] = (float(row["mean_return"]), float(row["std_return"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
ax1.plot(gens, best_r, color="#9FD8CF", lw=1.2, label="best in population")
ax1.plot(gens, mean_r, color="#0E9F8E", lw=2.2, label="population mean")
ax1.set_xlabel("ES generation")
ax1.set_ylabel("Episode return (max 500)")
ax1.set_title("Evolving Hebbian rules (weights random at every episode start)", fontsize=11)
ax1.legend(frameon=False)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

labels = ["Plasticity OFF\n(random weights)", "Evolved Hebbian\nrules ON"]
vals = [test["random_weights_no_plasticity"], test["hebbian_plastic"]]
ax2.bar(labels, [v[0] for v in vals], yerr=[v[1] for v in vals], color=["#B9C4C9", "#0E9F8E"], capsize=4)
for i, v in enumerate(vals):
    ax2.text(i, v[0] + v[1] + 8, f"{v[0]:.0f}", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 560)
ax2.set_title("Test: 100 new random inits", fontsize=11)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(res / "pilot_cartpole.png", dpi=160)
print("saved", res / "pilot_cartpole.png")
