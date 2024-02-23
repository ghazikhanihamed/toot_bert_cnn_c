import matplotlib.pyplot as plt
import numpy as np
import os
from settings import settings

# Data from the table
tasks = ["IC-MP", "IT-MP", "IC-IT"]
accuracy_taju = [98.24, 95.98, 93.07]
accuracy_new = [99.49, 98.55, 95.35]
mcc_taju = [0.85, 0.69, 0.87]
mcc_new = [0.94, 0.90, 0.90]

# Indices and width for the bar plot
ind = np.arange(len(tasks))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting accuracy
ax1.bar(
    ind - width / 2,
    accuracy_taju,
    width,
    label="Taju et al.",
    color="skyblue",
    edgecolor="black",
)
ax1.bar(
    ind + width / 2,
    accuracy_new,
    width,
    label="Updated Dataset",
    color="lightgreen",
    edgecolor="black",
)

ax1.set_ylabel("Accuracy (%)", fontsize=14)
ax1.set_xticks(ind)
ax1.set_xticklabels(tasks, fontsize=12)
ax1.legend(loc="lower left", fontsize=12)

# Creating a twin y-axis to plot MCC
ax2 = ax1.twinx()
ax2.plot(
    ind - width / 2,
    mcc_taju,
    color="blue",
    marker="o",
    linestyle="None",
    markersize=10,
    label="MCC (Taju et al.)",
)
ax2.plot(
    ind + width / 2,
    mcc_new,
    color="green",
    marker="s",
    linestyle="None",
    markersize=10,
    label="MCC (Updated Dataset)",
)

ax2.set_ylabel("MCC", fontsize=14)
ax2.legend(loc="lower right", fontsize=12)

# Final layout adjustments
# plt.title('Comparative Performance on Independent Test Set', fontsize=16, pad=20)
fig.tight_layout()
# plt.show()
# save the plot
plt.savefig(
    os.path.join(settings.LATEX_PATH, "comparative_performance_new.png"),
    dpi=300,
    bbox_inches="tight",
)
