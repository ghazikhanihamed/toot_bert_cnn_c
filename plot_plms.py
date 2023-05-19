import matplotlib.pyplot as plt
from settings.settings import (
    PLM_ORDER_SHORT,
    PLOT_PATH
)
import os

# extract the model sizes in billions of parameters for plotting
sizes_in_billion = [0.42, 0.42, 0.65, 0.65, 3, 15]

# create the horizontal bar plot
plt.figure(figsize=(8,6))
bars = plt.barh(PLM_ORDER_SHORT, sizes_in_billion, color='skyblue')

# adding value labels to the right of the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2.0, round(bar.get_width(),2), 
             ha='left', va='center')  # ha: horizontal alignment x positional argument

plt.ylabel('Protein Language Models', fontsize=12)  # y-axis label
plt.xlabel('Size in billions of parameters', fontsize=12)  # x-axis label
# plt.title('Size of Protein Language Models', fontsize=16)  # title of the plot

plt.tight_layout()  # to ensure the whole labels are visible in the saved figure
plt.savefig(os.path.join(
    PLOT_PATH, "plms_size.png"), bbox_inches='tight', dpi=300)  # save the figure
# plt.show()  # display the figure
