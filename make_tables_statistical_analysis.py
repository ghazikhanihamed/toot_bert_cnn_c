import os
import pandas as pd
from settings import settings
import numpy as np



# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results_trad_cnn.csv"))


aa=1