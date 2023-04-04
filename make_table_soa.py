import pandas as pd
import os
from settings import settings
import numpy as np


# We load the results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH + "results_best_test.csv"))

a=1