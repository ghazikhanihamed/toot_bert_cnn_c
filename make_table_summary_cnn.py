import os
import pandas as pd
from settings import settings
import numpy as np



# We read the csv file of the full results
df = pd.read_csv(os.path.join(settings.RESULTS_PATH,
                 "mean_balanced_imbalanced_results_trad_cnn.csv"))

# separate mean and std dev in MCC column
df[['Mean MCC', 'Std MCC']] = df['MCC'].str.split('±', expand=True)

# concatenate mean and std dev with '±' separator
df['MCC'] = df['Mean MCC'] + '±' + df['Std MCC']

# order of tasks
task_order = ['ionchannels_membraneproteins', 'iontransporters_membraneproteins', 'ionchannels_iontransporters']

# create pivot table
pivot_table_task = pd.pivot_table(df, values='MCC', index=['Task', 'Representation'], columns=['Representer'], aggfunc='first', fill_value='-')

# reindex to get the tasks in the desired order and get each task with its frozen and fine-tuned representations next to each other
pivot_table_task = pivot_table_task.reindex(pd.MultiIndex.from_product([task_order, ['frozen', 'finetuned']], names=['Task', 'Representation']))

print(pivot_table_task)





# convert MCC to a numerical column by taking the mean of the range
df['MCC'] = df['MCC'].str.split('±').str[0].astype(float)

# create a pivot table
pivot_table_rep = pd.pivot_table(df, values='MCC', index=['Representation'], columns=['Representer'], aggfunc=np.mean)


pivot_table_classifier = pd.pivot_table(df, values='MCC', index=['Representation', 'Classifier'], columns=['Representer'], aggfunc=np.mean)


pivot_table_task = pd.pivot_table(df, values='MCC', index=['Representation', 'Task'], columns=['Representer'], aggfunc=np.mean)


aa=1