import numpy as np
import pandas as pd

import os
from keras.utils import get_file

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

print("Downloading the dataset...")
download_path = os.path.join(os.path.abspath('.'), "mbti_1.csv")
get_file(download_path, 'https://storage.googleapis.com/fingal_trees/mbti_1.csv')

sns.set_style(style="dark")
sns.set_context('paper')

print("Reading the dataset...")
df = pd.read_csv(download_path)
df.head()

print("Plotting distribution of types...")
s = df.type.value_counts()
s.plot(kind='bar')
plt.show()

df_types = pd.DataFrame(s).reset_index()
df_types.columns = ['type', 'count']

sns.barplot(x='type', y='count', data=df_types)
plt.show()

labels = 'INFP'

def extract_type(row):
  #new_row = row.to_dict()
  t = row['type']
  for a, b in zip(t, labels):
    if a == b:
      row[b] = 1
    else:
      row[b] = 0
  return row

rev_labels = {k:v for k, v in zip(labels, 'ESTJ')}
# rev_labels = {'F': 'T', 'I': 'E', 'N': 'S', 'P': 'J'}

print("Transforming labels to categorical columns...")
df_pref = df[['type']].apply(extract_type, axis=1)

inv_ratios = 1 - df_pref.iloc[:, 1:].mean()
inv_ratios.index =  list('ESTJ')
#E    0.230432
#S    0.137983
#T    0.458905
#J    0.395850

print("Further split for plotting...")
ratios = df_pref.iloc[:, 1:].mean()
df_dict = defaultdict(list)
df_ind = []
for i, (k, v) in enumerate(ratios.to_dict().items()):
  df_ind.append(f"{k} or {rev_labels[k]}")
  df_ind.append(f"{k} or {rev_labels[k]}")

  df_dict['type'].append(1)
  df_dict['type'].append(0)

  df_dict['ratio'].append(v)
  df_dict['ratio'].append(1-v)

df_ratios = pd.DataFrame(df_dict, index=df_ind)
df_ratios.reset_index(inplace=True)

df_ratios.columns = ['type', 'first, second', 'ratio']
print(df_ratios)

a = sns.distplot(df.posts.apply(lambda x: len(x)))
a.axes.set_title("Length of the texts")

ax = sns.catplot(x='first, second', y='ratio', col='type', data=df_ratios, kind='bar', col_wrap=2, height=2.5)
plt.show()

for a in ax.axes:
  a.set_xticklabels([])
  a.set_xlabel('')
  a.set_title(a.get_title().split('=')[1])
