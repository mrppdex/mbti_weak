import os
from keras.utils import get_file

import pandas as pd
from collections import defaultdict

print("Downloading the dataset...")
download_path = os.path.join(os.path.abspath('.'), "mbti_1.csv")
get_file(download_path, 'https://storage.googleapis.com/fingal_trees/mbti_1.csv')

print("Reading the dataset...")
df = pd.read_csv(download_path)


df['posts'] = df.apply(lambda x: x['posts'].split('|||'), axis=1)

new_df_dict = defaultdict(list)

def augment_data(x, max_len=1500):
  t = x['type']
  segments = []
  #row_dict = x.to_dict()
  segment_len = 0
  for p in x['posts']: #.split('|||'):
    p_len = len(p)
    if segment_len + p_len <= max_len:
      segment_len += p_len
      segments.append(p)
    elif p_len <= max_len:
      new_df_dict['type'].append(t)
      new_df_dict['posts'].append(segments)
      segment_len = p_len
      segments = [p]
    else:
      continue

  return

print("Augmenting data...")
#print(df.iloc[0,1])
_ = df.apply(augment_data, axis=1)
df_augmented = pd.DataFrame(new_df_dict)

print("Saving augmented csv...")
df_augmented['posts'] = df_augmented.posts.apply(lambda x: "|||".join(x))
df_augmented.to_csv("mtbi_augmented_bert_512.csv", index=False)

#sns.distplot(df_augmented.posts.apply(lambda x: len(x)))

#print(df_augmented.type.value_counts())
