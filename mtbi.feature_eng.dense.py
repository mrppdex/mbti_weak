from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import os
import sys
import datetime
from time import time

import re
from collections import defaultdict, OrderedDict
import itertools
import pickle


df = pd.read_csv("mtbi_augmented_bert_512.csv")

def create_labels(df, source, labels):
    # 'INTJ'
    new_df_dict = defaultdict(list)
    for personality_type in df[source]:
      for i, l in enumerate(labels):
        new_df_dict[l].append(int(personality_type[i] == l))
    return pd.DataFrame(new_df_dict)
    #return np.array([x[0] == 'I', x[1] == 'N', x[2] == 'T', x[3] == 'J'], dtype=int)

label_df = create_labels(df, 'type', 'INTJ')

personality = "I"

X, y = RandomUnderSampler().fit_resample(df.posts.values[:,None],
                                         label_df[personality])

train_features, dev_features, train_labels, dev_labels = train_test_split(X, y, test_size=0.2)

train_features = train_features.flatten()
dev_features = dev_features.flatten()

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1500

train_df = pd.DataFrame({'posts': train_features,
                         'labels': train_labels})

eval_df = pd.DataFrame({'posts': dev_features,
                        'labels': dev_labels})

try:
    with open("tfid_vectorizer.pkl", "rb") as tf:
        tfid_vectorizer = pickle.load(tf)
    print("Tfid vectorizer restored...")
except:
    tfid_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  max_df=0.5,
                                  stop_words='english')
    tfid_vectorizer.fit(train_features)
    with open("tfid_vectorizer.pkl", "wb") as tf:
        pickle.dump(tfid_vectorizer, tf)
    print("Tfid vectorizer saved...")

tfid_features = tfid_vectorizer.transform(train_features)

try:
    with open("tsvd_red.pkl", "rb") as f:
        tsvd_red = pickle.load(f)
    print("TSVD object restored...")
except:
    tsvd_red = TruncatedSVD(n_components=500)
    tsvd_red.fit(tfid_features)
    with open("tsvd_red.pkl", "wb") as f:
        pickle.dump(tsvd_red, f)
    print("TSVD object saved...")

try:
    with open("posts_tokenizer.pkl", "rb") as f:
        posts_tokenizer = pickle.load(f)
    print("Tokenizer restored...")
except:
    posts_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    posts_tokenizer.fit_on_texts(train_features)
    with open("posts_tokenizer.pkl", "wb") as f:
        pickle.dump(posts_tokenizer, f)
    print("Tokenizer saved...")



stemmer = SnowballStemmer("english")

pos_columns = []


def preprocess_pipeline(df):

    def count_words(x):
      return len(x.split())

    def element_ratio(x, count):
      length = len(x.split())
      if length > 0:
        return float(count/length)
      return 0.

    def unique_words_ratio(x):
      words = x.split()

      if len(words) > 0:
        return float(len(set(words))/len(words))
      return 0.

    def process_emoji(x):
      regex = r":[^\s]+:"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)
      #return len(re.findall(regex, x))

    def exclamation_mark_count(x):
      regex = r"!"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def question_mark_count(x):
      regex = r"\?"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def capital_letters_count(x):
      regex = r"[^A-Z]"
      count = len(re.sub(regex, "", x))
      return element_ratio(x, count)

    def capital_letters_ratio(x):
      regex = r"[^A-Z]"
      cap = re.sub(regex, "", x)
      if len(x) > 0:
        return float(len(cap)/len(x))
      return 0.0

    def ellypsis_count(x):
      regex = r"\.\.\."
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def emoji_faces_count(x):
      regex = r"[;:]+[_-]?[\)\(]"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def capitalized_words_ratio(x):
      regex = r"[A-Z]{2,}\s"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)


    def pof_list(x):
      pof = nltk.pos_tag(word_tokenize(x))
      result = [p for _, p in pof]
      return result


    df["number_of_words"] = df["posts"].apply(count_words)
    df["emoji"] = df["posts"].apply(process_emoji)
    df["exclamation_mark_count"] = df["posts"].apply(exclamation_mark_count)
    df["question_mark_count"] = df["posts"].apply(question_mark_count)
    df["capital_letters_count"] = df["posts"].apply(capital_letters_count)
    df["ellypsis_count"] = df["posts"].apply(ellypsis_count)
    df["emoji_faces_count"] = df["posts"].apply(emoji_faces_count)
    df["capitalized_words_ratio"] = df["posts"].apply(capitalized_words_ratio)
    df["unique_words_ratio"] = df["posts"].apply(unique_words_ratio)

    t0 = time()

    print("**** Parts of Speech Decomposition ****")

    df["pof_list"] = df["posts"].apply(pof_list)

    print("Time: {:.2f} min".format((time()-t0)/60))

    convtag_dict={'ADJ':['JJ','JJR','JJS'], 'ADP':['EX','TO'], 'ADV':['RB','RBR','RBS','WRB'], 'CONJ':['CC','IN'],'DET':['DT','PDT','WDT'],
              'NOUN':['NN','NNS','NNP','NNPS'], 'NUM':['CD'],'PRT':['RP'],'PRON':['PRP','PRP$','WP','WP$'],
              'VERB':['MD','VB','VBD','VBG','VBN','VBP','VBZ'],'.':['#','$',"''",'(',')',',','.',':'],'X':['FW','LS','UH']}

    expanded_convtag = {}
    for pos_key, pos_list in convtag_dict.items():
      for el in pos_list:
        expanded_convtag[el] = pos_key

    def count_pos(x):
      counter = defaultdict(float)
      x_length = len(x)

      for w in x:
        if w in expanded_convtag.keys():
          counter[expanded_convtag[w]] += 1
        else:
          counter[w] += 1

      for k, v in counter.items():
        counter[k] = v/x_length

      counter_dict = OrderedDict()
      for k in sorted(convtag_dict.keys()):
          counter_dict[k] = counter.get(k,0.0)

      return counter_dict

    def count_all_pos(df, column):
      pos_df = None
      for i in range(len(df)):
        if pos_df is None:
          pos_df = pd.DataFrame(index=[0], data=count_pos(df.loc[i, column]))
        else:
          pos_df = pos_df.append(pd.DataFrame(index=[i], data=count_pos(df.loc[i, column])), sort=True)
      return pos_df

    pos_df = count_all_pos(df, 'pof_list')
    pos_df.fillna(0.0, inplace=True)

    tfid_features = tfid_vectorizer.transform(df.posts)

    tfid_features_dimred = tsvd_red.transform(tfid_features)

    dense_features = np.concatenate([df.iloc[:,3:-1].values, pos_df.values, tfid_features_dimred], axis=1)

    print('**** Initializing Scaler ***')
    try:
        with open("scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
    except FileNotFoundError:
        scaler = MinMaxScaler()
        scaler.fit(dense_features)
        with open("scaler.pkl", "wb") as sf:
            pickle.dump(scaler, sf)

    scaled_dense_features = scaler.transform(dense_features)

    y = df.labels.values

    return scaled_dense_features, y

'''
def preprocess_pipeline(df):

    def count_words(x):
      return len(x.split())

    def element_ratio(x, count):
      length = len(x.split())
      if length > 0:
        return float(count/length)
      return 0.

    def unique_words_ratio(x):
      words = x.split()

      if len(words) > 0:
        return float(len(set(words))/len(words))
      return 0.

    def process_emoji(x):
      regex = r":[^\s]+:"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)
      #return len(re.findall(regex, x))

    def exclamation_mark_count(x):
      regex = r"!"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def question_mark_count(x):
      regex = r"\?"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def capital_letters_count(x):
      regex = r"[^A-Z]"
      count = len(re.sub(regex, "", x))
      return element_ratio(x, count)

    def capital_letters_ratio(x):
      regex = r"[^A-Z]"
      cap = re.sub(regex, "", x)
      if len(x) > 0:
        return float(len(cap)/len(x))
      return 0.0

    def ellypsis_count(x):
      regex = r"\.\.\."
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def emoji_faces_count(x):
      regex = r"[;:]+[_-]?[\)\(]"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)

    def capitalized_words_ratio(x):
      regex = r"[A-Z]{2,}\s"
      count = len(re.findall(regex, x))
      return element_ratio(x, count)


    def pof_list(x):
      pof = nltk.pos_tag(word_tokenize(x))
      result = [p for _, p in pof]
      return result


    df["number_of_words"] = df["posts"].apply(count_words)
    df["emoji"] = df["posts"].apply(process_emoji)
    df["exclamation_mark_count"] = df["posts"].apply(exclamation_mark_count)
    df["question_mark_count"] = df["posts"].apply(question_mark_count)
    df["capital_letters_count"] = df["posts"].apply(capital_letters_count)
    df["ellypsis_count"] = df["posts"].apply(ellypsis_count)
    df["emoji_faces_count"] = df["posts"].apply(emoji_faces_count)
    df["capitalized_words_ratio"] = df["posts"].apply(capitalized_words_ratio)
    df["unique_words_ratio"] = df["posts"].apply(unique_words_ratio)

    t0 = time()

    print("**** Parts of Speech Decomposition ****")

    df["pof_list"] = df["posts"].apply(pof_list)

    print("Time: {:.2f} min".format((time()-t0)/60))


    convtag_dict={'ADJ':['JJ','JJR','JJS'], 'ADP':['EX','TO'], 'ADV':['RB','RBR','RBS','WRB'], 'CONJ':['CC','IN'],'DET':['DT','PDT','WDT'],
              'NOUN':['NN','NNS','NNP','NNPS'], 'NUM':['CD'],'PRT':['RP'],'PRON':['PRP','PRP$','WP','WP$'],
              'VERB':['MD','VB','VBD','VBG','VBN','VBP','VBZ'],'.':['#','$',"''",'(',')',',','.',':'],'X':['FW','LS','UH']}

    expanded_convtag = {}
    for pos_key, pos_list in convtag_dict.items():
      for el in pos_list:
        expanded_convtag[el] = pos_key

    def count_pos(x):
      counter = defaultdict(float)
      x_length = len(x)

      for w in x:
        if w in expanded_convtag.keys():
          counter[expanded_convtag[w]] += 1
        else:
          counter[w] += 1

      for k, v in counter.items():
        counter[k] = v/x_length

      counter_dict = OrderedDict()
      for k in sorted(convtag_dict.keys()):
          counter_dict.append(counter.get(k,0.0))

      return counter_dict

    def count_all_pos(df, column):
      pos_df = None
      for i in range(len(df)):
        if pos_df is None:
          pos_df = pd.DataFrame(index=[0], data=count_pos(df.loc[i, column]))
        else:
          pos_df = pos_df.append(pd.DataFrame(index=[i], data=count_pos(df.loc[i, column])), sort=True)
      return pos_df

    pos_df = count_all_pos(df, 'pof_list')
    pos_df.fillna(0.0, inplace=True)

    tfid_features = tfid_vectorizer.transform(df.posts)


    df["stem_posts"] = df["posts"].apply(lambda x: " ".join([stemmer.stem(w) for w in x.split()]))

    df["posts_split"] = df["posts"].apply(lambda x: x.split())
    df["post_sequence"] = df["posts_split"].apply(posts_tokenizer.texts_to_sequences)

    def pad_posts(x):

      a = np.array(list(itertools.chain(*x)))[None, :]

      post_seq = pad_sequences(a, maxlen=MAX_SEQUENCE_LENGTH)

      nd_output = np.array(post_seq)
      return nd_output

    df['post_sequence'] = df['post_sequence'].apply(pad_posts)


    tfid_features_dimred = tsvd_red.transform(tfid_features)

    if len(pos_columns)>0:
      for dc in pos_columns:
        if dc not in pos_df:
          pos_df[dc] = 0

    dense_features = np.concatenate([df.iloc[:,3:-4].values, tfid_features_dimred, pos_df.values], axis=1)

    print('**** Initializing Scaler ***')
    try:
        with open("scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
    except FileNotFoundError:
        scaler = MinMaxScaler()
        scaler.fit(dense_features)
        with open("scaler.pkl", "wb") as sf:
            pickle.dump(scaler, sf)

    scaled_dense_features = scaler.transform(dense_features)

    y = df.labels.values
    return scaled_dense_features, y
'''
train_x, train_y = preprocess_pipeline(train_df)
eval_x, eval_y = preprocess_pipeline(eval_df)

pd.DataFrame(np.hstack([train_x, train_y[:,None]])).to_csv("train_dense_dataset.csv", index=False)
pd.DataFrame(np.hstack([eval_x, eval_y[:,None]])).to_csv("eval_dense_dataset.csv", index=False)
