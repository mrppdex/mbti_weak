from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import re
from collections import defaultdict
import itertools
import pickle

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1500

#sample_text = "hi all. i'm an extrovert and I don't like to be pushed around"

sample_text = '''
You, like your hero Trump, are a person of very limited intelligence if you cannot see that there is a compassionate middle ground in dealing with migration. Migration needs compassionate controls. How many of the Nutters-clan are in the UK or USA? I will wager that, like most Irish families, the number is high. Or perhaps you think the Nutter-clan are somehow different because they are Caucasian and not South American or African?
'''

try:
    with open("tfid_vectorizer.pkl", "rb") as tf:
        tfid_vectorizer = pickle.load(tf)
    print("Tfid vectorizer restored...")
except:
    print("Tfid vectorizer does not exist...")

try:
    with open("tsvd_red.pkl", "rb") as f:
        tsvd_red = pickle.load(f)
    print("TSVD object restored...")
except:
    print("TSVD object does not exist...")

try:
    with open("posts_tokenizer.pkl", "rb") as f:
        posts_tokenizer = pickle.load(f)
    print("Tokenizer restored...")
except:
    print("Tokenizer does not exist...")

stemmer = SnowballStemmer("english")

def preprocess_pipeline(text):

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

    f_vec = []

    #f_vec.append(count_words(text))
    f_vec.append(process_emoji(text))
    f_vec.append(exclamation_mark_count(text))
    f_vec.append(question_mark_count(text))
    f_vec.append(capital_letters_count(text))
    f_vec.append(ellypsis_count(text))
    f_vec.append(emoji_faces_count(text))
    f_vec.append(capitalized_words_ratio(text))
    f_vec.append(unique_words_ratio(text))

    f_vec_poflist = pof_list(text)

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

      return counter

    pos_count = count_pos(f_vec_poflist)
    for k in sorted(convtag_dict.keys()):
        f_vec.append(pos_count.get(k,0.0))

    tfid_features = tfid_vectorizer.transform([text])


    post_stemmed = " ".join([stemmer.stem(w) for w in text.split()])
    post_splitted = post_stemmed.split()
    post_sequence = posts_tokenizer.texts_to_sequences(post_splitted)

    def pad_posts(x):

      a = np.array(list(itertools.chain(*x)))[None, :]

      post_seq = pad_sequences(a, maxlen=MAX_SEQUENCE_LENGTH)

      nd_output = np.array(post_seq)
      return nd_output

    post_padded_sequence = pad_posts(post_sequence)


    tfid_features_dimred = tsvd_red.transform(tfid_features)

    dense_features = np.hstack([f_vec, tfid_features_dimred.flatten()])
    #dense_features = np.concatenate([df.iloc[:,3:-4].values, tfid_features_dimred, pos_df.values], axis=1)

    print('**** Initializing Scaler ***')
    try:
        with open("scaler.pkl", "rb") as sf:
            scaler = pickle.load(sf)
    except FileNotFoundError:
        print("Scaler object does not exist...")

    #scaled_dense_features = scaler.transform(dense_features)

    return dense_features

feature = preprocess_pipeline(sample_text)
print(f"shape: {feature.shape}")

from keras.models import load_model
model = load_model("dense_model.h5")
pred = model.predict(feature[None, :])
print("Prediction: ", pred)
