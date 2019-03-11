import pandas as pd
import numpy as np
import time
from bert_serving.client import BertClient
from sklearn.utils import shuffle


#import data
#output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_train_quora.npy"
output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_test_quora.npy"

#data_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/train.csv"
data_file  = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/test.csv"
data = pd.read_csv(data_file)

#subset a part of data for debug
#data = data[:100]
'''
negative = data.loc[data["target"] == 1]

positive = data.loc[data["target"] == 0]

positive_subset = positive.sample(n=negative.shape[0]*2, random_state=1024,replace = False, axis = 0)

new_data = pd.concat([negative, positive_subset], ignore_index=True)
new_data = shuffle(new_data, random_state = 1024)
new_data.to_csv("/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/subset_train.csv")
questions = new_data["question_text"].tolist()
'''
questions = data["question_text"].tolist()


print("questions loaded")
start = time.time()
with BertClient(check_length = False) as bc:
    doc_vecs = bc.encode(questions)
    print("questions encoded")
    np.save(output_file, doc_vecs)

print("Time used:{}".format(time.time() - start))