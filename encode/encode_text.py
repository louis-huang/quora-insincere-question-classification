import pandas as pd
import numpy as np
import time
from bert_serving.client import BertClient

#import data
output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_train_quora.npy"
data_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/train.csv"
data = pd.read_csv(data_file)
data = data.sort_values("qid")
#subset a part of data for debug
#data = data[:100]

questions = data["question_text"].tolist()

print("questions loaded")
start = time.time()
with BertClient(check_length = False) as bc:
    doc_vecs = bc.encode(questions)
    print("questions encoded")
    np.save(output_file, doc_vecs)

print("Time used:{}".format(time.time() - start))