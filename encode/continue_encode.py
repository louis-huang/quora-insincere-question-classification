import pandas as pd
import numpy as np
import time
from bert_serving.client import BertClient
import gc


output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_train_quora_4.npy"
'''
data_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/train.csv"

data = pd.read_csv(data_file)

already_done = pd.read_csv("/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/subset_train.csv")

already_done.drop("Unnamed: 0", inplace=True,axis=1)
new_data = data.merge(already_done,on = ['qid', 'question_text', 'target'],how = "left")

df=pd.merge(data,already_done,on=['qid', 'question_text', 'target'],how="outer",indicator=True)
df=df[df['_merge']=='left_only']
df.drop("_merge",inplace =True, axis = 1)
'''


df = pd.read_csv("/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/data_left.csv")
length = df.shape[0]
step = int(length / 180000)
inds = [180000*i for i in range(step + 1)]

#3/11/2019
#data = df.iloc[inds[0]:inds[1],:]
#df.to_csv("/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/data_left.csv", index = None)

#3/12/2019
#data = df.iloc[inds[1]:inds[2],:]

#3/13/2019
data = df.iloc[inds[4]:,:]


questions = data["question_text"].tolist()

del df,data
gc.collect()
print("questions loaded")
start = time.time()
with BertClient(check_length = False) as bc:
    doc_vecs = bc.encode(questions)
    print("questions encoded")
    np.save(output_file, doc_vecs)
print("Time used:{}".format(time.time() - start))

