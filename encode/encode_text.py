import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

from bert_serving.client import BertClient

output_file = "encoded_quora.npy"
data_file = "train.csv"

data = data.sort_values("qid")

questions = data["question_text"]

print("questions loaded")

with BertClient(check_length = False) as bc:
    doc_vecs = bc.encode(questions)
    print("questions encoded")
    np.save(output_file, doc_vecs)
