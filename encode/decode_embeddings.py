import pandas as pd
import numpy as np

from bert_serving.client import BertClient

output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_quora.npy"
doc_vecs = np.load(output_file)

