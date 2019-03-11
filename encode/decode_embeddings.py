import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

output_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_train_quora.npy"
input_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/subset_train.csv"
out_dir = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/model/"
test_embeddings_file = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/word embeddings/encoded_test_quora.npy"
sub_dir = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/data/sample_submission.csv"
prediction_dir = "/Users/gaoyuan.huang/Desktop/school/quora-insincere-question-classification/submission/"

doc_vecs = np.load(output_file)

data = pd.read_csv(input_file)


#train model


def save_model(model):
    output = open(out_dir + 'logistic_model.pkl', 'wb')
    pickle.dump(model, output)
    output.close()


x_tr, x_te, y_tr, y_te = train_test_split(doc_vecs, data.target, random_state=1024)

clf = LogisticRegression()
clf.fit(x_tr,y_tr)

y_pred = clf.predict(x_te)

print(f1_score(y_te, y_pred))
print(confusion_matrix(y_te, y_pred))

y_pred = clf.predict_proba(x_te)[:,1]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(y_te, (y_pred>thresh).astype(int))))
#F1 score at threshold 0.39 is 0.8608391108569795
clf = LogisticRegression()
clf.fit(doc_vecs, data.target)
save_model(clf)





#make predictions
model_file = open(out_dir + 'logistic_model.pkl', 'rb')
clf = pickle.load(model_file)
model_file.close()


test_vecs = np.load(test_embeddings_file)
y_pred_proba = clf.predict_proba(test_vecs)
thresh = 0.39
y_pred = np.array((y_pred_proba[:,1] > thresh).astype(int))

submission = pd.read_csv(sub_dir)

submission.prediction = y_pred

submission.to_csv(prediction_dir + "logistic_model.csv",index = None)