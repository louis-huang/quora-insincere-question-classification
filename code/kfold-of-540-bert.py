import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
import gc
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import gensim.models.keyedvectors as word2vec
import pickle
import matplotlib.pyplot as plt

from collections import defaultdict

print(os.listdir("../input"))

# configurations
seed = 2019
embed_size = 300 # how big is each word/sentence vector
max_features = 50000 # how many unique words to use
maxlen = 50 # max number of words in a question to use
n_thresh = 3 # best threshold value for f1 socre
score_dict = {}
n_epochs = 20
fine_tune = False
num_splits = 5
embedding_name = "BERT"
saving_name = "_{}_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}".format(embedding_name, embed_size,maxlen,n_epochs,fine_tune)

def get_threshold(pred_noemb_val_y,y_true, n = 1):
    range_list = np.arange(0.1, 0.501, 0.01)
    f1 = []
    for thresh in range_list:
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(y_true, (pred_noemb_val_y>thresh).astype(int))
        f1.append(score)
        #print("F1 score at threshold {0} is {1}".format(thresh, score))

    big_n_idx = np.array(f1).argsort()[-n:][::-1]
    print("Best {} F1 scores:".format(n))
    for i in big_n_idx:
        print("F1 score at threshold {0} is {1}".format(round(range_list[i],3), f1[i]))
    return f1[big_n_idx[0]]
# function to plot the history of training for all epochs
def plot_performance(history, file_name):
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    fig1 = plt.figure(figsize = (5,4))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(file_name +" accuacry")
    plt.show()
    fig1.savefig(file_name + "_accuracy_.png")
    
    
    # summarize history for loss
    fig2 = plt.figure(figsize = (5,4))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(file_name +" loss")
    plt.show()
    fig2.savefig(file_name + "_loss_.png")

def train_pred(model,train_X, train_y, val_X, val_y, epochs=2, callback=None):
    max_score = 0
    train_history = defaultdict(list)
    for e in range(epochs):
        cur_hist = model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=2)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        best_score = get_threshold(pred_val_y, val_y, n_thresh)
        max_score = max(best_score, max_score)
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))
        for k, v in cur_hist.history.items():
            train_history[k].append(v)

    print('=' * 60)
    return pred_val_y, max_score, train_history
    
dt1 = pd.read_csv("../input/subset_train.csv")
dt1.drop("Unnamed: 0", axis = 1, inplace = True)
dt2 = pd.read_csv("../input/data_left.csv")
dt = pd.concat([dt1,dt2], axis = 0, ignore_index = True)

e1 = np.load("../input/encoded_train_quora.npy")
e2 = np.load("../input/encoded_train_quora_1.npy")
e3 = np.load("../input/encoded_train_quora_2.npy")
e4 = np.load("../input/encoded_train_quora_3.npy")
e5 = np.load("../input/encoded_train_quora_4.npy")

e1 = np.vstack((e1, e2))
del e2
gc.collect()
e1 = np.vstack((e1, e3))
del e3
gc.collect()
e1 = np.vstack((e1, e4))
del e4
gc.collect()
e1 = np.vstack((e1, e5))
del e5
gc.collect()

time.sleep(10)

targets = dt["target"].values

del dt
gc.collect()

targets = targets.reshape(-1,1)

new_X, new_targets = shuffle(e1, targets, random_state = seed)

del e1, targets
gc.collect()
time.sleep(10)

splits = list(StratifiedKFold(n_splits=num_splits, random_state=seed).split(new_X, new_targets))

train_meta = np.zeros(new_targets.shape)
cur_scores = []
### Build models ### 
for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = new_X[train_idx]
        y_train = new_targets[train_idx]
        X_val = new_X[valid_idx]
        y_val = new_targets[valid_idx]
        #build model
        inp = Input(shape=(1024,))
        x = Dense(16, activation="relu")(inp)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        
        pred_val_y, best_score, history = train_pred(model, X_train, y_train, X_val, y_val, epochs = n_epochs)
        plot_performance(history, embedding_name+"Cross validation {}".format(idx))
        cur_scores.append(best_score)
        train_meta[valid_idx] = pred_val_y.reshape(-1,1)
        
        del X_train, y_train, X_val, y_val, inp, x, model,pred_val_y, best_score, history
        gc.collect()
        time.sleep(10)
        
score_dict[embedding_name] = [np.mean(np.array(cur_scores)), np.std(np.array(cur_scores))]

print(score_dict)

print(get_threshold(train_meta,new_targets))

#save as pickle
output = open(saving_name +".pkl", 'wb')
pickle.dump(score_dict, output)
output.close()

#save txt
with open(saving_name +".txt","a") as f:
    f.write(str(score_dict.keys()) + str(score_dict.values()))
f.close()
