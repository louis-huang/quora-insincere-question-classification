import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
import gc
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

from sklearn import metrics

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model

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
score_dict = {} #save best f1 scores and std
n_epochs = 4
num_splits = 5
fine_tune = False
all_predictions = [] #save all predictions
#saving_name = "_{}_embed_{}_maxlen_{}_epochs_{}_{}_folds_fine_tune_{}".format(embedding_name, embed_size,maxlen,n_epochs,num_splits,fine_tune)

# define helper functions
#read subsets of data I created
def read_data():
    dt1 = pd.read_csv("../input/quora-data/subset_train.csv")
    dt1.drop("Unnamed: 0", axis = 1, inplace = True)
    dt2 = pd.read_csv("../input/quora-data/data_left.csv")
    dt = pd.concat([dt1,dt2], axis = 0, ignore_index = True)
    dt_3 = shuffle(dt, random_state = seed)
    return dt_3

#get the best threshold value for F1 score
def get_threshold(pred_noemb_val_y, y_true, n = 1):
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
# Get word ans its embedding matrix from pretrain embedding file and return a dictionary
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

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
    
def kfold_cv(embedding_matrix,embedding_name,score_dict, all_predictions):
    train_meta = np.zeros(train_y.shape)
    cur_scores = []
    for idx, (train_idx, valid_idx) in enumerate(splits):
            X_train = train_X[train_idx]
            y_train = train_y[train_idx]
            X_val = train_X[valid_idx]
            y_val = train_y[valid_idx]
            #build model
            inp = Input(shape=(maxlen,))
            if embedding_name == "no_pretrained":
                x = Embedding(max_features, embed_size)(inp)
            else:
                x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=fine_tune)(inp)
            x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
            x = GlobalMaxPool1D()(x)
            x = Dense(16, activation="relu")(x)
            x = Dropout(0.1)(x)
            x = Dense(1, activation="sigmoid")(x)
            model = Model(inputs=inp, outputs=x)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print(model.summary())
            
            pred_val_y, best_score, history = train_pred(model, X_train, y_train, X_val, y_val, epochs = n_epochs)
            plot_performance(history, embedding_name+" Cross validation {}".format(idx))
            cur_scores.append(best_score)
            train_meta[valid_idx] = pred_val_y.reshape(-1)
            
    score_dict[embedding_name] = [np.mean(np.array(cur_scores)), np.std(np.array(cur_scores))]
    all_predictions.append(train_meta)
    
    return score_dict, all_predictions
    
def laod_paragram():
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    #use lenght to filter out first line which is not word embedding
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    #embedding size is 300
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
    return embedding_matrix, "Paragram"

def load_glove():
    ##2 Use Glove Embeddings
    
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    embedding_name = "glove"
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    #embedding size is 300
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix, "Glove"
    
def load_fasttext():
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    #use lenght to filter out first line which is not word embedding
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    #embedding size is 300
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix, "FastText"
    

### Prepare data
total_data = read_data()
#create x and y
train_X = total_data["question_text"].values
train_y = total_data['target'].values
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
## pad the sentences, so all sentences have same length
train_X = pad_sequences(train_X, maxlen=maxlen)
#make splits
splits = list(StratifiedKFold(n_splits=num_splits, random_state=seed).split(train_X, train_y))

##############1 train no pretrained embeddings model
embedding_name = "no_pretrained"
print("{} embeddings loaded.".format(embedding_name))
score_dict, all_predictions = kfold_cv(None,embedding_name,score_dict, all_predictions)

del total_data
gc.collect()
time.sleep(10)

#############2 glove
embedding_matrix,embedding_name = load_glove()
print("{} embeddings loaded.".format(embedding_name))
score_dict, all_predictions = kfold_cv(embedding_matrix,embedding_name,score_dict, all_predictions)


del embedding_matrix
gc.collect()
time.sleep(10)

###3 FastText Embeddings
embedding_matrix,embedding_name = load_fasttext()
print("{} embeddings loaded.".format(embedding_name))
score_dict, all_predictions = kfold_cv(embedding_matrix,embedding_name,score_dict, all_predictions)

del embedding_matrix
gc.collect()
time.sleep(10)


##3 Paragram Embeddings
embedding_matrix,embedding_name = laod_paragram()
score_dict, all_predictions = kfold_cv(embedding_matrix,embedding_name,score_dict, all_predictions)

del embedding_matrix
gc.collect()
time.sleep(10)

##### save files
#save as pickle
print(score_dict)
output = open("score_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}.pkl".format(embed_size,maxlen,n_epochs,fine_tune), 'wb')
pickle.dump(score_dict, output)
output.close()

#save txt
for k, v in score_dict.items():
    with open("result_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}.txt".format(embed_size,maxlen,n_epochs,fine_tune),"a") as f:
        f.write(k+","+str(score_dict[k])+"\n")
    f.close()

sav = pd.DataFrame(np.array(all_predictions).T)
sav.to_csv("result of three embeddings.csv", index = False)








