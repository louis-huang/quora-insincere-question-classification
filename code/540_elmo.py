# -*- coding: utf-8 -*-
"""540_elmo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MeTQopcCa_FZZG0Mf90zmi7AhMTsQxlZ
"""

import tensorflow as tf

import pandas as pd
import tensorflow_hub as hub
import re
from keras import backend as K
from keras.models import Model, load_model
from keras.engine import Layer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,Bidirectional,GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)


#prepare data
# configurations
seed = 2019
embed_size = 300  # how big is each word/sentence vector
max_features = 50000  # how many unique words to use
maxlen = 50  # max number of words in a question to use
n_thresh = 3  # best threshold value for f1 socre
score_dict = {}
n_epochs = 3
fine_tune = False

# define helper functions

# read subsets of data I created
def read_data():
    dt1 = pd.read_csv("drive/colab_data/540/subset_train.csv")
    dt1.drop("Unnamed: 0", axis=1, inplace=True)
    dt2 = pd.read_csv("drive/colab_data/540/data_left.csv")
    dt = pd.concat([dt1, dt2], axis=0, ignore_index=True)
    dt_3 = shuffle(dt, random_state=seed)
    return dt_3


# get the best threshold value for F1 score
def get_threshold(pred_noemb_val_y, y_true, n=1):
    range_list = np.arange(0.1, 0.501, 0.01)
    f1 = []
    for thresh in range_list:
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(y_true, (pred_noemb_val_y > thresh).astype(int))
        f1.append(score)
        # print("F1 score at threshold {0} is {1}".format(thresh, score))

    big_n_idx = np.array(f1).argsort()[-n:][::-1]
    print("Best {} F1 scores:".format(n))
    for i in big_n_idx:
        print("F1 score at threshold {0} is {1}".format(round(range_list[i], 3), f1[i]))
    return f1[big_n_idx[0]]


# function to plot the history of training for all epochs
def plot_performance(history, file_name):
    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    fig1 = plt.figure(figsize=(5, 4))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig1.savefig(file_name + "_accuracy_.png")

    # summarize history for loss
    fig2 = plt.figure(figsize=(5, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig2.savefig(file_name + "_loss_.png")


# Get word ans its embedding matrix from pretrain embedding file and return a dictionary
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# build model, train, save, predict and plot
def train_model(clf, embedding_matrix, fine_tune, embedding_name, record_dict):
    # train glove model
    history = clf.fit(train_X, train_y, batch_size=512, epochs=n_epochs, validation_data=(val_X, val_y), verbose=2)
    # save model
    clf.save("{}.h5".format(embedding_name))
    # find best threshold value for f1 score
    pred_noemb_val_y = clf.predict([val_X], batch_size=1024, verbose=2)
    score = get_threshold(pred_noemb_val_y, n_thresh)
    record_dict[embedding_name] = [score, history]
    # plot performance respect to epoch
    plot_performance(history, embedding_name)

    return record_dict

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = fine_tune
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Function to build model
def build_model_old():
    input_text = layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def build_model():
    inp = Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(inp)
    #x = Dense(256, activation="relu")(embedding)
    x = Dense(16, activation="relu")(embedding)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

### Prepare data
total_data = read_data()

# split data into training and testing
train_df, val_df = train_test_split(total_data, test_size=0.3, random_state=seed)
'''
# create x and y
train_X = train_df["question_text"].values
val_X = val_df["question_text"].values

train_y = train_df['target'].values
val_y = val_df['target'].values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)

## pad the sentences, so all sentences have same length
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
'''



# Create datasets (Only take up to 150 words for memory)
train_text = train_df['question_text'].tolist()
train_text = [' '.join(t.split()[0:50]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['target'].tolist()

test_text = val_df['question_text'].tolist()
test_text = [' '.join(t.split()[0:50]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = val_df['target'].tolist()

model = build_model()

history = model.fit(train_text,
          train_label,
          validation_data=(test_text, test_label),
          epochs=n_epochs,
          batch_size=256)

embedding_name = "Elmo"
model.save("{}.h5".format(embedding_name))

model = None
model = build_model()
model.load_weights('{}.h5'.format(embedding_name))

pred_noemb_val_y = model.predict(test_text, batch_size=256)
score = get_threshold(pred_noemb_val_y, test_label,n_thresh)
score_dict[embedding_name] = [score,history]
#plot performance respect to epoch
plot_performance(history,embedding_name)

#save as pickle
output = open("score_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}.pkl".format(embed_size,maxlen,n_epochs,fine_tune), 'wb')
pickle.dump(score_dict, output)
output.close()

#save txt
for k, v in score_dict.items():
    with open("result_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}.txt".format(embed_size,maxlen,n_epochs,fine_tune),"a") as f:
        f.write(k+","+str(score_dict[k][0]))
    f.close()
    performance_dict = score_dict[k][1].history
    with open("training_history_embed_{}_maxlen_{}_epochs_{}_fine_tune_{}.txt".format(embed_size,maxlen,n_epochs,fine_tune),"a") as f:
        for k,v in performance_dict.items(): 
            f.write(k)
            f.write(",")
            f.write(",".join(map(str, v)))
            f.write("\n")
    f.close()

