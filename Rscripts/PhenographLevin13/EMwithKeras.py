#http://sujitpal.blogspot.com/2018/02/generating-labels-with-lu-learning-case.html

from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = '../../data/common'

VOCAB_SIZE = 5000

# convert texts to int sequence
def convert_to_intseq(text, vocab_size=VOCAB_SIZE, hash_fn="md5", lower=True):
    return hashing_trick(text, n=vocab_size, hash_function=hash_fn, lower=lower)

xs_l = [convert_to_intseq(text) for text in texts_l]
xs_v = [convert_to_intseq(text) for text in texts_v]
xs_u = [convert_to_intseq(text) for text in texts_u]

# pad to equal length input
maxlen = max([max([len(x) for x in xs_l]),
              max([len(x) for x in xs_v]),
              max([len(x) for x in xs_u])])

Xl = pad_sequences(xs_l, maxlen=maxlen)
Xv = pad_sequences(xs_v, maxlen=maxlen)
Xu = pad_sequences(xs_u, maxlen=maxlen)

# labels are 1-based, making it 0 based for to_categorical
Yl = np_utils.to_categorical(np.array(labels_l)-1, num_classes=NUM_CLASSES)
Yv = np_utils.to_categorical(np.array(labels_v)-1, num_classes=NUM_CLASSES)

print(Xl.shape, Yl.shape, Xv.shape, Yv.shape, Xu.shape)


EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
NUM_CLASSES = 4

BATCH_SIZE = 64
NUM_EPOCHS = 5

def build_model(maxlen=0, vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE,
                num_filters=NUM_FILTERS, kernel_size=NUM_WORDS,
                num_classes=NUM_CLASSES, print_model=False):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                     activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation="softmax"))
    # compile
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    if print_model:
        model.summary()
    return model


def train_model(model, X, Y, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                verbosity=0, model_name_template=MODEL_TEMPLATE,
                iter_num=0):
    best_model_fn = model_name_template.format(iter_num)
    checkpoint = ModelCheckpoint(filepath=best_model_fn, save_best_only=True)
    history = model.fit(X, Y, batch_size=batch_size, epochs=num_epochs,
                        verbose=verbosity, validation_split=0.1,
                        callbacks=[checkpoint])
    return model, history


def evaluation_report(model_path, X, Y):
    model = load_model(model_path)
    Y_ = model.predict(X)
    y = np.argmax(Y, axis=1)
    y_ = np.argmax(Y_, axis=1)
    acc = accuracy_score(y, y_)
    cm = confusion_matrix(y, y_)
    cr = classification_report(y, y_)
    print("\naccuracy: {:.3f}".format(acc))
    print("\nconfusion matrix")
    print(cm)
    print("\nclassification report")
    print(cr)

#We then build our initial model, training it against the labeled set of 9,000 records, and evaluating the trained model against the labeled validation set of 1,000 records.

MODEL_TEMPLATE = os.path.join(DATA_DIR, "keras-lu-{:d}.h5")

model = build_model(maxlen=maxlen, vocab_size=VOCAB_SIZE,
                    embed_size=EMBED_SIZE, num_filters=NUM_FILTERS,
                    kernel_size=NUM_WORDS, print_model=True)
model, _ = train_model(model, Xl, Yl, batch_size=BATCH_SIZE,
                       num_epochs=NUM_EPOCHS, verbosity=1,
                       model_name_template=MODEL_TEMPLATE)
evaluation_report(MODEL_TEMPLATE.format(0), Xv, Yv)


BEST_MODEL_EM = os.path.join(DATA_DIR, "keras-em-best.h5")

def e_step(model, Xu, Yu=None):
    if Yu is None:
        # predict labels for unlabeled set U with current model
        return np.argmax(model.predict(Xu), axis=1)
    else:
        # reuse prediction we got from M-step
        return np.argmax(Yu, axis=1)


def m_step(Xl, Yl, Xu, yu, iter_num, **kwargs):
    # train a model on the combined set L+U
    model = build_model(maxlen=kwargs["maxlen"],
                        vocab_size=kwargs["vocab_size"],
                        embed_size=kwargs["embed_size"],
                        num_filters=kwargs["num_filters"],
                        kernel_size=kwargs["kernel_size"],
                        print_model=False)
    X = np.concatenate([Xl, Xu], axis=0)
    Y = np.concatenate([Yl, np_utils.to_categorical(yu)], axis=0)
    model, _ = train_model(model, X, Y,
                          batch_size=kwargs["batch_size"],
                          num_epochs=kwargs["num_epochs"],
                          verbosity=1,
                          model_name_template=kwargs["model_name_template"],
                          iter_num=iter_num+1)
    # load new model
    model = load_model(kwargs["model_name_template"].format(iter_num+1))
    return model


# expectation maximization loop
epsilon = 1e-3
model = load_model(MODEL_TEMPLATE.format(0))
q = None
prev_loss = None
losses = []
for i in range(10):
    # E-step (prediction on U)
    p = e_step(model, Xu, q)
    # M-step (train on L+U, prediction on U, compute log-loss)
    model = m_step(Xl, Yl, Xu, p, i, maxlen=maxlen,
                   vocab_size=VOCAB_SIZE,
                   embed_size=EMBED_SIZE,
                   num_filters=NUM_FILTERS,
                   kernel_size=NUM_WORDS,
                   batch_size=BATCH_SIZE,
                   num_epochs=NUM_EPOCHS,
                   model_name_template=MODEL_TEMPLATE)
    q = model.predict(Xu)
    loss = log_loss(p, q)
    losses.append(loss)
    print("\n**** Iteration {:d}, log-loss: {:.7f} ****\n\n".format(i+1, loss))
    if prev_loss is None:
        model.save(BEST_MODEL_EM)
    else:
        if loss < prev_loss:
            model.save(BEST_MODEL_EM)
        if math.fabs(prev_loss - loss) < epsilon:
            break
    prev_loss = loss
