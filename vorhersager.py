from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
from sklearn.metrics import accuracy_score
# , f1_score, recall_score, precision_score

# from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

"""
VOKABULAR
Vorhersager - Foreteller
"""


def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

# Target log path
# logs_path = '/tmp/tensorflow/lstm_logs'
# writer = tf.summary.FileWriter(logs_path)


def RNN_KR(n_input, m_hidden, k_output):

    # inputs = Input(name='inputs', shape=[max_len])
    # layer = Embedding(max_words,50, input_length=max_len)(inputs)
    inputs = Input(name='inputs', shape=(n_input, 1))
    # layer = Embedding(input_dim=16,  # 10
    #                   output_dim=64,
    #                   input_length=n_input)(inputs)
    layer = LSTM(64)(inputs)  # (layer)
    layer = Dense(m_hidden, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(k_output, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    return Model(inputs=inputs, outputs=layer)


# def prep_KR(
#     max_words = 1000
#     max_len = 8
#     tok = Tokenizer(num_words=max_words)
#     tok.fit_on_texts(X_train)
#     sequences = tok.texts_to_sequences(X_train)
#     sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


def build_KR(lamba_lrate, n_input, m_hidden, k_output):
    model = RNN_KR(n_input, m_hidden, k_output)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


def train_KR(model, x_trn, y_trn, batch_size, epochs):
    model.fit(x_trn, y_trn, batch_size=batch_size, epochs=epochs,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
    return model


def test_KR(model, x_tst, y_tst):
    # test_sequences = tok.texts_to_sequences(x_tst)
    # test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(x_tst, y_tst)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return model


def final_KR(model, x_fnl, y_fnl):
    # test_sequences = tok.texts_to_sequences(x_tst)
    # test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    y_out = model.predict(x_fnl)

    y_pred = np.argmax(y_out, axis=1)
    print(y_pred)
    y_conf = np.amax(y_out, axis=1)
    print(y_conf)
    # y_pred = [max(y) for y in y_out]
    # print(y_pred[0])
    # print(y_pred[0].shape)
    # print(max(y_pred[0]))
    # y_predc = model.predict_classes(x_fnl)
    # print(y_predc[:10])
    return model, y_pred, y_conf


## ---------------------


def RNN(x, y, weights, biases, n_input, m_hidden, k_output):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # x = tf.reshape(x, [n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)
    # x = tf.split(x, n_input)

    # 2-layer LSTM, each layer has m_hidden units.
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(m_hidden),rnn.BasicLSTMCell(m_hidden)])
    rnn_cell = rnn.MultiRNNCell([tf.keras.layers.LSTMCell(m_hidden),tf.keras.layers.LSTMCell(m_hidden)])
    # rnn_cell = ([tf.keras.layers.LSTMCell(m_hidden),tf.keras.layers.LSTMCell(m_hidden)])
    # rnn_cell = rnn.BasicLSTMCell(rnn.BasicLSTMCell(m_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # rnn_cell = rnn.BasicLSTMCell(m_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # outputs = tf.keras.layers.RNN(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    # Appear to need the softmax
    return tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    # return tf.nn.softmax(tf.matmul(outputs, weights['out']) + biases['out'])


def build(lamba_lrate, n_input, m_hidden, k_output):

    # tf Graph input
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, k_output])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([m_hidden, k_output]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([k_output]))
    }

    vorhersagt = RNN(x, y, weights, biases, n_input, m_hidden, k_output)
    # vorhersagt = tf.reshape(vorhersagt, [-1])

    # Cost function and optimization definitions
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=vorhersagt, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lamba_lrate).minimize(cost)

    # Metric definitions
    delta_y = tf.equal(tf.argmax(vorhersagt, 1), tf.argmax(y, 1))
    error_signal = tf.reduce_mean(tf.cast(delta_y, tf.float32))

    return x, y, vorhersagt, cost, optimizer, error_signal  # , weights, biases


def train(sess, x_trn, y_trn, x, y, vorhersagt,
          cost, optimizer, error_signal, epochs, batch_size,
          n_input, m_hidden, k_output, start_time):

    step = 0
    acc_total = 0
    loss_total = 0

    # writer.add_graph(sess.graph)

    for epoch in range(epochs):

        i = 0
        epoch_loss = 0
        for i in range(int(len(x_trn) / batch_size)):

            start = i
            end = i + batch_size

            batch_x = np.array(x_trn[start:end])
            batch_y = np.array(y_trn[start:end])
            # batch_x = np.expand_dims(batch_x, axis=0)

            # print(batch_x)
            # print(batch_y)
            # print(x_trn.shape)
            # print(y_trn.shape)
            # print(batch_x.shape)
            # print(batch_y.shape)
            # assert all(x.shape == (n_input, 1) for x in batch_x)
            # assert all(y.shape == (k_output, 1) for y in batch_y)

            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            _, c = sess.run([optimizer, error_signal, cost, vorhersagt], feed_dict={x: batch_x, y: batch_y})

            epoch_loss += c

            i += batch_size

        print('       - epoch', epoch, '/', epochs, ', loss:', epoch_loss)

    pred = tf.round(tf.nn.sigmoid(vorhersagt)).eval({x: np.array(x_trn), y: np.array(y_trn)})

    # f1 = f1_score(np.array(y_tst), pred, average='macro')
    accuracy_sc = accuracy_score(np.array(y_trn), pred)
    # recall = recall_score(y_true=np.array(y_tst), y_pred= pred)
    # precision = precision_score(y_true=np.array(y_tst), y_pred=pred)

    # print("F1 Score:", f1)
    print("Train accuracy: ", accuracy_sc)
    # print("Recall:", recall)
    # print("Precision:", precision)

    return accuracy_sc


def test(sess, x_tst, y_tst, x, y, vorhersagt):

    pred = tf.round(tf.nn.sigmoid(vorhersagt)).eval({x: np.array(x_tst), y: np.array(y_tst)})

    # f1 = f1_score(np.array(y_tst), pred, average='macro')
    accuracy_sc = accuracy_score(np.array(y_tst), pred)
    # recall = recall_score(y_true=np.array(y_tst), y_pred= pred)
    # precision = precision_score(y_true=np.array(y_tst), y_pred=pred)

    # print("F1 Score:", f1)
    print("Test accuracy:", accuracy_sc)
    # print("Recall:", recall)
    # print("Precision:", precision)
    return accuracy_sc


def run_lstm(x_trn, x_tst, x_fnl, y_trn, y_tst, y_fnl, m_hidden, epochs, lamba_lrate, batch_size):
    start_time = time.time()

    # number of units in RNN cell
    n_input = len(x_trn[0])
    k_output = len(y_trn[0])

    x, y, vorhersagt, cost, optimizer, error_signal = build(lamba_lrate, n_input, m_hidden, k_output)
    print('     --- lstm build time: ', elapsed(time.time() - start_time), '\n')

    # Initializing the variables
    # init = tf.global_variables_initializer()
    # init = tf.local_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        # sess.run(init)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        train_acc = train(sess, x_trn, y_trn, x, y, vorhersagt,
                          cost, optimizer, error_signal, epochs, batch_size,
                          n_input, m_hidden, k_output, start_time)

        test_acc = test(sess, x_tst, y_tst, x, y, vorhersagt)
    # print('     --- total lstm time: ', elapsed(time.time() - start_time), '\n')

    return train_acc, test_acc


def run_lstm_KR(x_trn, x_tst, x_fnl, y_trn, y_tst, y_fnl, m_hidden, epochs, lamba_lrate, batch_size):
    start_time = time.time()

    # number of units in RNN cell
    n_input = len(x_trn[0])
    k_output = len(y_trn[0])

    model = build_KR(lamba_lrate, n_input, m_hidden, k_output)
    model = train_KR(model, x_trn, y_trn, batch_size, epochs)

    print(x_tst.shape)
    if (x_tst.shape[0] > 0):
        model = test_KR(model, x_tst, y_tst)
    _, y_pred, y_conf = final_KR(model, x_fnl, y_fnl)

    print('     --- total KR lstm time: ', elapsed(time.time() - start_time), '\n')
    return y_pred, y_conf
