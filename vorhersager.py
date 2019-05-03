from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

"""
VOKABULAR
Vorhersager - Foreteller
"""


def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Target log path
logs_path = '/tmp/tensorflow/lstm_logs'
writer = tf.summary.FileWriter(logs_path)

def RNN(x, y, weights, biases, n_input, m_hidden, k_output):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # x = tf.reshape(x, [n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)
    # x = tf.split(x, n_input)

    # 2-layer LSTM, each layer has m_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(m_hidden),rnn.BasicLSTMCell(m_hidden)])
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
    correct_pred = tf.equal(tf.argmax(vorhersagt,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return x, y, vorhersagt, cost, optimizer, accuracy # , weights, biases


def train(sess, x_trn, y_trn, x, y, vorhersagt, 
          cost, optimizer, accuracy, epochs, batch_size, disp_freq, 
          n_input, m_hidden, k_output, start_time):

    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(sess.graph)

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
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # _, c = sess.run([optimizer, accuracy, cost, vorhersagt], feed_dict={x: batch_x, y: batch_y})

            epoch_loss += c

            i += batch_size

        print('       - epoch', epoch, '/', epochs, ', loss:', epoch_loss)

    pred = tf.round(tf.nn.sigmoid(vorhersagt)).eval({x: np.array(x_trn), y: np.array(y_trn)})

    # f1 = f1_score(np.array(y_tst), pred, average='macro')
    accuracy = accuracy_score(np.array(y_trn), pred)
    # recall = recall_score(y_true=np.array(y_tst), y_pred= pred)
    # precision = precision_score(y_true=np.array(y_tst), y_pred=pred)

    # print("F1 Score:", f1)
    print("Train accuracy: ", accuracy)
    # print("Recall:", recall)
    # print("Precision:", precision)


    #     _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, vorhersagt], \
    #                                             feed_dict={x: x_trn, y: y_trn})
    #     loss_total += loss
    #     acc_total += acc
    #     if (epoch + 1) % disp_freq == 0:
    #         print("Iter= " + str(step+1) + ", Average Loss= " + \
    #               "{:.6f}".format(loss_total/disp_freq) + ", Average Accuracy= " + \
    #               "{:.2f}%".format(100*acc_total/disp_freq))
    #         acc_total = 0
    #         loss_total = 0

    # print("Optimization Finished!")
    # print("Elapsed time: ", elapsed(time.time() - start_time))
    # print("Run on command line.")
    # print("\ttensorboard --logdir=%s" % (logs_path))
    # print("Point your web browser to: http://localhost:6006/")

    return


def test(sess, x_tst, y_tst, x, y, vorhersagt):

    pred = tf.round(tf.nn.sigmoid(vorhersagt)).eval({x: np.array(x_tst), y: np.array(y_tst)})

    # f1 = f1_score(np.array(y_tst), pred, average='macro')
    accuracy = accuracy_score(np.array(y_tst), pred)
    # recall = recall_score(y_true=np.array(y_tst), y_pred= pred)
    # precision = precision_score(y_true=np.array(y_tst), y_pred=pred)

    # print("F1 Score:", f1)
    print("Test accuracy:", accuracy)
    # print("Recall:", recall)
    # print("Precision:", precision)
    return


def run_lstm(x_trn, x_tst, y_trn, y_tst, m_hidden, epochs, lamba_lrate, batch_size):

    start_time = time.time()

    # Parameters
    # lamba_lrate = 0.001
    # epochs = 7
    disp_freq = 1000
    # batch_size = 13

     # number of units in RNN cell
    n_input = len(x_trn[0])
    # m_hidden = 128
    k_output = len(y_trn[0])


    # print(type(x_trn))
    # print(type(y_trn))

    # x, y, weights, biases = 
    x, y, vorhersagt, cost, optimizer, accuracy = build(lamba_lrate, n_input, m_hidden, k_output)
    print('     --- lstm build time: ', elapsed(time.time() - start_time), '\n')

    # Initializing the variables
    #init = tf.global_variables_initializer()

    #init = tf.local_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        #sess.run(init)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        train(sess, x_trn, y_trn, x, y, vorhersagt, 
            cost, optimizer, accuracy, epochs, batch_size, disp_freq, 
            n_input, m_hidden, k_output, start_time) # x, y, weights, biases, n_input, k_output, x_trn, y_trn)
        test(sess, x_tst, y_tst, x, y, vorhersagt)

    # print('     --- total lstm time: ', elapsed(time.time() - start_time), '\n')

    return