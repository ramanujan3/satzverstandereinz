from __future__ import division

"""
VOKABULAR
Vorhersager - Foreteller
"""



def build(n_hidden, m_output):

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, m_output]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([m_output]))
    }
    
    return

def train(x_trn, y_trn):
    return

def test(y_trn, y_tst):
    return


def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def run_lstm(x_trn, x_tst, y_trn, y_tst):

     # number of units in RNN cell
    n_hidden = 512
    n_input = len(x_trn[0])
    m_input = len(y_trn[0])

    build(n_hidden, n_input, m_input)

    n_input = len(x_trn[0])

    model = train(x_trn, y_trn)
    test(y_trn, y_tst)
    return