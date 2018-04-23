import tensorflow as tf
from tensorflow.python import debug as tf_debug

class LSTM_model:

    def __init__(self, n_labels, data, label, hidden=75):
        self._data = data
        self._n_labels = n_labels
        self._label = label
        self._hidden = hidden
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden)
        self._out, state = tf.nn.dynamic_rnn(lstm_cell,
                                             data,
                                             dtype=tf.float32,
                                             time_major=False)

        self._predict = None
        self._loss = None
        self._optimizer = None
        self._accuracy = None
        self._prob = None

        self.prob
        self.predict
        self.loss
        self.optimizer
        self.accuracy

    @property
    def prob(self):
        if self._prob is None:
            weight = tf.Variable(tf.truncated_normal([self._hidden,
                                                      self._n_labels]))
            bias = tf.Variable(tf.zeros(self._n_labels))

            self._prob = tf.nn.softmax(tf.matmul(self._out[:, -1],
                                                 weight) + bias)

        return self._prob

    @property
    def predict(self):
        if self._predict is None:
            self._predict = tf.argmax(self.prob, axis=1)
        return self._predict

    @property
    def loss(self):
        if self._loss is None:
            self._loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(onehot_labels=self._label,
                                                logits=self.prob))
        return self._loss

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer =  tf.train.AdamOptimizer(
            ).minimize(self.loss)
        return self._optimizer

    @property
    def accuracy(self):
        if self._accuracy is None:
            self._accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.prob, axis=1),
                                 tf.argmax(self._label, axis=1)),
                        tf.float32))

        return self._accuracy
