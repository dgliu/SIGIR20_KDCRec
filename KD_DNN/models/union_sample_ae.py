import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from utils.progress import WorkSplitter
from models.predictor import predict
from evaluation.metrics import evaluate


class UnionSampleAE(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb,
                 gpu_on,
                 confidence,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.gpu_on = gpu_on
        self.confidence = confidence
        self.get_graph()
        if self.gpu_on:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim))
        self.mask = tf.placeholder(tf.float32, (None, self.input_dim))

        with tf.variable_scope('encode'):
            self.encode_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                                              name="Weights")
            self.encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim]), name="Bias")

            self.encoded = tf.nn.relu(tf.matmul(self.inputs, self.encode_weights) + self.encode_bias)

        with tf.variable_scope('decode'):
            self.decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim, self.output_dim], stddev=1 / 500.0),
                                              name="Weights")
            self.decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="Bias")
            self.prediction = tf.matmul(self.encoded, self.decode_weights) + self.decode_bias

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(self.encode_weights) + tf.nn.l2_loss(self.decode_weights)

            mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)), tf.zeros(tf.shape(self.inputs)))
            mf_loss = tf.square(self.inputs - self.prediction * mask)

            self.loss = tf.reduce_mean(
                self.confidence * self.mask * mf_loss + (1 - self.mask) * mf_loss) + self.lamb*tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    @staticmethod
    def get_batches(rating_matrix, marks, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        marks_batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
                marks_batches.append(marks[batch_index * batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
                marks_batches.append(marks[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches, marks_batches

    def train_model(self, rating_matrix, marks, matrix_valid, epoch, metric_names):
        batches, marks_batches = self.get_batches(rating_matrix, marks, self.batch_size)

        # Training
        best_AUC, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, [], [], [], [], []
        for i in tqdm(range(epoch)):
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense(), self.mask: marks_batches[step].todense()}
                _ = self.sess.run([self.optimizer], feed_dict=feed_dict)

            RQ, X, xBias = self.get_RQ(marks, rating_matrix)
            Y = self.get_Y()
            yBias = self.get_yBias()
            prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=matrix_valid, bias=yBias, gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_X, best_xBias, best_Y, best_yBias = RQ, X, xBias, Y, yBias

        return best_RQ, best_X, best_xBias, best_Y, best_yBias

    def get_RQ(self, marks, rating_matrix):
        batches, _ = self.get_batches(rating_matrix, marks, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.encode_weights), self.sess.run(self.encode_bias)

    def get_Y(self):
        return self.sess.run(self.decode_weights)

    def get_yBias(self):
        return self.sess.run(self.decode_bias)


def unionsampleae(matrix_train, matrix_valid, matrix_unif_train, iteration=100, lam=0.01, rank=50, seed=0,
                  batch_size=256, confidence=0.9, gpu_on=True, **unused):
    progress = WorkSplitter()

    progress.section("UnionSampleAE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("UnionSampleAE: Training")
    m, n = matrix_train.shape
    model = UnionSampleAE(n, rank, lamb=lam, batch_size=batch_size, gpu_on=gpu_on, confidence=confidence)
    metric_names = ['NLL', 'AUC']

    marks = sparse.csr_matrix(matrix_train.shape)
    marks[(matrix_train != 0).nonzero()] = 1

    matrix_train += matrix_unif_train
    RQ, X, xBias, Y, yBias = model.train_model(matrix_train, marks, matrix_valid, iteration, metric_names)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias