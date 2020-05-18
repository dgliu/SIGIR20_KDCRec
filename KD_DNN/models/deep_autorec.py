import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
from models.predictor import predict
from evaluation.metrics import evaluate


class DeepAutoRec(object):
    def __init__(self,
                 input_dim,
                 embed_dim1,
                 embed_dim2,
                 batch_size,
                 lamb,
                 gpu_on,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim1 = embed_dim1
        self.embed_dim2 = embed_dim2
        self.batch_size = batch_size
        self.lamb = lamb
        self.gpu_on = gpu_on
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

        with tf.variable_scope('encode'):
            self.first_encode_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.embed_dim1], stddev=1 / 500.0),
                                                    name="first_Weights")
            self.first_encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim1]), name="first_Bias")

            self.first_encoded = tf.nn.relu(tf.matmul(self.inputs, self.first_encode_weights) + self.first_encode_bias)

            self.second_encode_weights = tf.Variable(tf.truncated_normal([self.embed_dim1, self.embed_dim2], stddev=1 / 500.0),
                                                     name="second_Weights")
            self.second_encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim2]), name="second_Bias")

            self.encoded = tf.nn.relu(tf.matmul(self.first_encoded, self.second_encode_weights) + self.second_encode_bias)

        with tf.variable_scope('decode'):
            self.first_decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim2, self.embed_dim1], stddev=1 / 500.0),
                                                    name="first_Weights")
            self.first_decode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim1]), name="first_Bias")

            self.first_decode = tf.nn.relu(tf.matmul(self.encoded, self.first_decode_weights) + self.first_decode_bias)

            self.second_decode_weights = tf.Variable(tf.truncated_normal([self.embed_dim1, self.output_dim], stddev=1 / 500.0),
                                                     name="second_Weights")
            self.second_decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="second_Bias")

            self.prediction = tf.matmul(self.first_decode, self.second_decode_weights) + self.second_decode_bias

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(self.first_encode_weights) + tf.nn.l2_loss(self.second_encode_weights) + \
                      tf.nn.l2_loss(self.first_decode_weights) + tf.nn.l2_loss(self.second_decode_weights)

            mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)), tf.zeros(tf.shape(self.inputs)))
            mf_loss = tf.square(self.inputs - self.prediction * mask)

            self.loss = tf.reduce_mean(mf_loss) + self.lamb*tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    @staticmethod
    def get_batches(rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, matrix_valid, epoch, metric_names):
        batches = self.get_batches(rating_matrix, self.batch_size)

        # Training
        best_AUC, best_RQ, best_X, best_xBias, best_Y, best_yBias, best_Z, best_zBias, best_K, best_kBias = 0, [], [], [], [], [], [], [], [], []
        for i in tqdm(range(epoch)):
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                _ = self.sess.run([self.optimizer], feed_dict=feed_dict)

            RQ, X, xBias, Y, yBias, Z, zBias = self.get_RQ(rating_matrix)
            K = self.get_K()
            kBias = self.get_kBias()
            prediction = predict(matrix_U=RQ, matrix_V=K.T, matrix_Valid=matrix_valid, bias=yBias, gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_X, best_xBias, best_Y, best_yBias, best_Z, best_zBias, best_K, best_kBias = RQ, X, xBias, Y, yBias, Z, zBias, K, kBias

        return best_RQ, best_X, best_xBias, best_Y, best_yBias, best_Z, best_zBias, best_K, best_kBias

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.first_decode, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.first_encode_weights), self.sess.run(self.first_encode_bias), self.sess.run(self.second_encode_weights), self.sess.run(self.second_encode_bias), self.sess.run(self.first_decode_weights), self.sess.run(self.first_decode_bias)

    def get_K(self):
        return self.sess.run(self.second_decode_weights)

    def get_kBias(self):
        return self.sess.run(self.second_decode_bias)


def deepautorec(matrix_train, matrix_valid, matrix_unif_train, iteration=100, lam=0.01, rank=50, rank2=50, seed=0,
                batch_size=256, way=None, gpu_on=True, **unused):
    progress = WorkSplitter()

    progress.section("DeepAutoRec: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("DeepAutoRec: Training")
    m, n = matrix_train.shape
    model = DeepAutoRec(n, rank, rank2, lamb=lam, batch_size=batch_size, gpu_on=gpu_on)
    metric_names = ['NLL', 'AUC']

    if way == 'unif':
        RQ, X, xBias, Y, yBias, Z, zBias, K, kBias = model.train_model(matrix_unif_train, matrix_valid, iteration, metric_names)
    else:
        RQ, X, xBias, Y, yBias, Z, zBias, K, kBias = model.train_model(matrix_train, matrix_valid, iteration, metric_names)
    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias, Z, zBias, K, kBias