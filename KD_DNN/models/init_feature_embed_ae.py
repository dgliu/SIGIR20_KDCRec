import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
from models.predictor import predict
from evaluation.metrics import evaluate


class InitFeatureEmbedAE(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb,
                 gpu_on,
                 init_X,
                 init_Y,
                 init_xBias,
                 init_yBias,
                 way,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.init_X = init_X
        self.init_Y = init_Y
        self.init_xBias = init_xBias
        self.init_yBias = init_yBias
        self.way = way
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
            if self.way == 'both':
                self.encode_weights = tf.get_variable('Weights', initializer=self.init_X)
                self.encode_bias = tf.get_variable('Bias', initializer=self.init_xBias)

            elif self.way == 'user':
                self.encode_weights = tf.get_variable('Weights', initializer=self.init_X)
                self.encode_bias = tf.get_variable('Bias', initializer=self.init_xBias)

            elif self.way == 'item':
                self.encode_weights = tf.Variable(
                    tf.truncated_normal([self.input_dim, self.embed_dim], stddev=1 / 500.0),
                    name="Weights")
                self.encode_bias = tf.Variable(tf.constant(0., shape=[self.embed_dim]), name="Bias")

            self.encoded = tf.nn.relu(tf.matmul(self.inputs, self.encode_weights) + self.encode_bias)

        with tf.variable_scope('decode'):
            if self.way == 'both':
                self.decode_weights = tf.get_variable('Weights', initializer=self.init_Y)
                self.decode_bias = tf.get_variable('Bias', initializer=self.init_yBias)

            elif self.way == 'user':
                self.decode_weights = tf.Variable(
                    tf.truncated_normal([self.embed_dim, self.output_dim], stddev=1 / 500.0),
                    name="Weights")
                self.decode_bias = tf.Variable(tf.constant(0., shape=[self.output_dim]), name="Bias")

            elif self.way == 'item':
                self.decode_weights = tf.get_variable('Weights', initializer=self.init_Y)
                self.decode_bias = tf.get_variable('Bias', initializer=self.init_yBias)

            self.prediction = tf.matmul(self.encoded, self.decode_weights) + self.decode_bias

        with tf.variable_scope('loss'):
            l2_loss = tf.nn.l2_loss(self.encode_weights) + tf.nn.l2_loss(self.decode_weights)

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
        best_AUC, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, [], [], [], [], []
        for i in tqdm(range(epoch)):
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                _ = self.sess.run([self.optimizer], feed_dict=feed_dict)

            RQ, X, xBias = self.get_RQ(rating_matrix)
            Y = self.get_Y()
            yBias = self.get_yBias()
            prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=matrix_valid, bias=yBias, gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_X, best_xBias, best_Y, best_yBias = RQ, X, xBias, Y, yBias

        return best_RQ, best_X, best_xBias, best_Y, best_yBias

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self.batch_size)
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


def initfeatureembedae(matrix_train, matrix_valid, iteration=100, lam=0.01, rank=50, seed=0, batch_size=256, way='both',
                       dataset=None, gpu_on=True, **unused):
    progress = WorkSplitter()

    progress.section("InitFeatureEmbedAE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("InitFeatureEmbedAE: Load the variables trained on S_t")

    X = np.load('latent/' + dataset + 'unif_X_AutoRec_200.npy')
    xBias = np.load('latent/' + dataset + 'unif_xB_AutoRec_200.npy')
    Y = np.load('latent/' + dataset + 'unif_Y_AutoRec_200.npy')
    yBias = np.load('latent/' + dataset + 'unif_yB_AutoRec_200.npy')

    progress.section("InitFeatureEmbedAE: Training")
    m, n = matrix_train.shape
    model = InitFeatureEmbedAE(n, rank, lamb=lam, batch_size=batch_size, gpu_on=gpu_on, init_X=X, init_Y=Y,
                               init_xBias=xBias, init_yBias=yBias, way=way)
    metric_names = ['NLL', 'AUC']
    RQ, X, xBias, Y, yBias = model.train_model(matrix_train, matrix_valid, iteration, metric_names)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias