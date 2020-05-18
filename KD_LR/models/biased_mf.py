import tensorflow as tf
import numpy as np
import cupy as cp
from tqdm import tqdm
from utils.progress import WorkSplitter
from scipy.sparse import lil_matrix
from models.predictor import predict
from evaluation.metrics import evaluate


class BiasedMF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 lamb,
                 batch_size,
                 gpu_on,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lamb = lamb
        self.batch_size = batch_size
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
        # Placehoder
        self.user_idx = tf.placeholder(tf.int32, [None])
        self.item_idx = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.float32, [None])

        # Variable to learn
        self.user_embeddings = tf.get_variable(name='users', shape=[self.num_users, self.embed_dim],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.item_embeddings = tf.get_variable(name='items', shape=[self.num_items, self.embed_dim],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.user_bias_embeddings = tf.get_variable(name='users_bias', shape=[self.num_users, ],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.item_bias_embeddings = tf.get_variable(name='items_bias', shape=[self.num_items, ],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

        with tf.variable_scope("mf_loss"):
            users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx)
            users_bias = tf.nn.embedding_lookup(self.user_bias_embeddings, self.user_idx)
            items = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx)
            items_bias = tf.nn.embedding_lookup(self.item_bias_embeddings, self.item_idx)

            x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + users_bias + items_bias

            mf_loss = tf.reduce_mean(tf.square(self.label - x_ij))

        with tf.variable_scope('l2_loss'):
            unique_user_idx, _ = tf.unique(self.user_idx)
            unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)
            # unique_ubias = tf.nn.embedding_lookup(self.user_bias_embeddings, unique_user_idx)

            unique_item_idx, _ = tf.unique(self.item_idx)
            unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)
            # unique_ibias = tf.nn.embedding_lookup(self.item_bias_embeddings, unique_item_idx)

            l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))
            # l2_loss_bias = tf.reduce_mean(tf.nn.l2_loss(unique_ubias)) + tf.reduce_mean(tf.nn.l2_loss(unique_ibias))

        with tf.variable_scope('loss'):
            self.loss = mf_loss + self.lamb * l2_loss

        with tf.variable_scope('optimizer'):
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.loss)
            self.optimizer = tf.contrib.opt.LazyAdamOptimizer().minimize(self.loss)

    @staticmethod
    def get_batches(user_item_pairs, rating_matrix, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(user_item_pairs) / batch_size)):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], label])

        return batches

    def train_model(self, rating_matrix, matrix_valid, epoch, metric_names):
        user_item_matrix = lil_matrix(rating_matrix)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        best_AUC, best_RQ, best_Y, best_uBias, best_iBias, best_prediction = 0, [], [], [], [], []
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, rating_matrix, self.batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2]
                             }
                _ = self.sess.run([self.optimizer], feed_dict=feed_dict)

            RQ, Y, uBias, iBias = self.sess.run(
                [self.user_embeddings, self.item_embeddings,
                 self.user_bias_embeddings, self.item_bias_embeddings])
            prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=matrix_valid, ubias=uBias, ibias=iBias, gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_Y, best_uBias, best_iBias, best_prediction = RQ, Y, uBias, iBias, prediction

        return best_RQ, best_Y, best_uBias, best_iBias, best_prediction


def biasedmf(matrix_train, matrix_valid, matrix_unif_train, iteration=100, lam=0.01, rank=50, seed=0, batch_size=500,
             way=None, gpu_on=True, **unused):
    progress = WorkSplitter()

    progress.section("BiasedMF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("BiasedMF: Training")
    m, n = matrix_train.shape
    model = BiasedMF(m, n, rank, lamb=lam, batch_size=batch_size, gpu_on=gpu_on)
    metric_names = ['NLL', 'AUC']

    if way == 'unif':
        RQ, Y, user_bias, item_bias, _ = model.train_model(matrix_unif_train, matrix_valid, iteration, metric_names)
    elif way == 'combine':
        matrix_train += matrix_unif_train
        RQ, Y, user_bias, item_bias, _ = model.train_model(matrix_train, matrix_valid, iteration, metric_names)
    else:
        RQ, Y, user_bias, item_bias, prediction = model.train_model(matrix_train, matrix_valid, iteration, metric_names)

    # if gpu_on:
    #     np.savetxt('Matlab/biasedmf_prediction.txt', cp.asnumpy(prediction))
    # else:
    #     np.savetxt('Matlab/biasedmf_prediction.txt', prediction)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, user_bias, item_bias
