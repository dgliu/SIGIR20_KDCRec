import tensorflow as tf
import numpy as np
import cupy as cp
from tqdm import tqdm
from utils.progress import WorkSplitter
from scipy.sparse import lil_matrix
from models.predictor import predict
from evaluation.metrics import evaluate


class RefineLabelMF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 lamb,
                 confidence,
                 norm_init_U,
                 norm_init_V,
                 norm_init_uBias,
                 norm_init_iBias,
                 unif_init_U,
                 unif_init_V,
                 unif_init_uBias,
                 unif_init_iBias,
                 batch_size,
                 gpu_on,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.double_num_items = num_items * 2
        self.embed_dim = embed_dim
        self.lamb = lamb
        self.confidence = confidence
        self.norm_init_U = norm_init_U
        self.norm_init_V = norm_init_V
        self.norm_init_uBias = norm_init_uBias
        self.norm_init_iBias = norm_init_iBias
        self.unif_init_U = unif_init_U
        self.unif_init_V = unif_init_V
        self.unif_init_uBias = unif_init_uBias
        self.unif_init_iBias = unif_init_iBias
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

        # Import Pre-Trained Variables
        self.norm_user_embeddings = tf.get_variable('users', initializer=self.norm_init_U)
        self.norm_item_embeddings = tf.get_variable('items', initializer=self.norm_init_V)
        self.norm_user_bias_embeddings = tf.get_variable('users_bias', initializer=self.norm_init_uBias)
        self.norm_item_bias_embeddings = tf.get_variable('items_bias', initializer=self.norm_init_iBias)

        self.unif_user_embeddings = tf.get_variable('unif_users', initializer=self.unif_init_U, trainable=False)
        self.unif_item_embeddings = tf.get_variable('unif_items', initializer=self.unif_init_V, trainable=False)
        self.unif_user_bias_embeddings = tf.get_variable('unif_users_bias', initializer=self.unif_init_uBias, trainable=False)
        self.unif_item_bias_embeddings = tf.get_variable('unif_items_bias', initializer=self.unif_init_iBias, trainable=False)

        with tf.variable_scope("refine_label"):
            unif_users = tf.nn.embedding_lookup(self.unif_user_embeddings, self.user_idx)
            unif_users_bias = tf.nn.embedding_lookup(self.unif_user_bias_embeddings, self.user_idx)
            unif_items = tf.nn.embedding_lookup(self.unif_item_embeddings, self.item_idx)
            unif_item_bias = tf.nn.embedding_lookup(self.unif_item_bias_embeddings, self.item_idx)

            predict_label = tf.reduce_sum(
                tf.multiply(unif_users, unif_items), axis=1) + unif_users_bias + unif_item_bias
            self.predict_label = (predict_label - tf.reduce_min(predict_label)) / (
                    tf.reduce_max(predict_label) - tf.reduce_min(predict_label))

            self.refined_label = self.label + self.confidence * self.predict_label

        with tf.variable_scope("mf_loss"):
            users = tf.nn.embedding_lookup(self.norm_user_embeddings, self.user_idx)
            users_bias = tf.nn.embedding_lookup(self.norm_user_bias_embeddings, self.user_idx)
            items = tf.nn.embedding_lookup(self.norm_item_embeddings, self.item_idx)
            item_bias = tf.nn.embedding_lookup(self.norm_item_bias_embeddings, self.item_idx)

            x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + users_bias + item_bias
            mf_loss = tf.reduce_mean(tf.square(self.refined_label - x_ij))

        with tf.variable_scope('l2_loss'):
            unique_user_idx, _ = tf.unique(self.user_idx)
            unique_users = tf.nn.embedding_lookup(self.norm_user_embeddings, unique_user_idx)

            unique_item_idx, _ = tf.unique(self.item_idx)
            unique_items = tf.nn.embedding_lookup(self.norm_item_embeddings, unique_item_idx)

            l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

        with tf.variable_scope('loss'):
            self.loss = mf_loss + self.lamb * l2_loss

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.contrib.opt.LazyAdamOptimizer().minimize(self.loss)

    def get_batches(self, user_item_pairs, matrix_train, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(user_item_pairs) / batch_size)):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(matrix_train[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], label])

        return batches

    def train_model(self, matrix_train, matrix_valid, epoch, metric_names):
        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        best_AUC, best_RQ, best_Y, best_uBias, best_iBias, best_refined_label, best_prediction = 0, [], [], [], [], [], []
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, matrix_train, self.batch_size)
            refined_label = []
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2]
                             }
                _, temp_refined_label = self.sess.run([self.optimizer, self.refined_label], feed_dict=feed_dict)
                refined_label.append(np.stack((batches[step][0], batches[step][1], np.asarray(temp_refined_label)), axis=-1))

            RQ, Y, uBias, iBias = self.sess.run(
                [self.norm_user_embeddings, self.norm_item_embeddings,
                 self.norm_user_bias_embeddings, self.norm_item_bias_embeddings])
            prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=matrix_valid, ubias=uBias, ibias=iBias,
                                 gpu=self.gpu_on)
            result = evaluate(prediction, matrix_valid, metric_names, gpu=self.gpu_on)

            if result['AUC'][0] > best_AUC:
                best_AUC = result['AUC'][0]
                best_RQ, best_Y, best_uBias, best_iBias, best_refined_label, best_prediction = RQ, Y, uBias, iBias, np.vstack(refined_label), prediction

        return best_RQ, best_Y, best_uBias, best_iBias, best_refined_label, user_item_pairs, best_prediction


def refinelabelmf(matrix_train, matrix_valid, iteration=100, lam=0.01, confidence=0.9, rank=50,
                  seed=0, batch_size=500, gpu_on=True, dataset=None, **unused):
    progress = WorkSplitter()

    progress.section("RefineLabelMF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("RefineLabelMF: Load the variables trained on S_c/S_t")

    norm_RQ = np.load('latent/' + dataset + 'U_BiasedMF_10.npy')
    norm_Y = np.load('latent/' + dataset + 'V_BiasedMF_10.npy')
    norm_uBias = np.load('latent/' + dataset + 'uB_BiasedMF_10.npy')
    norm_iBias = np.load('latent/' + dataset + 'iB_BiasedMF_10.npy')

    unif_RQ = np.load('latent/' + dataset + 'unif_U_BiasedMF_10.npy')
    unif_Y = np.load('latent/' + dataset + 'unif_V_BiasedMF_10.npy')
    unif_uBias = np.load('latent/' + dataset + 'unif_uB_BiasedMF_10.npy')
    unif_iBias = np.load('latent/' + dataset + 'unif_iB_BiasedMF_10.npy')

    progress.section("RefineLabelMF: Training")
    m, n = matrix_train.shape
    model = RefineLabelMF(m, n, rank, lamb=lam, confidence=confidence, batch_size=batch_size, gpu_on=gpu_on,
                          norm_init_U=norm_RQ, norm_init_V=norm_Y, norm_init_uBias=norm_uBias, norm_init_iBias=norm_iBias,
                          unif_init_U=unif_RQ, unif_init_V=unif_Y, unif_init_uBias=unif_uBias, unif_init_iBias=unif_iBias)
    metric_names = ['NLL', 'AUC']
    RQ, Y, user_bias, item_bias, refined_label, user_item_pairs, prediction = model.train_model(matrix_train,
                                                                                                matrix_valid,
                                                                                                iteration,
                                                                                                metric_names)

    # if gpu_on:
    #     np.savetxt('Matlab/refinelabelmf_prediction.txt', cp.asnumpy(prediction))
    # else:
    #     np.savetxt('Matlab/refinelabelmf_prediction.txt', prediction)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, user_bias, item_bias