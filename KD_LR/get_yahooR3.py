import argparse
import scipy.sparse as sparse
import pandas as pd
import numpy as np

from utils.io import save_numpy
from utils.arg_check import ratio
from utils.split import seed_randomly_split
from utils.progress import WorkSplitter


def main(args):
    progress = WorkSplitter()

    progress.section("Yahoo R3: Load Raw Data")
    user_df = pd.read_csv(args.path + args.dataset + args.user, sep=args.sep, header=None, names=args.names)
    random_df = pd.read_csv(args.path + args.dataset + args.random, sep=args.sep, header=None, names=args.names)

    if args.implicit:
        """
        If only implicit (clicks, views, binary) feedback, convert to implicit feedback
        """
        user_df['rating'].loc[user_df['rating'] < args.threshold] = -1
        user_df['rating'].loc[user_df['rating'] >= args.threshold] = 1

        random_df['rating'].loc[random_df['rating'] < args.threshold] = -1
        random_df['rating'].loc[random_df['rating'] >= args.threshold] = 1

    progress.section("Yahoo R3: Randomly Split Random Set")
    m, n = max(user_df['uid']) + 1, max(user_df['iid']) + 1
    unif_train, validation, test = seed_randomly_split(df=random_df, ratio=args.ratio,
                                                       split_seed=args.seed, shape=(m, n))

    progress.section("Yahoo R3: Save NPZ")
    save_dir = args.path + args.dataset
    train = sparse.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
    save_numpy(train, save_dir, "S_c")
    save_numpy(unif_train, save_dir, "S_t")
    save_numpy(validation, save_dir, "S_va")
    save_numpy(test, save_dir, "S_te")

    progress.section("Yahoo R3: Statistics of Data Sets")
    print('* S_c  #num: %6d, pos: %.6f, neg: %.6f' % (train.count_nonzero(),
                                                      np.sum(train == 1) / train.count_nonzero(),
                                                      1 - np.sum(train == 1) / train.count_nonzero()))
    print('* S_t  #num: %6d, pos: %.6f, neg: %.6f' % (unif_train.count_nonzero(),
                                                      np.sum(unif_train == 1) / unif_train.count_nonzero(),
                                                      1 - np.sum(unif_train == 1) / unif_train.count_nonzero()))
    print('* S_va #num: %6d, pos: %.6f, neg: %.6f' % (validation.count_nonzero(),
                                                      np.sum(validation == 1) / validation.count_nonzero(),
                                                      1 - np.sum(validation == 1) / validation.count_nonzero()))
    print('* S_te #num: %6d, pos: %.6f, neg: %.6f' % (test.count_nonzero(),
                                                      np.sum(test == 1) / test.count_nonzero(),
                                                      1 - np.sum(test == 1) / test.count_nonzero()))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datasets/')
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-user', dest='user', default='user.txt')
    parser.add_argument('-random', dest='random', default='random.txt')
    parser.add_argument('-sep', dest='sep', help='separate', default=',')
    parser.add_argument('-n', dest='names', help='column names of dataframe',
                        default=['uid', 'iid', 'rating'])
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-r', dest='ratio', type=ratio, default='0.05,0.05,0.9')
    parser.add_argument('-threshold', dest='threshold', default=4)
    parser.add_argument('--implicit', dest='implicit', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
