import os
import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy
from utils.arg_check import check_float_positive, check_int_positive
from utils.model_names import models
from models.predictor import predict
from evaluation.metrics import evaluate


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.dataset + args.train))
    print("Uniform Train File Name: {0}".format(args.dataset + args.unif_train))
    print("Valid File Name: {0}".format(args.dataset + args.valid))
    print("Algorithm: {0}".format(args.model))
    print("Way: {0}".format(args.way))
    print("Seed: {0}".format(args.seed))
    print("Batch Size: {0}".format(args.batch_size))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("Iteration: {0}".format(args.iter))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()

    train = load_numpy(path=args.path, name=args.dataset + args.train)

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(train.shape))

    # Train Model
    valid = load_numpy(path=args.path, name=args.dataset + args.valid)

    unif_train = load_numpy(path=args.path, name=args.dataset + args.unif_train)

    if args.model in ['DeepAutoRec', 'HintAE', 'SoftLabelAE']:
        RQ, X, xBias, Y, yBias, Z, zBias, K, kBias = models[args.model](train, valid, dataset=args.dataset,
                                                                        matrix_unif_train=unif_train, iteration=args.iter,
                                                                        rank=args.rank, rank2=args.rank2, gpu_on=args.gpu,
                                                                        lam=args.lamb, seed=args.seed,
                                                                        batch_size=args.batch_size, way=args.way,
                                                                        confidence=args.confidence, step=args.step,
                                                                        tau=args.tau)

        save_path = 'latent/' + args.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.way is None:
            np.save(save_path + '/U_{0}_{1}'.format(args.model, args.rank), RQ)
            np.save(save_path + '/Y_{0}_{1}'.format(args.model, args.rank), Y)
            np.save(save_path + '/X_{0}_{1}'.format(args.model, args.rank), X)
            np.save(save_path + '/Z_{0}_{1}'.format(args.model, args.rank), Z)
            np.save(save_path + '/K_{0}_{1}'.format(args.model, args.rank), K)
            if xBias is not None:
                np.save(save_path + '/xB_{0}_{1}'.format(args.model, args.rank), xBias)
                np.save(save_path + '/yB_{0}_{1}'.format(args.model, args.rank), yBias)
                np.save(save_path + '/zB_{0}_{1}'.format(args.model, args.rank), zBias)
                np.save(save_path + '/kB_{0}_{1}'.format(args.model, args.rank), kBias)
        else:
            np.save(save_path + '/' + args.way + '_U_{0}_{1}'.format(args.model, args.rank), RQ)
            np.save(save_path + '/' + args.way + '_Y_{0}_{1}'.format(args.model, args.rank), Y)
            np.save(save_path + '/' + args.way + '_X_{0}_{1}'.format(args.model, args.rank), X)
            np.save(save_path + '/' + args.way + '_Z_{0}_{1}'.format(args.model, args.rank), Z)
            np.save(save_path + '/' + args.way + '_K_{0}_{1}'.format(args.model, args.rank), K)
            if xBias is not None:
                np.save(save_path + '/' + args.way + '_xB_{0}_{1}'.format(args.model, args.rank), xBias)
                np.save(save_path + '/' + args.way + '_yB_{0}_{1}'.format(args.model, args.rank), yBias)
                np.save(save_path + '/' + args.way + '_zB_{0}_{1}'.format(args.model, args.rank), zBias)
                np.save(save_path + '/' + args.way + '_kB_{0}_{1}'.format(args.model, args.rank), kBias)

        progress.section("Predict")
        prediction = predict(matrix_U=RQ, matrix_V=K.T, matrix_Valid=valid, bias=yBias, gpu=args.gpu)

        progress.section("Evaluation")
        start_time = time.time()
        metric_names = ['NLL', 'AUC']
        result = evaluate(prediction, valid, metric_names, gpu=args.gpu)

        print("----Final Result----")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))
    else:
        RQ, X, xBias, Y, yBias = models[args.model](train, valid, dataset=args.dataset, matrix_unif_train=unif_train,
                                                    iteration=args.iter, rank=args.rank, gpu_on=args.gpu,
                                                    lam=args.lamb, lam2=args.lamb2, seed=args.seed,
                                                    batch_size=args.batch_size, way=args.way, confidence=args.confidence,
                                                    step=args.step)

        save_path = 'latent/' + args.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.way is None:
            np.save(save_path + '/U_{0}_{1}'.format(args.model, args.rank), RQ)
            np.save(save_path + '/Y_{0}_{1}'.format(args.model, args.rank), Y)
            np.save(save_path + '/X_{0}_{1}'.format(args.model, args.rank), X)
            if xBias is not None:
                np.save(save_path + '/xB_{0}_{1}'.format(args.model, args.rank), xBias)
                np.save(save_path + '/yB_{0}_{1}'.format(args.model, args.rank), yBias)
        else:
            np.save(save_path + '/' + args.way + '_U_{0}_{1}'.format(args.model, args.rank), RQ)
            np.save(save_path + '/' + args.way + '_Y_{0}_{1}'.format(args.model, args.rank), Y)
            np.save(save_path + '/' + args.way + '_X_{0}_{1}'.format(args.model, args.rank), X)
            if xBias is not None:
                np.save(save_path + '/' + args.way + '_xB_{0}_{1}'.format(args.model, args.rank), xBias)
                np.save(save_path + '/' + args.way + '_yB_{0}_{1}'.format(args.model, args.rank), yBias)

        progress.section("Predict")
        prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=valid, bias=yBias, gpu=args.gpu)

        progress.section("Evaluation")
        start_time = time.time()
        metric_names = ['NLL', 'AUC']
        result = evaluate(prediction, valid, metric_names, gpu=args.gpu)

        print("----Final Result----")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="KD")

    parser.add_argument('-i', dest='iter', type=check_int_positive, default=100)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1e-5)
    parser.add_argument('-l2', dest='lamb2', type=check_float_positive, default=0.01)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=200)
    parser.add_argument('-r2', dest='rank2', type=check_int_positive, default=100)
    parser.add_argument('-b', dest='batch_size', type=check_int_positive, default=128)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-c', dest='confidence', type=check_float_positive, default=0.2)
    parser.add_argument('-tau', dest='tau', type=check_float_positive, default=5)
    parser.add_argument('-st', dest='step', type=check_int_positive, default=3)
    # Optional:
    # "AutoRec",
    # "InitFeatureEmbedAE",
    # "AlterFeatureEmbedAE",
    # "ConcatFeatureEmbedAE",
    # "UnionSampleAE",
    # "WRSampleAE",
    # "BatchSampleAE",
    # "BridgeLabelAE",
    # "RefineLabelAE"
    # "DeepAutoRec",
    # "HintAE",
    # "SoftLabelAE",
    parser.add_argument('-m', dest='model', default='SoftLabelAE')
    # Optional:
    # For AutoRec               | None, "unif", "combine"
    # For InitFeatureEmbedAE    | "user", "item", "both"
    # For AlterFeatureEmbedAE   | None
    # For ConcatFeatureEmbedAE  | None
    # For UnionSampleAE         | None
    # For WRSampleAE            | None
    # For BatchSampleAE         | None
    # For BridgeLabelAE         | None
    # For RefineLabelAE         | None
    # For DeepAutoRec           | None, "unif"
    # For HintAE                | None
    # For SoftLabelAE           | None
    parser.add_argument('-w', dest='way', default=None)
    parser.add_argument('-p', dest='path', default='datasets/')
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)