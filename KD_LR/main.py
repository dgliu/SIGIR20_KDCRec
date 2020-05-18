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
    RQ, Y, uBias, iBias = models[args.model](train, valid, dataset=args.dataset, matrix_unif_train=unif_train,
                                             iteration=args.iter, rank=args.rank, gpu_on=args.gpu, lam=args.lamb,
                                             lam2=args.lamb2, seed=args.seed, batch_size=args.batch_size, way=args.way,
                                             confidence=args.confidence, step=args.step)

    save_path = 'latent/' + args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.way is None:
        np.save(save_path + '/U_{0}_{1}'.format(args.model, args.rank), RQ)
        np.save(save_path + '/V_{0}_{1}'.format(args.model, args.rank), Y)
        if uBias is not None:
            np.save(save_path + '/uB_{0}_{1}'.format(args.model, args.rank), uBias)
            np.save(save_path + '/iB_{0}_{1}'.format(args.model, args.rank), iBias)
    else:
        np.save(save_path + '/' + args.way + '_U_{0}_{1}'.format(args.model, args.rank), RQ)
        np.save(save_path + '/' + args.way + '_V_{0}_{1}'.format(args.model, args.rank), Y)
        if uBias is not None:
            np.save(save_path + '/' + args.way + '_uB_{0}_{1}'.format(args.model, args.rank), uBias)
            np.save(save_path + '/' + args.way + '_iB_{0}_{1}'.format(args.model, args.rank), iBias)

    progress.section("Predict")
    prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=valid, ubias=uBias, ibias=iBias, gpu=args.gpu)

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
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=0.1)
    parser.add_argument('-l2', dest='lamb2', type=check_float_positive, default=0.1)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=10)
    parser.add_argument('-b', dest='batch_size', type=check_int_positive, default=128)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-c', dest='confidence', type=check_float_positive, default=1.0)
    parser.add_argument('-st', dest='step', type=check_int_positive, default=20)
    # Optional:
    # "BiasedMF",
    # "PropensityMF",
    # "InitFeatureEmbedMF",
    # "AlterFeatureEmbedMF",
    # "ConcatFeatureEmbedMF",
    # "CausalSampleMF",
    # "UnionSampleMF",
    # "WRSampleMF",
    # "BatchSampleMF", (note: For gain analysis, we use variant RestrictedBatchSampleMF)
    # "BridgeLabelMF",
    # "RefineLabelMF"
    parser.add_argument('-m', dest='model', default='WRSampleMF')
    # Optional:
    # For BiasedMF              | None, "unif", "combine"
    # For PropensityMF          | None
    # For InitFeatureEmbedMF    | "user", "item", "both"
    # For AlterFeatureEmbedMF   | None
    # For ConcatFeatureEmbedMF  | None
    # For CausalSampleMF        | None
    # For UnionSampleMF         | None
    # For WRSampleMF            | None
    # For BatchSampleMF         | None
    # (note: For RestrictedBatchSampleMF, including None,"head_users","tail_users","head_items","tail_items")
    # For BridgeLabelMF         | None
    # For RefineLabelMF         | None
    parser.add_argument('-w', dest='way', default=None)
    parser.add_argument('-p', dest='path', default='datasets/')
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
