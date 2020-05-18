import argparse
from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, load_yaml
from utils.arg_check import check_int_positive
from utils.model_names import models
from utils.progress import WorkSplitter


def main(args):
    progress = WorkSplitter()
    progress.section("Tune Parameters")
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}
    train = load_numpy(path=args.path, name=args.dataset + args.train)
    unif_train = load_numpy(path=args.path, name=args.dataset + args.unif_train)
    valid = load_numpy(path=args.path, name=args.dataset + args.valid)
    hyper_parameter_tuning(train, valid, params, unif_train=unif_train, save_path=args.dataset + args.name,
                           gpu_on=args.gpu, seed=args.seed, way=args.way, dataset=args.dataset)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    # Optional:
    # For AutoRec               | "ae_tuning.csv", "unif_ae_tuning.csv', "combine_ae_tuning.csv'
    # For InitFeatureEmbedAE    | "user_initfeatureembedae_tuning.csv", "item_initfeatureembedae_tuning.csv",
    #                           | "both_initfeatureembedae_tuning.csv"
    # For AlterFeatureEmbedAE   | "alterfeatureembedae_tuning.csv"
    # For ConcatFeatureEmbedAE  | "concatfeatureembedae_tuning.csv"
    # For UnionSampleAE         | "unionsampleae_tuning.csv"
    # For BatchSampleAE         | "batchsampleae_tuning.csv"
    # For BridgeLabelAE         | "bridgelabelae_tuning.csv"
    # For RefineLabelAE         | "refinelabelae_tuning.csv"
    # For SoftLabelAE           | "softlabelae_tuning.csv"
    # For HintAE                | "hintae_tuning.csv"
    parser.add_argument('-n', dest='name', default="softlabelae_tuning.csv")
    parser.add_argument('-p', dest='path', default="datasets/")
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    # Optional:
    # For AutoRec               | None, "unif", "combine"
    # For InitFeatureEmbedAE    | "user", "item", "both"
    # For AlterFeatureEmbedAE   | None
    # For ConcatFeatureEmbedAE  | None
    # For UnionSampleMF         | None
    # For BatchSampleMF         | None
    # For BridgeLabelMF         | None
    # For RefineLabelMF         | None
    # For SoftLabelAE           | None
    # For HintAE                | None
    parser.add_argument('-w', dest='way', default=None)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    # Optional:
    # For AutoRec               | "ae.yml", "unifae.yml"(when way='unif')
    # For InitFeatureEmbedAE    | "initfeatureembedae.yml",
    # For AlterFeatureEmbedAE   | "alterfeatureembedae.yml"
    # For ConcatFeatureEmbedAE  | "concatfeatureembedae.yml"
    # For UnionSampleAE         | "unionsampleae.yml"
    # For BatchSampleAE         | "batchsampleae.yml"
    # For BridgeLabelAE         | "bridgelabelae.yml"
    # For RefineLabelAE         | "refinelabelae.yml"
    # For SoftLabelAE           | "softlabelae.yml'
    # For HintAE                | "hintae.yml'
    parser.add_argument('-g', dest='grid', default='config/softlabelae.yml')
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
