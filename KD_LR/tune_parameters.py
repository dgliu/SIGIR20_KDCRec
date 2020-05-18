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
    # For BiasedMF              | "biasedmf_tuning.csv", "unif_biasedmf_tuning.csv', "combine_biasedmf_tuning.csv'
    # For PropensityMF          | "propensitymf_tuning.csv",
    # For InitFeatureEmbedMF    | "user_initfeatureembedmf_tuning.csv", "item_initfeatureembedmf_tuning.csv",
    #                           | "both_initfeatureembedmf_tuning.csv",
    # For AlterFeatureEmbedMF   | "alterfeatureembedmf_tuning.csv"
    # For ConcatFeatureEmbedMF  | "concatfeatureembedmf_tuning.csv"
    # For CausalSampleMF        | "causalsamplemf_tuning.csv"
    # For UnionSampleMF         | "unionsamplemf_tuning.csv"
    # For WRSampleMF            | "wrsamplemf_tuning.csv"
    # For BatchSampleMF         | "batchsamplemf_tuning.csv"
    # For BridgeLabelMF         | "bridgelabelmf_tuning.csv"
    # For RefineLabelMF         | "refinelabelmf_tuning.csv"
    parser.add_argument('-n', dest='name', default="refinelabelmf_tuning.csv")
    parser.add_argument('-p', dest='path', default="datasets/")
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
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
    # For BridgeLabelMF         | None
    # For RefineLabelMF         | None
    parser.add_argument('-w', dest='way', default=None)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    # Optional:
    # For BiasedMF              | "biasedmf.yml", "unifbiasedmf.yml"(when way='unif')
    # For PropensityMF          | "propensitymf.yml",
    # For InitFeatureEmbedMF    | "initfeatureembedmf.yml",
    # For AlterFeatureEmbedMF   | "alterfeatureembedmf.yml"
    # For ConcatFeatureEmbedMF  | "concatfeatureembedmf.yml"
    # For CausalSampleMF        | "causalsamplemf.yml"
    # For UnionSampleMF         | "unionsamplemf.yml"
    # For WRSampleMF            | "wrsamplemf.yml"
    # For BatchSampleMF         | "batchsamplemf.yml"
    # For BridgeLabelMF         | "bridgelabelmf.yml"
    # For RefineLabelMF         | "refinelabelmf.yml"
    parser.add_argument('-g', dest='grid', default='config/refinelabelmf.yml')
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
