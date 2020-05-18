import argparse
import pandas as pd
from experiment.execute import execute
from utils.io import load_numpy, save_dataframe_csv
from utils.progress import WorkSplitter


def main(args):
    progress = WorkSplitter()

    table_path = 'tables/'

    test = load_numpy(path=args.path, name=args.dataset + args.test)

    df = pd.DataFrame({'model': ['RestrictedBatchSampleMF', 'RestrictedBatchSampleMF', 'RestrictedBatchSampleMF',
                                 'RestrictedBatchSampleMF', 'RestrictedBatchSampleMF'],
                       'way': [None, 'head_users', 'tail_users', 'head_items', 'tail_items']})

    progress.subsection("Gain Analysis")
    frame = []
    for idx, row in df.iterrows():
        row = row.to_dict()
        row['metric'] = ['NLL', 'AUC']
        row['rank'] = 10
        result = execute(test, row, folder=args.model_folder + args.dataset)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Gain Analysis")
    parser.add_argument('-n', dest='name', default="yahooR3/gain_result.csv")
    parser.add_argument('-p', dest='path', default="datasets/")
    parser.add_argument('-d', dest='dataset', default='yahooR3/')
    parser.add_argument('-e', dest='test', default='S_te.npz')
    parser.add_argument('-s', dest='model_folder', default='latent/')
    args = parser.parse_args()

    main(args)