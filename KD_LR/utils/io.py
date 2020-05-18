from scipy.sparse import save_npz, load_npz
import pandas as pd
import yaml
import numpy as np


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def convert_npz_to_csv(params, folder, name):
    if params['special'] is not None:
        RQ = np.load('{3}/{2}_U_{0}_{1}.npy'.format(params['model'], params['rank'], params['special'], folder))
        Y = np.load('{3}/{2}_V_{0}_{1}.npy'.format(params['model'], params['rank'], params['special'], folder))
        uBias = np.load('{3}/{2}_uB_{0}_{1}.npy'.format(params['model'], params['rank'], params['special'], folder))
        iBias = np.load('{3}/{2}_iB_{0}_{1}.npy'.format(params['model'], params['rank'], params['special'], folder))

        np.savetxt('{2}/{1}_U_{0}.txt'.format(params['model'], params['special'], name), RQ)
        np.savetxt('{2}/{1}_V_{0}.txt'.format(params['model'], params['special'], name), Y)
        np.savetxt('{2}/{1}_uB_{0}.txt'.format(params['model'], params['special'], name), uBias)
        np.savetxt('{2}/{1}_iB_{0}.txt'.format(params['model'], params['special'], name), iBias)
    else:
        RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        uBias = np.load('{2}/uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        iBias = np.load('{2}/iB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))

        np.savetxt('{1}/U_{0}.txt'.format(params['model'], name), RQ)
        np.savetxt('{1}/V_{0}.txt'.format(params['model'], name), Y)
        np.savetxt('{1}/uB_{0}.txt'.format(params['model'], name), uBias)
        np.savetxt('{1}/iB_{0}.txt'.format(params['model'], name), iBias)

