import numpy as np
import pandas as pd
from evaluation.metrics import evaluate
from models.predictor import predict


def execute(test, params, folder='latent'):
    RQ, Y, uBias, iBias = None, None, None, None
    df = pd.DataFrame(columns=['model', 'way'])

    if params['way'] is not None:
        RQ = np.load('{3}/{2}_U_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
        Y = np.load('{3}/{2}_V_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
        uBias = np.load('{3}/{2}_uB_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
        iBias = np.load('{3}/{2}_iB_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))

    else:
        RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        Y = np.load('{2}/V_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        uBias = np.load('{2}/uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
        iBias = np.load('{2}/iB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))

    prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=test, ubias=uBias, ibias=iBias)

    result = evaluate(prediction, test, params['metric'])

    result_dict = {'model': params['model'], 'way': params['way']}

    for name in result.keys():
        result_dict[name] = round(result[name][0], 8)

    df = df.append(result_dict, ignore_index=True)
    return df
