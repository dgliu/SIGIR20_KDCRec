import numpy as np
import pandas as pd
from evaluation.metrics import evaluate
from models.predictor import predict


def execute(test, params, folder='latent'):
    df = pd.DataFrame(columns=['model', 'way'])

    if params['model'] in ['DeepAutoRec', 'HintAE', 'SoftLabelAE']:
        if params['way'] is not None:
            RQ = np.load('{3}/{2}_U_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
            Y = np.load('{3}/{2}_K_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
            Bias = np.load('{3}/{2}_kB_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))

        else:
            RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            Y = np.load('{2}/K_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            Bias = np.load('{2}/kB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
    else:
        if params['way'] is not None:
            RQ = np.load('{3}/{2}_U_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
            Y = np.load('{3}/{2}_Y_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))
            Bias = np.load('{3}/{2}_yB_{0}_{1}.npy'.format(params['model'], params['rank'], params['way'], folder))

        else:
            RQ = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            Y = np.load('{2}/Y_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            Bias = np.load('{2}/yB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))

    prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=test, bias=Bias)

    result = evaluate(prediction, test, params['metric'])

    result_dict = {'model': params['model'], 'way': params['way']}

    for name in result.keys():
        result_dict[name] = round(result[name][0], 8)

    df = df.append(result_dict, ignore_index=True)
    return df
