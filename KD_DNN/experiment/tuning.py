import pandas as pd
import os
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from models.predictor import predict
from utils.io import save_dataframe_csv


def hyper_parameter_tuning(train, validation, params, unif_train, save_path, seed, way, dataset, gpu_on):
    progress = WorkSplitter()

    table_path = 'tables/'
    data_name = save_path.split('/')[0]
    save_dir = 'tables/' + data_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for algorithm in params['models']:
        if algorithm in ['AutoRec']:
            df = pd.DataFrame(columns=['model', 'rank', 'batch_size', 'lambda', 'iter'])
            for rank in params['rank']:
                for batch_size in params['batch_size']:
                    for lam in params['lambda']:
                        format = "model: {0}, rank: {1}, batch_size: {2}, lambda: {3}"
                        progress.section(format.format(algorithm, rank, batch_size, lam))
                        RQ, X, xBias, Y, yBias = params['models'][algorithm](train, validation,
                                                                             matrix_unif_train=unif_train,
                                                                             iteration=params['iter'],
                                                                             rank=rank, gpu_on=gpu_on,
                                                                             lam=lam, seed=seed,
                                                                             batch_size=batch_size,
                                                                             way=way,
                                                                             dataset=dataset)

                        progress.subsection("Prediction")
                        prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                        progress.subsection("Evaluation")
                        result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                        result_dict = {'model': algorithm, 'rank': rank, 'batch_size': batch_size, 'lambda': lam,
                                       'iter': params['iter']}
                        for name in result.keys():
                            result_dict[name] = round(result[name][0], 8)
                        df = df.append(result_dict, ignore_index=True)
                        save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['InitFeatureEmbedAE', 'ConcatFeatureEmbedAE']:
            df = pd.DataFrame(columns=['model', 'batch_size', 'lambda', 'iter'])
            for batch_size in params['batch_size']:
                for lam in params['lambda']:
                    format = "model: {0}, batch_size: {1}, lambda: {2}"
                    progress.section(format.format(algorithm, batch_size, lam))
                    RQ, X, xBias, Y, yBias = params['models'][algorithm](train, validation,
                                                                         matrix_unif_train=unif_train,
                                                                         iteration=params['iter'],
                                                                         rank=params['rank'], gpu_on=gpu_on,
                                                                         lam=lam, seed=seed,
                                                                         batch_size=batch_size,
                                                                         way=way,
                                                                         dataset=dataset)

                    progress.subsection("Prediction")
                    prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                    result_dict = {'model': algorithm, 'batch_size': batch_size, 'lambda': lam,
                                   'iter': params['iter']}
                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 8)
                    df = df.append(result_dict, ignore_index=True)
                    save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['UnionSampleAE', 'RefineLabelAE']:
            df = pd.DataFrame(columns=['model', 'confidence', 'iter'])
            for conf in params['confidence']:
                format = "model: {0}, confidence: {1}"
                progress.section(format.format(algorithm, conf))
                RQ, X, xBias, Y, yBias = params['models'][algorithm](train, validation,
                                                                     matrix_unif_train=unif_train,
                                                                     iteration=params['iter'],
                                                                     rank=params['rank'], gpu_on=gpu_on,
                                                                     lam=params['lambda'], seed=seed,
                                                                     batch_size=params['batch_size'],
                                                                     way=way,
                                                                     confidence=conf,
                                                                     dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'confidence': conf, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['BatchSampleAE']:
            df = pd.DataFrame(columns=['model', 'step', 'iter'])
            for step in params['step']:
                format = "model: {0}, step: {1}"
                progress.section(format.format(algorithm, step))
                RQ, X, xBias, Y, yBias = params['models'][algorithm](train, validation,
                                                                     matrix_unif_train=unif_train,
                                                                     iteration=params['iter'],
                                                                     rank=params['rank'], gpu_on=gpu_on,
                                                                     lam=params['lambda'], seed=seed,
                                                                     batch_size=params['batch_size'],
                                                                     way=way,
                                                                     step=step,
                                                                     dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'step': step, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['BridgeLabelAE']:
            df = pd.DataFrame(columns=['model', 'lambda', 'lambda2', 'iter'])
            for lam in params['lambda']:
                for lam2 in params['lambda2']:
                    format = "model: {0}, lambda: {1}, lambda2: {2}"
                    progress.section(format.format(algorithm, lam, lam2))
                    RQ, X, xBias, Y, yBias = params['models'][algorithm](train, validation,
                                                                         matrix_unif_train=unif_train,
                                                                         iteration=params['iter'],
                                                                         rank=params['rank'], gpu_on=gpu_on,
                                                                         lam=lam, lam2=lam2,
                                                                         seed=seed,
                                                                         batch_size=params['batch_size'],
                                                                         way=way,
                                                                         dataset=dataset)

                    progress.subsection("Prediction")
                    prediction = predict(matrix_U=RQ, matrix_V=Y.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                    result_dict = {'model': algorithm, 'lambda': lam, 'lambda2': lam2, 'iter': params['iter']}
                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 8)
                    df = df.append(result_dict, ignore_index=True)
                    save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['SoftLabelAE']:
            df = pd.DataFrame(columns=['model', 'confidence', 'tau', 'iter'])
            for conf in params['confidence']:
                for tau in params['tau']:
                    format = "model: {0}, confidence: {1}, tau: {2}"
                    progress.section(format.format(algorithm, conf, tau))
                    RQ, X, xBias, Y, yBias, Z, zBias, K, kBias = params['models'][algorithm](train, validation,
                                                                                             matrix_unif_train=unif_train,
                                                                                             iteration=params['iter'],
                                                                                             rank=params['rank'],
                                                                                             rank2=params['rank2'],
                                                                                             gpu_on=gpu_on,
                                                                                             lam=params['lambda'],
                                                                                             seed=seed,
                                                                                             batch_size=params['batch_size'],
                                                                                             confidence=conf,
                                                                                             tau=tau,
                                                                                             dataset=dataset)

                    progress.subsection("Prediction")
                    prediction = predict(matrix_U=RQ, matrix_V=K.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                    result_dict = {'model': algorithm, 'confidence': conf, 'tau': tau, 'iter': params['iter']}
                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 8)
                    df = df.append(result_dict, ignore_index=True)
                    save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['HintAE']:
            df = pd.DataFrame(columns=['model', 'confidence', 'iter'])
            for conf in params['confidence']:
                format = "model: {0}, confidence: {1}"
                progress.section(format.format(algorithm, conf))
                RQ, X, xBias, Y, yBias, Z, zBias, K, kBias = params['models'][algorithm](train, validation,
                                                                                         matrix_unif_train=unif_train,
                                                                                         iteration=params['iter'],
                                                                                         rank=params['rank'],
                                                                                         rank2=params['rank2'],
                                                                                         gpu_on=gpu_on,
                                                                                         lam=params['lambda'],
                                                                                         seed=seed,
                                                                                         batch_size=params['batch_size'],
                                                                                         confidence=conf,
                                                                                         dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=K.T, matrix_Valid=validation, bias=yBias, gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'confidence': conf, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)