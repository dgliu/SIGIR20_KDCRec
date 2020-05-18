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
        if algorithm in ['BiasedMF', 'PropensityMF']:
            df = pd.DataFrame(columns=['model', 'batch_size', 'lambda', 'iter'])
            for batch_size in params['batch_size']:
                for lam in params['lambda']:
                    format = "model: {0}, batch_size: {1}, lambda: {2}"
                    progress.section(format.format(algorithm, batch_size, lam))
                    RQ, Y, uBias, iBias = params['models'][algorithm](train, validation,
                                                                      matrix_unif_train=unif_train,
                                                                      iteration=params['iter'],
                                                                      rank=params['rank'], gpu_on=gpu_on,
                                                                      lam=lam, seed=seed,
                                                                      batch_size=batch_size,
                                                                      way=way,
                                                                      dataset=dataset)

                    progress.subsection("Prediction")
                    prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=validation, ubias=uBias, ibias=iBias,
                                         gpu=gpu_on)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                    result_dict = {'model': algorithm, 'batch_size': batch_size, 'lambda': lam, 'iter': params['iter']}
                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 8)
                    df = df.append(result_dict, ignore_index=True)
                    save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['InitFeatureEmbedMF', 'AlterFeatureEmbedMF', 'WRSampleMF']:
            df = pd.DataFrame(columns=['model', 'lambda', 'iter'])
            for lam in params['lambda']:
                format = "model: {0}, lambda: {1}"
                progress.section(format.format(algorithm, lam))
                RQ, Y, uBias, iBias = params['models'][algorithm](train, validation,
                                                                  matrix_unif_train=unif_train,
                                                                  iteration=params['iter'],
                                                                  rank=params['rank'],
                                                                  gpu_on=gpu_on,
                                                                  lam=lam, seed=seed,
                                                                  batch_size=params['batch_size'],
                                                                  way=way,
                                                                  dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=validation, ubias=uBias, ibias=iBias,
                                     gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'lambda': lam, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['CausalSampleMF', 'BridgeLabelMF']:
            df = pd.DataFrame(columns=['model', 'lambda', 'lambda2', 'iter'])
            for lam in params['lambda']:
                for lam2 in params['lambda2']:
                    format = "model: {0}, lambda: {1}, lambda2: {2}"
                    progress.section(format.format(algorithm, lam, lam2))
                    RQ, Y, uBias, iBias = params['models'][algorithm](train, validation,
                                                                      matrix_unif_train=unif_train,
                                                                      iteration=params['iter'],
                                                                      rank=params['rank'],
                                                                      gpu_on=gpu_on,
                                                                      lam=lam, lam2=lam2,
                                                                      seed=seed,
                                                                      batch_size=params['batch_size'],
                                                                      way=way,
                                                                      dataset=dataset)

                    progress.subsection("Prediction")
                    prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=validation, ubias=uBias, ibias=iBias,
                                         gpu=gpu_on)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                    result_dict = {'model': algorithm, 'lambda': lam, 'lambda2': lam2, 'iter': params['iter']}
                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 8)
                    df = df.append(result_dict, ignore_index=True)
                    save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['UnionSampleMF', 'RefineLabelMF']:
            df = pd.DataFrame(columns=['model', 'confidence', 'iter'])
            for conf in params['confidence']:
                format = "model: {0}, confidence: {1}"
                progress.section(format.format(algorithm, conf))
                RQ, Y, uBias, iBias = params['models'][algorithm](train, validation,
                                                                  matrix_unif_train=unif_train,
                                                                  iteration=params['iter'],
                                                                  rank=params['rank'],
                                                                  gpu_on=gpu_on,
                                                                  lam=params['lambda'], seed=seed,
                                                                  batch_size=params['batch_size'],
                                                                  way=way,
                                                                  confidence=conf,
                                                                  dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=validation, ubias=uBias, ibias=iBias,
                                     gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'confidence': conf, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)
        elif algorithm in ['BatchSampleMF']:
            df = pd.DataFrame(columns=['model', 'step', 'iter'])
            for step in params['step']:
                format = "model: {0}, step: {1}"
                progress.section(format.format(algorithm, step))
                RQ, Y, uBias, iBias = params['models'][algorithm](train, validation,
                                                                  matrix_unif_train=unif_train,
                                                                  iteration=params['iter'],
                                                                  rank=params['rank'],
                                                                  gpu_on=gpu_on,
                                                                  lam=params['lambda'], seed=seed,
                                                                  batch_size=params['batch_size'],
                                                                  way=way,
                                                                  step=step,
                                                                  dataset=dataset)

                progress.subsection("Prediction")
                prediction = predict(matrix_U=RQ, matrix_V=Y, matrix_Valid=validation, ubias=uBias, ibias=iBias,
                                     gpu=gpu_on)

                progress.subsection("Evaluation")
                result = evaluate(prediction, validation, params['metric'], gpu=gpu_on)
                result_dict = {'model': algorithm, 'step': step, 'iter': params['iter']}
                for name in result.keys():
                    result_dict[name] = round(result[name][0], 8)
                df = df.append(result_dict, ignore_index=True)
                save_dataframe_csv(df, table_path, save_path)
