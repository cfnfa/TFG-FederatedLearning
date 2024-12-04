import os
import tensorflow as tf
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import librerias.GraphModel as tf_model
import librerias.TF_Network as linear_model
import librerias.DataTransformation as dt
import librerias.ConfiguracionOptimizacionParametrosForServer as config
from utils import RMSE
import manage_dataset


def main(raw_evaluation=False, year=2018):
    data_dir = 'dataset/'
    models_root = 'weights/'
    results_root = 'results_{}/'.format(year)

    pathlib.Path(models_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_root).mkdir(parents=True, exist_ok=True)

    all_models = os.listdir(models_root)
    result_list = []

    # recogemos todas las muestras, ya divididas
    all_samples, training_samples, eval_samples, test_samples = dataset_samples.get_dataset(year=year)
    # cargamos y dividimos los dataset, juntando todos los datos en un mismo array
    train_dataset, eval_dataset, test_dataset = manage_dataset.create_dataset(training_samples, eval_samples,
                                                                              test_samples,
                                                                              data_dir, features=3)
    if (raw_evaluation):
        _, _, raw_y = manage_dataset.create_extrapolated_test_dataset(training_samples,
                                                                      eval_samples,
                                                                      test_samples,
                                                                      data_dir)
    # normalizamos los datos de train y validation
    test_dataset, mean = dt.data_normalization(test_dataset)  #### OJO: Especificar los máximos ####

    for model_name in all_models:
        parameters = model_name.split('_')
        t_seq = int(parameters[1])
        q = int(parameters[3])
        H = 1
        num_layers = int(parameters[5])
        layer1 = int(parameters[7])
        layer2 = int(parameters[9])
        lr = (parameters[11])
        dropout, recurrent_dropout = 0, 0

        model_result_root = results_root + model_name + '/'
        pathlib.Path(model_result_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(model_result_root + 'Graficas/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(model_result_root + 'Ficheros/').mkdir(parents=True, exist_ok=True)

        testX, testY = manage_dataset.test_dataset(t_seq, q, H, test_dataset)  # No mezclamos las muestras
        if (raw_evaluation):
            testX_raw, testY_raw = manage_dataset.test_dataset(t_seq, q, H, raw_y)  # No mezclamos las muestras

        weights_dir = models_root + model_name + '/model.tf'
        if num_layers > 1:
            LSTM_model = linear_model.TF_LSTM(input_shape=(testX.shape[1], testX.shape[2]),
                                              hidden_units=layer1, hidden_units_2=0, hidden_units_3=layer2,
                                              layers=num_layers, dropout=dropout,
                                              recurrent_dropout=recurrent_dropout)
        else:
            LSTM_model = linear_model.TF_LSTM(input_shape=(testX.shape[1], testX.shape[2]),
                                              hidden_units=layer1, hidden_units_2=layer2, hidden_units_3=0,
                                              layers=num_layers, dropout=dropout,
                                              recurrent_dropout=recurrent_dropout)
        model = LSTM_model.build()
        model.compile(loss=RMSE,
                      optimizer='adam',
                      metrics=[tf.metrics.RootMeanSquaredError(), tf.metrics.MeanAbsoluteError()])
        model.load_weights(weights_dir).expect_partial()
        model.summary()

        y_hat = model.predict(testX)
        y_hat = dt.data_denormalisation(y_hat, mean)
        y_hat = y_hat.flatten()
        # y_hat = (y_hat * 400) + mean
        y = dt.data_denormalisation(testY, mean)
        # y = np.array([(p * 400) + mean for p in testY])
        if raw_evaluation:
            y = testY_raw
        y = y.flatten()

        pearson_correlation = pearsonr(y_hat, y)
        rmse = mean_squared_error(y, y_hat, squared=False)
        mse = mean_squared_error(y, y_hat, squared=True)
        mae = mean_absolute_error(y, y_hat)
        num_samples = test_dataset.shape[1] - t_seq - q - H
        for i in range(len(test_samples)):
            plt.plot(y[i * num_samples:i * num_samples + num_samples], label='Original')
            plt.plot(y_hat[i * num_samples:i * num_samples + num_samples], '-r', label='Prediction')
            plt.legend()
            dir_graf = model_result_root + 'Graficas/Y_predicha para el profile_{}.png'.format(test_samples[i])
            plt.savefig(dir_graf, format='png')
            plt.close()

            dir_pred = model_result_root + 'Ficheros/prediction_profile_{}.csv'.format(test_samples[i])
            y_1 = y[i * num_samples:i * num_samples + num_samples]
            y_hat_1 = y_hat[i * num_samples:i * num_samples + num_samples]
            y_1 = np.reshape(y_1, (len(y_1), 1))
            y_hat_1 = np.reshape(y_hat_1, (len(y_hat_1), 1))
            predictions = np.concatenate((y_1, y_hat_1), axis=1)
            pred_df = pd.DataFrame(predictions, columns=['Y_1', 'y_hat_1'])
            pred_df.to_csv(dir_pred, sep='\t', header=True, index=False)

        data = [model_name, mse, rmse, mae, pearson_correlation[0]]
        print('MSE:{}, RMSE:{}, MAE:{}, R:{}'.format(mse, rmse, mae, pearson_correlation[0]))

        metrics = pd.DataFrame([data], columns=['model_name', 'MSE', 'RMSE', 'MAE', 'Pearson_correlation'])
        metrics.to_excel(model_result_root + 'results.xlsx', header=True, index=False)

        result_list.append(data)

    all_results = pd.DataFrame(result_list, columns=['model_name', 'MSE', 'RMSE', 'MAE', 'Pearson_correlation'])
    all_results.to_excel(results_root + 'all_results.xlsx', header=True, index=False)

    # a = tf.keras.utils.serialize_keras_object(model)
    # print(model.layers[2].get_config())
