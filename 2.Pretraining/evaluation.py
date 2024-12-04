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
import manage_all_dataset as manage_dataset


def main(raw_evaluation=False, year=2018, patients=None, rescaling=False):
    data_dir = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    
    models_root = 'weights/' # Directorio donde se guardan los modelos preentrenados
    results_root = 'results_{}/'.format(year)

    # Inicialización de las variables para la media y desviación estándar de normalización (por defecto 0 y 1)
    mean = 0
    std = 1

    pathlib.Path(models_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(results_root).mkdir(parents=True, exist_ok=True)

    # all_models = os.listdir(models_root)
    # Obtener la lista de modelos preentrenados del directorio models_root
    all_models = os.listdir(models_root)
    result_list = []
    # normalizamos los datos de train y validation
    # test_dataset, mean = dt.data_normalization(test_dataset)  #### OJO: Especificar los máximos ####

    # Iterar sobre cada modelo en el directorio de modelos
    for model_name in all_models:
        print(model_name)
        parameters = model_name.split('_')
        t_seq = int(parameters[1])  # Longitud de la secuencia temporal
        q = int(parameters[3])       # Horizonte de predicción
        H = 1                        # Horizonte fijo (parece ser un hiperparámetro)
        num_layers = int(parameters[5])  # Número de capas LSTM
        layer1 = int(parameters[7])      # Número de neuronas en la primera capa LSTM
        layer2 = int(parameters[9])      # Número de neuronas en la segunda capa LSTM (si existe)
        lr = (parameters[11])            # Tasa de aprendizaje
        dropout, recurrent_dropout = 0, 0  # Parámetros de dropout
        feat = 3  # Número de características (features) en los datos de entrada

        # Directorios para almacenar los resultados y gráficas de este modelo
        model_result_root = results_root + model_name + '/'
        pathlib.Path(model_result_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(model_result_root + 'Graficas/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(model_result_root + 'Ficheros/').mkdir(parents=True, exist_ok=True)

        # Cargar los pesos del modelo desde el archivo correspondiente
        weights_dir = models_root + model_name + '/model.tf'

        # Construir el modelo LSTM en función del número de capas
        if num_layers > 1:
            LSTM_model = linear_model.TF_LSTM(input_shape=(t_seq, feat),
                                              hidden_units=layer1, hidden_units_2=0, hidden_units_3=layer2,
                                              layers=num_layers, dropout=dropout,
                                              recurrent_dropout=recurrent_dropout)
        else:
            LSTM_model = linear_model.TF_LSTM(input_shape=(t_seq, feat),
                                              hidden_units=layer1, hidden_units_2=layer2, hidden_units_3=0,
                                              layers=num_layers, dropout=dropout,
                                              recurrent_dropout=recurrent_dropout)
        # Compilar el modelo con la función de pérdida RMSE y optimizador Adam
        model = LSTM_model.build()
        model.compile(loss=RMSE,
                      optimizer='adam',
                      metrics=[tf.metrics.RootMeanSquaredError(), tf.metrics.MeanAbsoluteError()])
        # Cargar los pesos entrenados en el modelo
        model.load_weights(weights_dir).expect_partial()
        model.summary()

        # Evaluar el modelo para cada paciente proporcionado
        for k in patients:
            # Crear datasets de prueba, ya sea extrapolados o normales
            if raw_evaluation:
                testX, testY = manage_dataset.create_extrapolated(t_seq, q, H, data_dir + '{}/'.format(k), features=feat, mean=mean, std=std, rescaling=rescaling)
            else:
                testX, testY = manage_dataset.create_dataset(t_seq, q, H, data_dir, mean=mean, std=std)

# Evaluar el modelo en el conjunto de prueba
            testY = testY.flatten()
            eval = model.evaluate(testX, testY)
            rmse2 = eval[1]
            mae2 = eval[2]
# Predecir los valores con el modelo
            y_hat = model.predict(testX)
            plot_y_hat = y_hat
            y_hat = y_hat.flatten()
            y = testY.flatten()
# Eliminar valores donde el valor real es 0
            plot_y_hat[y == 0] = 0
            plot_y = testY

            y_hat = y_hat[y != 0]
            y = y[y != 0]
# Calcular la correlación de Pearson, RMSE, MSE y MAE entre las predicciones y los valores reales
            pearson_correlation = pearsonr(y_hat, y)
            rmse = RMSE(y, y_hat)
            mse = mean_squared_error(y, y_hat, squared=True)
            mae = mean_absolute_error(y, y_hat)
            print('MSE:{}, RMSE:{}, MAE:{}, R:{}, RMSE_EVAL:{}, MAE_EVAL:{}'.format(mse, rmse, mae, pearson_correlation[0], rmse2, mae2))


# Graficar los resultados para diferentes perfiles
        num_samples = len(plot_y_hat) / 288

        for i in range(int(num_samples)):
            plt.plot(plot_y[i * 288:i * 288 + 288], label='Original')
            plt.plot(plot_y_hat[i * 288:i * 288 + 288], '-r', label='Prediction')
            plt.legend()
            dir_graf = model_result_root + 'Graficas/Y_predicha para el profile_{}.png'.format(i)
            plt.savefig(dir_graf, format='png')
            plt.close()
# Guardar las predicciones en un archivo CSV
        dir_pred = model_result_root + 'Ficheros/prediction_profile.csv'
        y = np.reshape(y, (len(y), 1))
        y_hat = np.reshape(y_hat, (len(y_hat), 1))
        predictions = np.concatenate((y, y_hat), axis=1)
        pred_df = pd.DataFrame(predictions, columns=['Y_1', 'y_hat_1'])
        pred_df.to_csv(dir_pred, sep='\t', header=True, index=False)

# Guardar los resultados de evaluación en un archivo Excel
        data = [model_name, mse, rmse2, mae, pearson_correlation[0]]

        metrics = pd.DataFrame([data], columns=['model_name', 'MSE', 'RMSE', 'MAE', 'Pearson_correlation'])
        metrics.to_excel(model_result_root + 'results.xlsx', header=True, index=False)

        result_list.append(data)

    all_results = pd.DataFrame(result_list, columns=['model_name', 'MSE', 'RMSE', 'MAE', 'Pearson_correlation'])
    all_results.to_excel(results_root + 'all_results.xlsx', header=True, index=False)

    # a = tf.keras.utils.serialize_keras_object(model)
    # print(model.layers[2].get_config())
