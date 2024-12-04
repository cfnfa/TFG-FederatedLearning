import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# t_seq: Tamaño de la secuencia (cuántos pasos atrás se toman en cuenta para hacer una predicción).
# q: Offset (cuántos pasos en el futuro se quiere predecir).
# H: Horizonte de predicción (el tiempo exacto que se está prediciendo en el futuro).
# data_path: Ruta donde se encuentran los archivos Excel con los datos crudos.
# features: Número de características (variables) que se incluirán en el conjunto de datos.
# rescaling y normalize: Indicadores de si los datos deben ser escalados o normalizados.

def create_dataset(t_seq, q, H, data_path, rescaling=False, normalize=False):
    all_dataset = os.listdir(data_path)  # Lista todos los archivos en el directorio
    first = True  # Variable para identificar el primer archivo
    calculate_mean = pd.DataFrame()  # DataFrame para calcular la media de los datos

    for i in all_dataset:
    
        data = pd.read_excel(os.path.join(data_path, i))
        

        # Seleccionamos las columnas necesarias para el modelo
        columns = ['old_CGM', 'total_insulin']  # Lista de columnas
        if "training" in i:
            raw_val = data.loc[:, ['old_CGM']]
            data = data.loc[:, columns[:-1]]
        else:
            raw_val = data.loc[127:, ['old_CGM']]
            data = data.loc[127:, columns[:-1]]

        # Rescaling opcional
        if rescaling:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
            norm = scaler.transform(data)
            data = pd.DataFrame(norm, columns=data.columns)

        # Creamos las secuencias de entrada (trainX) y las salidas (trainY)
        trainX, trainY = dataset(t_seq, q, H, data.to_numpy(), raw_val.to_numpy())
        calculate_mean = pd.concat([calculate_mean, data])

        # Dividimos el dataset en 80% para entrenamiento y 20% para validación
        TRAIN_SIZE = int(0.8 * trainX.shape[0])
        VAL_SIZE = int(0.2 * trainX.shape[0])

        # Convertimos los datos a tensores de float32
        trainX = tf.cast(trainX, dtype='float32')
        trainY = tf.cast(trainY, dtype='float32')

        # Creamos conjuntos de datos de TensorFlow para entrenamiento y validación
        full_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))
        train_ds = full_ds.take(TRAIN_SIZE)
        val_ds = full_ds.skip(TRAIN_SIZE).take(VAL_SIZE)

        # Concatenamos los datasets
        if first:
            all_train = train_ds
            all_val = val_ds
            first = False
        else:
            all_train = all_train.concatenate(train_ds)
            all_val = all_val.concatenate(val_ds)

    # Calculamos el valor máximo y mínimo de los datos (para desescalado)
    maxi = calculate_mean.max()
    mini = calculate_mean.min()

    return all_train, all_val, maxi, mini


# Esta función transforma los datos en secuencias de tiempo (inputs y outputs) para el modelo.
def dataset(t_seq, q, H, train_dataset, raw_val):
    trainX = []  # Lista para las secuencias de entrada
    trainY = []  # Lista para los valores objetivo (valores de glucosa futuros)
    index = 0

    # Verificamos si hay filas con valores nulos o cero y las eliminamos
    while raw_val[index, 0] == 0:
        index += 1

    # Cortamos los datos eliminando las filas inválidas al principio
    train_dataset = train_dataset[index:]
    raw_val = raw_val[index:]

    # Creamos secuencias con TimeseriesGenerator
    train_set = TimeseriesGenerator(train_dataset[:-q], train_dataset[q:, 0], length=t_seq, batch_size=1)
    raw_set = TimeseriesGenerator(raw_val[:-q], raw_val[q:, 0], length=t_seq, batch_size=1)

    # Filtramos las secuencias con valores no nulos
    for i in range(len(train_set)):
        x, y = train_set[i]  # x: secuencia de entrada, y: valor objetivo
        r_x, r_y = raw_set[i]  # r_x, r_y: valores crudos de glucosa

        if np.count_nonzero(r_x) != 0 and r_y != 0:  # Filtramos secuencias con valores válidos
            trainX.append(x)
            trainY.append(y.flatten())

    # Convertimos las listas en arrays de NumPy con la forma adecuada
    trainX = np.array(trainX).reshape((len(trainX), t_seq, train_dataset.shape[-1]))
    trainY = np.array(trainY)

    return trainX, trainY  # Devolvemos las secuencias de entrada y valores futuros




# A PARTIR DE AQUI, Estas funciónes generan secuencias extrapoladas si fuera necesario

# def create_extrapolated(t_seq, q, H, data_path, features=3, mean=0, std=1, rescaling=False):
#     all_dataset = os.listdir(data_path)
#     all_trainX = []
#     all_trainY = []

#     for i in all_dataset:
#         data = pd.read_csv(data_path + i)
#         '''aa = data[data['old_CGM'] <= 0].groupby((data['old_CGM'] != 0).cumsum())
#         for k, v in aa:
#             if v.shape[0] > 40:
#                 print(k)'''
#         if features == 2:
#             columns = ['CGM_value', 'total_insulin', 'old_CGM']
#         elif features == 3:
#             columns = ['old_CGM', 'total_insulin', 'meal', 'old_CGM']
#         elif features == 4:
#             columns = ['CGM_value', 'total_insulin', 'meal', 'HR', 'old_CGM']
#         elif features == 5:
#             columns = ['CGM_value', 'total_insulin', 'meal', 'HR', 'Steps', 'old_CGM']
#         else:
#             columns = ['CGM_value', 'old_CGM']

#         if "training" in i:
#             data = data.loc[:, columns]
#         else:
#             data = data.loc[127:, columns]
#         trainX, trainY = dataset_raw(t_seq, q, H, data.to_numpy())
#         all_trainX.extend(trainX)
#         all_trainY.extend(trainY)
#         # data_days = pd.concat([data_days, data])
#         # data_days = data_days.values.reshape(int(data.shape[0]/288), 288, int(data.shape[-1]))

#     trainX = np.array(all_trainX).reshape((len(all_trainX), t_seq, data.shape[-1] - 1))
#     trainY = np.array(all_trainY)

#     if rescaling:
#         trainX = trainX / [400, 25, 175]

#     return trainX, trainY


# def dataset_raw(t_seq, q, H, train_dataset):
#     trainX = []
#     trainY = []
#     index = 0
#     a = train_dataset[index, 0]
#     while train_dataset[index, 0] == 0:
#         index += 1
#     train_dataset = train_dataset[index:]
#     train_set = TimeseriesGenerator(train_dataset[:-q, :-1], train_dataset[q:, -1], length=t_seq,
#                                     batch_size=1)

#     for i in range(len(train_set)):
#         x, y = train_set[i]
#         if y[-1] != 0:
#             trainX.append(x)
#             trainY.append(y.flatten())
#     trainX = np.array(trainX).reshape((len(trainX), t_seq, train_dataset.shape[-1] - 1))
#     trainY = np.array(trainY)
#     return trainX, trainY
