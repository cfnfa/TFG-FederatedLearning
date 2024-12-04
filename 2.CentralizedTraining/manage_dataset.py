import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def create_dataset(train_samples, eval_samples, test_samples, data_path, features=3):

    train_dataset = np.empty(shape=(len(train_samples), 288, features))
    eval_dataset = np.empty(shape=(len(eval_samples), 288, features))
    test_dataset = np.empty(shape=(len(test_samples), 288, features))

    glu_columns = ['finger', 'glucose']
    ins_columns = ['mg/dl']
    ing_columns = ['CHO']
    hr_columns = ['bpm']
    steps_columns = ['steps']

    index = 0
    for i in train_samples:
        glu_path = data_path + 'glucose/Glucometro_sensor_dia_{}.txt'.format(i)
        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        glu_data = glu_data.to_numpy()
        train_dataset[index, :, 0] = glu_data[:, 1]

        if features >= 2:
            ins_path = data_path + 'insulin/Insulina_ADM_288_dia_{}.txt'.format(i)
            ins_data = pd.read_csv(ins_path, delim_whitespace=True, names=ins_columns, dtype=np.float64)
            ins_data = ins_data.to_numpy()
            train_dataset[index, :, 1] = ins_data[:, 0]

        if features >= 3:
            ing_path = data_path + 'ingest/Glucosa_ING_288_dia_{}.txt'.format(i)
            ing_data = pd.read_csv(ing_path, delim_whitespace=True, names=ing_columns, dtype=np.float64)
            ing_data = ing_data.to_numpy()
            train_dataset[index, :, 2] = ing_data[:, 0]

        if features >= 4:
            hr_data = data_path + 'ritmo-cardiaco/Ritmo_cardiaco_dia_{}.txt'.format(i)
            hr_data = pd.read_csv(hr_data, delim_whitespace=True, names=hr_columns, dtype=np.float64)
            hr_data = hr_data.to_numpy()
            train_dataset[index, :, 3] = hr_data[:, 0]

        if features >= 5:
            steps_data = data_path + 'Pasos/Pasos_dia_{}.txt'.format(i)
            steps_data = pd.read_csv(steps_data, delim_whitespace=True, names=steps_columns, dtype=np.float64)
            steps_data = steps_data.to_numpy()
            train_dataset[index, :, 4] = steps_data[:, 0]

        index += 1

    index = 0

    for i in test_samples:
        glu_path = data_path + 'glucose/Glucometro_sensor_dia_{}.txt'.format(i)
        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        glu_data = glu_data.to_numpy()
        test_dataset[index, :, 0] = glu_data[:, 1]

        if features >= 2:
            ins_path = data_path + 'insulin/Insulina_ADM_288_dia_{}.txt'.format(i)
            ins_data = pd.read_csv(ins_path, delim_whitespace=True, names=ins_columns, dtype=np.float64)
            ins_data = ins_data.to_numpy()
            test_dataset[index, :, 1] = ins_data[:, 0]

        if features >= 3:
            ing_path = data_path + 'ingest/Glucosa_ING_288_dia_{}.txt'.format(i)
            ing_data = pd.read_csv(ing_path, delim_whitespace=True, names=ing_columns, dtype=np.float64)
            ing_data = ing_data.to_numpy()
            test_dataset[index, :, 2] = ing_data[:, 0]

        if features >= 4:
            hr_data = data_path + 'ritmo-cardiaco/Ritmo_cardiaco_dia_{}.txt'.format(i)
            hr_data = pd.read_csv(hr_data, delim_whitespace=True, names=hr_columns, dtype=np.float64)
            hr_data = hr_data.to_numpy()
            test_dataset[index, :, 3] = hr_data[:, 0]

        if features >= 5:
            steps_data = data_path + 'Pasos/Pasos_dia_{}.txt'.format(i)
            steps_data = pd.read_csv(steps_data, delim_whitespace=True, names=steps_columns, dtype=np.float64)
            steps_data = steps_data.to_numpy()
            test_dataset[index, :, 4] = steps_data[:, 0]

        index += 1

    return train_dataset, eval_dataset, test_dataset


def get_interpolated_glucose(train_samples, data_path):

    train_dataset = np.empty(shape=(len(train_samples), 288, 1))

    glu_columns = ['sample', 'finger', 'glucose']

    index = 0
    for i in train_samples:
        glu_path = data_path + 'Interpolados/Glucometro_sensor_dia_{}.txt'.format(i)
        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        glu_data = glu_data.to_numpy()
        train_dataset[index, :, 0] = glu_data[48:, 2]

        index += 1

    return train_dataset


def prepare_dataset(t_seq, q, H, dataset):
    # preparamos el dataset para crear una sliding window
    len_file = len(dataset[0]) - t_seq - q - H
    dataX2, dataY2 = [], []  # shape = (samples, file_length, prev_points=timesteps, features)
    for index in range(len(dataset)):  # dataset
        dataX = []
        dataY = np.empty(shape=(len_file, H))
        for i in range(len_file):
            a = dataset[index, i:(i + t_seq), :]
            b = dataset[index, i + t_seq + q, 0]
            dataX.append(a)
            dataY[i, 0] = b
        dataX2.append(dataX)
        dataY2.append(dataY)
    return np.array(dataX2), np.array(dataY2)


def dataset(t_seq, q, H, dataset):
    trainX, trainY = prepare_dataset(t_seq, q, H, dataset)
    # Adaptamos los set de datos para que puedan entrar en la LSTM
    trainX = np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], t_seq, trainX.shape[3]))
    trainY = np.reshape(trainY, (trainY.shape[0] * trainY.shape[1], H))
    # Mezclamos las muestras del training y validation set
    # trainX, trainY = shuffle(trainX, trainY)
    return trainX, trainY


def test_dataset(t_seq, q, H, dataset):
    trainX, trainY = prepare_dataset(t_seq, q, H, dataset)
    # Adaptamos los set de datos para que puedan entrar en la LSTM
    trainX = np.reshape(trainX, (trainX.shape[0] * trainX.shape[1], t_seq, trainX.shape[3]))
    trainY = np.reshape(trainY, (trainY.shape[0] * trainY.shape[1], H))
    return trainX, trainY


def create_extrapolated_test_dataset(train_samples, eval_samples, test_samples, data_path):     ### SIN USO ###
    train_dataset = np.empty(shape=(len(train_samples), 288, 3))
    eval_dataset = np.empty(shape=(len(eval_samples), 288, 3))
    test_dataset = np.empty(shape=(len(test_samples), 288, 3))
    read_dataset = np.empty(shape=(len(test_samples), 336, 3))

    glu_columns = ['finger', 'glucose']
    ins_columns = ['mg/dl']
    ing_columns = ['CHO']
    hr_columns = ['bpm']
    index = 0
    for i in train_samples:
        glu_path = data_path + 'glucose/Glucometro_sensor_dia_{}.txt'.format(i)
        ins_path = data_path + 'insulin/Insulina_ADM_288_dia_{}.txt'.format(i)
        ing_path = data_path + 'ingest/Glucosa_ING_288_dia_{}.txt'.format(i)

        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        ins_data = pd.read_csv(ins_path, delim_whitespace=True, names=ins_columns, dtype=np.float64)
        ing_data = pd.read_csv(ing_path, delim_whitespace=True, names=ing_columns, dtype=np.float64)

        glu_data = glu_data.to_numpy()
        ins_data = ins_data.to_numpy()
        ing_data = ing_data.to_numpy()

        train_dataset[index, :, 0] = glu_data[:, 1]
        train_dataset[index, :, 1] = ins_data[:, 0]
        train_dataset[index, :, 2] = ing_data[:, 0]

        index += 1

    index = 0
    for i in eval_samples:
        glu_path = data_path + 'glucose/Glucometro_sensor_dia_{}.txt'.format(i)
        ins_path = data_path + 'insulin/Insulina_ADM_288_dia_{}.txt'.format(i)
        ing_path = data_path + 'ingest/Glucosa_ING_288_dia_{}.txt'.format(i)

        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        ins_data = pd.read_csv(ins_path, delim_whitespace=True, names=ins_columns, dtype=np.float64)
        ing_data = pd.read_csv(ing_path, delim_whitespace=True, names=ing_columns, dtype=np.float64)

        glu_data = glu_data.to_numpy()
        ins_data = ins_data.to_numpy()
        ing_data = ing_data.to_numpy()

        eval_dataset[index, :, 0] = glu_data[:, 1]
        eval_dataset[index, :, 1] = ins_data[:, 0]
        eval_dataset[index, :, 2] = ing_data[:, 0]

        index += 1

    index = 0
    for i in test_samples:
        glu_path = data_path + 'raw_glucose/Glucometro_sensor_dia_{}.txt'.format(i)
        glu_data = pd.read_csv(glu_path, delim_whitespace=True, names=glu_columns, dtype=np.float64)
        glu_data = glu_data.to_numpy()
        print(i)
        read_dataset[index, :, 0] = glu_data[:, 1]
        test_dataset[index, :, 0] = read_dataset[index, 48:, 0]

        index += 1

    return train_dataset, eval_dataset, test_dataset


