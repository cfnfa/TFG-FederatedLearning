from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def data_normalize(train_dataset):
    scaler = MinMaxScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]

    scaler.fit(all_dataset)

    for i in range(len(train_dataset)):
        train_dataset[i, :, :] = scaler.transform(train_dataset[i, :, :])

    return train_dataset


def data_normalization(dataset):
    '''
    GLUCOSA: 400
    RITMO CARDIACO: 300
    INSULINA: 3
    CARBOHIDRATOS: 7,5
    '''
    reshape = np.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2]))
    max = reshape.max(axis=0)
    mean = reshape.mean(axis=0)
    scale = np.array([1, 1, 1])
    x_mean = np.subtract(dataset, mean)
    normalized_dataset = np.divide(x_mean, max)
    #normalized_dataset = np.divide(normalized_dataset, scale)

    return normalized_dataset, mean, max

def data_denormalisation(dataset, mean):
    '''
    GLUCOSA: 400
    RITMO CARDIACO: 300
    INSULINA: 3
    CARBOHIDRATOS: 7,5
    '''
    max = np.array([400])
    denormalised_dataset = np.add(np.multiply(dataset, max), mean[0])
    #normalized_dataset = np.divide(normalized_dataset, scale)

    return denormalised_dataset

def data_denormalisation_all(dataset, mean, maxi):
    '''
    GLUCOSA: 400
    RITMO CARDIACO: 300
    INSULINA: 3
    CARBOHIDRATOS: 7,5
    '''
    max = np.array([400, 3, 7.5])
    aa = np.multiply(dataset, maxi)
    denormalised_dataset = np.add(aa, mean)
    #normalized_dataset = np.divide(normalized_dataset, scale)

    return denormalised_dataset

def data_standarization(train_dataset, eval_dataset):
    scaler = StandardScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset) + len(eval_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset) + len(eval_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]
        else:
            index = i - len(train_dataset)
            all_dataset[i * samples:samples * (1 + i), :] = eval_dataset[index, :, :]

    scaler.fit(all_dataset)

    for i in range(len(train_dataset)):
        train_dataset[i, :, :] = scaler.transform(train_dataset[i, :, :])
    for i in range(len(eval_dataset)):
        eval_dataset[i, :, :] = scaler.transform(eval_dataset[i, :, :])

    return train_dataset, eval_dataset, scaler

def data_test_standarization_2(train_dataset, eval_dataset, test_dataset):
    scaler = StandardScaler()
    samples, features = train_dataset[0].shape
    all_dataset = np.empty(((len(train_dataset) + len(eval_dataset)) * samples, features))
    all_dataset[:samples] = train_dataset[0, :, :]
    for i in range(1, len(train_dataset) + len(eval_dataset)):
        if i < len(train_dataset):
            all_dataset[i * samples:samples * (1 + i), :] = train_dataset[i, :, :]
        else:
            index = i - len(train_dataset)
            all_dataset[i * samples:samples * (1 + i), :] = eval_dataset[index, :, :]

    scaler.fit(all_dataset)
    mean = np.mean(all_dataset, axis=0)
    stdev = np.std(all_dataset, axis=0)

    for i in range(len(test_dataset)):
        test_dataset[i, :, :] = scaler.transform(test_dataset[i, :, :])

    return test_dataset, mean[0], stdev[0]
