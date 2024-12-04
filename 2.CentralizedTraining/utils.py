import tensorflow.keras.backend as K

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean((K.square(y_pred - y_true))))

def MAE(true, preds):
    return sum(abs(true-preds))/true.shape[0]
