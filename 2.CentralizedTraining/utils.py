import keras
import tensorflow as tf

def RMSE(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true)))


#def MAE(true, preds):
#    return sum(abs(true-preds))/true.shape[0]
