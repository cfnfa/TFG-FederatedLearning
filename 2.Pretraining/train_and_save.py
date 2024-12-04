import pathlib
import tensorflow as tf
from matplotlib import pyplot as plt
import librerias.TF_Network as linear_model
from utils import RMSE
import numpy as np


def train_and_save(out_dir, t_seq, q, num_layers, layer1, layer2, lr, trainX, trainY, dropout, recurrent_dropout):
    weights_dir = out_dir + 't_{}_q_{}_l_{}_N1_{}_N2_{}_lr_{}'.format(t_seq, q, num_layers, layer1,
                                                                      layer2, lr)
    x = []
    y=[]
    val_x=[]
    val_y=[]

    for v,k in trainX:
        x.append(v)
        y.append(k)
    for v,k in trainY:
        val_x.append(v)
        val_y.append(k)

    x = np.array(x)
    y = np.array(y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    pathlib.Path(weights_dir).mkdir(parents=True, exist_ok=True)
    LSTM_model = linear_model.TF_LSTM(input_shape=(6, 2),
                                      hidden_units=layer1, hidden_units_2=0, hidden_units_3=layer2,
                                      layers=num_layers, dropout=dropout,
                                      recurrent_dropout=recurrent_dropout)
    model = LSTM_model.build()
    tf.keras.utils.plot_model(model, weights_dir + '/model.png', show_shapes=True, show_layer_names=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=1000, decay_rate=0.95,
                                                                 staircase=True)

    model.compile(loss=RMSE,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError()])
    stop_fit_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    history = model.fit(x, y, validation_data=(val_x, val_y), shuffle=True, epochs=10,
                        batch_size=32, verbose=1, callbacks=[stop_fit_early])
    weights_dir = out_dir + 't_{}_q_{}_l_{}_N1_{}_N2_{}_lr_{}'.format(t_seq, q, num_layers, layer1,
                                                                      layer2, lr)
    save_model_path = weights_dir + '/model.tf'
    model.save_weights(save_model_path, save_format='tf')
    save_model_config = weights_dir + '/model_config.json'
    json_config = model.to_json()
    with open(save_model_config, 'w') as json_file:
        json_file.write(json_config)

    # representamos la pérdida a lo largo del entrenamiento en una gráfica
    epochs_ran = len(history.history['loss'])
    plt.plot(range(0, epochs_ran), history.history['val_loss'],
             label='Validation')
    plt.plot(range(0, epochs_ran), history.history['loss'], label='Training')
    plt.legend()
    plt.savefig(weights_dir + '/Loss_Plot_Linear', format='eps')
    # plt.show()
    plt.close()

    # a = tf.keras.utils.serialize_keras_object(model)
    # print(model.layers[2].get_config())
