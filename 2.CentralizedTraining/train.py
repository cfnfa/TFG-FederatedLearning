import pathlib

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from librerias.DataTransformation import data_denormalisation, data_denormalisation_all
import librerias.TF_Network as linear_model
from utils import RMSE


def train_and_save(out_dir, t_seq, q, num_layers, layer1, layer2, lr, train, val, dropout, recurrent_dropout):
    # Parámetros básicos del entrenamiento
    BATCH_SIZE = 128  # Tamaño de batch para el entrenamiento
    patience_value = 3  # Valor de paciencia para early stopping (detención temprana)
    
    # Cargamos y preparamos los datasets de entrenamiento y validación con el tamaño de batch definido
    train_ds = train.shuffle(10000).batch(BATCH_SIZE)
    val_ds = val.shuffle(10000).batch(BATCH_SIZE)

    # Obtenemos el número de características del dataset (EN MI CASO SIEMPRE VAN A SER 3)
    features = train_ds.element_spec[0].shape[-1]

    # Definimos la ruta para guardar los pesos del modelo, con un nombre que refleja los parámetros del modelo
    weights_dir = out_dir + 't_{}_q_{}_l_{}_N1_{}_N2_{}_lr_{}'.format(t_seq, q, num_layers, layer1, layer2, lr)
    pathlib.Path(weights_dir).mkdir(parents=True, exist_ok=True)

    # Creación del modelo LSTM con las configuraciones proporcionadas
    LSTM_model = linear_model.TF_LSTM(input_shape=(t_seq, features),  # Dimensiones de entrada
                                      hidden_units=layer1,  # Neuronas en la primera capa oculta
                                      hidden_units_2=0,  # No se usa una segunda capa
                                      hidden_units_3=layer2,  # Neuronas en la segunda capa
                                      layers=num_layers,  # Número de capas en total
                                      dropout=dropout,  # Probabilidad de dropout
                                      recurrent_dropout=recurrent_dropout)  # Dropout recurrente
    model = LSTM_model.build()  # Construimos el modelo
    model.summary()  # Mostramos un resumen del modelo

    # Guardamos una imagen del gráfico del modelo, mostrando sus capas y conexiones
    tf.keras.utils.plot_model(model, weights_dir + '/model.png', show_shapes=True, show_layer_names=True)

    # Definimos el scheduler para la tasa de aprendizaje (learning rate), con una disminución exponencial
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=1000, decay_rate=0.95, staircase=True)
    
    # Definimos la función de pérdida y el optimizador
    loss_rmse = RMSE  # Función de pérdida basada en RMSE
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Optimizador Adam con tasa de aprendizaje ajustable

    # Métricas para el entrenamiento y la validación
    train_loss = tf.keras.metrics.Mean(name='train_loss')  # Promedio de la pérdida en el entrenamiento
    train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')  # RMSE como métrica de exactitud en el entrenamiento
    test_loss = tf.keras.metrics.Mean(name='test_loss')  # Promedio de la pérdida en la validación
    test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')  # RMSE como métrica de exactitud en la validación

    # Función de entrenamiento para una iteración (batch)
    @tf.function
    def train_step(series, labels):
        with tf.GradientTape() as tape:
            predictions = model(series, training=True)  # Predicción del modelo
            loss = loss_rmse(labels, predictions)  # Calculamos la pérdida
        gradients = tape.gradient(loss, model.trainable_variables)  # Calculamos los gradientes
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Actualizamos los pesos del modelo

        # Actualizamos las métricas de pérdida y exactitud
        train_loss(loss)
        train_accuracy(labels, predictions)

    # Función de validación para una iteración (batch)
    @tf.function
    def test_step(series, labels):
        predictions = model(series, training=False)  # Predicción sin actualizar pesos
        t_loss = loss_rmse(labels, predictions)  # Calculamos la pérdida

        # Actualizamos las métricas de pérdida y exactitud
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # Definimos el número máximo de épocas y los historiales de pérdida
    EPOCHS = 100  # Número máximo de épocas
    loss_history = []  # Almacenará el historial de pérdida del entrenamiento
    val_loss_history = []  # Almacenará el historial de pérdida de la validación
    patience = 0  # Inicializamos la paciencia para early stopping
    best = np.Inf  # Inicializamos la mejor pérdida observada

    # Bucle de entrenamiento principal
    for epoch in range(EPOCHS):
        # Entrenamos el modelo en todos los batches de entrenamiento
        for series, labels in train_ds:
            train_step(series, labels)

        # Validamos el modelo en todos los batches de validación
        for test_series, test_labels in val_ds:
            test_step(test_series, test_labels)

        # Imprimimos los resultados de la época actual
        template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))

        # Almacenamos los valores de pérdida en los historiales
        loss_history.append(train_loss.result())
        val_loss_history.append(test_loss.result())

        # Si la pérdida de validación no mejora, aumentamos la paciencia
        if best < test_loss.result():
            patience += 1
        else:
            patience = 0  # Reiniciamos la paciencia si la pérdida mejora
            best = test_loss.result()

        # Si se alcanza el valor de paciencia (sin mejora durante 128 épocas), detenemos el entrenamiento
        if patience == patience_value:
            break

        # Reiniciamos las métricas para la siguiente época
        train_loss.reset_state()
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()

    # Guardamos los pesos del modelo entrenado
    save_model_path = weights_dir + '/model.tf'
    model.save_weights(save_model_path)

    # Guardamos la configuración del modelo en un archivo JSON
    save_model_config = weights_dir + '/model_config.json'
    json_config = model.to_json()
    with open(save_model_config, 'w') as json_file:
        json_file.write(json_config)

    # Graficamos las pérdidas de entrenamiento y validación
    epochs_ran = len(loss_history)  # Número de épocas realizadas
    plt.plot(range(0, epochs_ran), val_loss_history, label='Validation')
    plt.plot(range(0, epochs_ran), loss_history, label='Training')
    plt.legend()
    plt.savefig(weights_dir + '/Loss_Plot_Linear', format='eps')  # Guardamos la gráfica en formato EPS
    plt.show()  # Mostramos la gráfica
    plt.close()