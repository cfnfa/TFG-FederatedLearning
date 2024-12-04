import pathlib

import numpy
import tensorflow

import librerias.ConfiguracionOptimizacionParametrosForServer as config
import librerias.DataTransformation as dt
from tensorflow.keras import layers
import librerias.dataset_pacientes_y_ficheros_nuevo as dataset_samples
import manage_all_dataset as manage_dataset
from train import train_and_save
import tensorflow as tf


def main(rescaling=False):
    data_dir = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    out_dir = 'weights/'# Ruta donde se guardarán los pesos del modelo entrenado

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Obtener los parámetros configurados para la optimización de la red
    parameters = config.CombinacionParametrosRedOptimizar()
    # Obtener combinaciones de longitud de secuencia de entrada (t_seq) y valores de horizonte de predicción (q)
    t_seq_comb = parameters.get_combinaciones_tseq()  # Posibles valores para t_seq (longitud de secuencia)
    q_comb = parameters.get_combinaciones_q()  # Posibles valores para q (horizonte de predicción)
    H = 1  # Horizonte fijo de predicción (1 paso hacia adelante)
    
    
    # Obtener combinaciones de número de capas ocultas (layers) y neuronas en cada capa
    layers_comb = parameters.get_combinaciones_n_layers()  # Número de capas
    neurons_layer_1, neurons_layer_2 = parameters.get_fixed_neurons()  # Neuronas en la primera y segunda capa

    # Obtener el learning rate fijo y las tasas de dropout
    lr = parameters.get_fixed_lr()  # Learning rate fijo
    dropout, recurrent_dropout = parameters.get_dropout()  # Dropout y recurrent dropout


    '''# recogemos todas las muestras, ya divididas
    all_samples, training_samples, eval_samples, test_samples = dataset_samples.get_dataset(year=201820)
    # cargamos y dividimos los dataset, juntando todos los datos en un mismo array
    train_dataset, eval_dataset, test_dataset = manage_dataset.create_dataset(training_samples, eval_samples,
                                                                              test_samples,
                                                                              data_dir, features=3)
    train_interpolated = manage_dataset.get_interpolated_glucose(training_samples, data_dir)'''
    # normalizamos los datos de train y validation
    # train_dataset, mean, maxi = dt.data_normalization(train_dataset)  #### OJO: Especificar los máximos ####

    # Iterar sobre todas las combinaciones de longitud de secuencia (t_seq)
    for t_seq in t_seq_comb:
        # Iterar sobre todas las combinaciones de horizonte de predicción (q)
        for q in q_comb:
            # Crear los datasets de entrenamiento y validación para cada combinación
            # También se normalizan los datos si se especifica en rescaling
            train, val, maxi, mini = manage_dataset.create_dataset(t_seq, q, H, data_dir, rescaling=rescaling)
            
            # Iterar sobre todas las combinaciones del número de capas ocultas
            for num_layers in layers_comb:
                # Iterar sobre el número de neuronas en la primera capa
                for layer1 in neurons_layer_1:
                    # Si el modelo tiene más de una capa oculta, iterar sobre las neuronas de la segunda capa
                    if num_layers > 1:
                        for layer2 in neurons_layer_2:
                            # Entrenar y guardar el modelo con las configuraciones actuales
                            train_and_save(out_dir, t_seq, q, num_layers, layer1, layer2, lr, train, val, dropout, recurrent_dropout)
                    else:
                        # Si solo hay una capa, entrenar y guardar el modelo sin segunda capa
                        train_and_save(out_dir, t_seq, q, num_layers, layer1, 0, lr, train, val, dropout, recurrent_dropout)
