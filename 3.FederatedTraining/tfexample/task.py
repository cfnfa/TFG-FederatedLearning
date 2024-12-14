"""tfexample: A Flower / TensorFlow app."""

import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras import layers, models, optimizers



# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def load_model(sequence_length: int = 12, input_dim: int = 4, learning_rate: float = 0.0005):
    # Define an LSTM model for regression y configura Adam optimizer
    model = models.Sequential()
    
    # Añadir la capa de entrada explícita
    model.add(layers.Input(shape=(sequence_length, input_dim)))
    
    # Añadir las capas LSTM y Dense
    model.add(layers.LSTM(64, return_sequences=True, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))  # Salida única para regresión
    
    # Configurar el optimizador y la función de pérdida
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error",metrics=["accuracy"])
    
    return model




fds = None  # Cache FederatedDataset


def load_data(Cid):
    # Ruta de los archivos Excel
    ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = [f for f in os.listdir(ruta_archivos) if f.endswith(".xlsx")]
    
    # Selección del archivo basado en el Cid
    if Cid < 0 or Cid >= len(archivos_excel):
        raise ValueError(f"El Client ID {Cid} no tiene un archivo asociado.")
    
    archivo = archivos_excel[Cid]
    print(f"Cargando el archivo: {archivo}")
    
    # Cargar el archivo Excel
    data = pd.read_excel(os.path.join(ruta_archivos, archivo))
    
    # Filtrar las columnas necesarias y eliminar filas con valores nulos
    data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].dropna()
    
    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_cleaned)

    # Dividir en conjuntos de entrenamiento y prueba (80%-20%)
    train_scaled, test_scaled = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Preparar datos de entrenamiento
    sequence_length = 12
    pasos_adelante = 6
    X_train, y_train = [], []
    for i in range(len(train_scaled) - sequence_length - pasos_adelante):  
        X_train.append(train_scaled[i:i+sequence_length, :])
        y_train.append(train_scaled[i + sequence_length + pasos_adelante - 1, 2])

    # Preparar datos de prueba
    X_test, y_test = [], []
    for i in range(len(test_scaled) - sequence_length - pasos_adelante):  
        X_test.append(test_scaled[i:i+sequence_length, :])
        y_test.append(test_scaled[i + sequence_length + pasos_adelante - 1, 2])

    # Convertir a arreglos NumPy
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print("Datos preparados:")
    print(f"  Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
    print(f"  Tamaño del conjunto de prueba: {X_test.shape[0]}")
    print(f"  Dimensiones de las entradas: {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test, archivo
