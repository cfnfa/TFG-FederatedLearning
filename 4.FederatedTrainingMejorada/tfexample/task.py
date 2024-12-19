"""tfexample: A Flower / TensorFlow app."""

import os
import json

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras import layers, models, optimizers
from datetime import datetime
from pathlib import Path
from flwr.common.typing import UserConfig





# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def load_model(sequence_length: int = 6, input_dim: int = 4, learning_rate: float = 0.005):
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
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError()])
    
    return model


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


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
    sequence_length = 6
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

    # Imprimir los primeros datos de X_test y y_test para debugging
    #print("Primeros datos de X_test:", X_test[:5])
    print("Primeros datos de y_test:", y_test[:5])

    return X_train, X_test, y_train, y_test, archivo, scaler
