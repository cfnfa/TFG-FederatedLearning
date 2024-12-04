import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def cargar_datos(Cid):
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

    return X_train, X_test, y_train, y_test


def prepare_test_set():
    """Prepare medical dataset for federated learning."""
    file_path = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data/VVC-extracted.xlsx"
    
    # Cargar y limpiar los datos
    data = pd.read_excel(file_path)
    data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].dropna()
    
    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_scaled = scaler.fit_transform(data_cleaned)  # Normalizar todo el conjunto

    # Parámetros para secuencias
    sequence_length = 12
    pasos_adelante = 6
    
    # Crear secuencias
    X_test, y_test = [], []
    for i in range(len(test_scaled) - sequence_length - pasos_adelante):  
        X_test.append(test_scaled[i:i+sequence_length, :])
        y_test.append(test_scaled[i + sequence_length + pasos_adelante - 1, 2])

    # Convertir a arreglos NumPy
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Validar que las dimensiones coincidan
    assert X_test.shape[0] == y_test.shape[0], "X_test y y_test no tienen el mismo número de muestras"

    # Imprimir información sobre los datos
    print("Datos preparados:")
    print(f"  Tamaño del conjunto de prueba: {X_test.shape[0]}")
    print(f"  Dimensiones de las entradas: {X_test.shape[1:]}")

    return X_test, y_test


        
   


