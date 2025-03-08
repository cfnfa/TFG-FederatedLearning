"""Author: Clara Fuertes Novillo"""

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



import matplotlib.pyplot as plt






# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def load_model(sequence_length: int = 12, input_dim: int = 4, learning_rate: float = 0.001):
    # Define LSTM model 
    model = models.Sequential()
    model.add(layers.Input(shape=(sequence_length, input_dim)))
    
    model.add(layers.LSTM(128, return_sequences=True, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1)) 
    
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

def RMSE(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true)))

def calcular_escalado_global():
    ruta_archivos="C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = [f for f in os.listdir(ruta_archivos) if f.endswith(".xlsx")]

    global_min = None
    global_max = None

    for archivo in archivos_excel:
        data = pd.read_excel(os.path.join(ruta_archivos, archivo))
        data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].dropna()

        if global_min is None:
            global_min = data_cleaned.min()
            global_max = data_cleaned.max()
        else:
            global_min = np.minimum(global_min, data_cleaned.min())
            global_max = np.maximum(global_max, data_cleaned.max())

    print("Mínimos globales:", global_min.values)
    print("Máximos globales:", global_max.values)
    return global_min.values, global_max.values


def load_data_crossdevice(Cid):
    
    #global_min, global_max = calcular_escalado_global()
    #para otra implementacion usar la linea anterior
    global_min=np.array([0,0,39,0])
    global_max=np.array([25,1.01325, 400, 200])
    ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = [f for f in os.listdir(ruta_archivos) if f.endswith(".xlsx")]

    # Selección del archivo basado en el Cid
    if Cid < 0 or Cid >= len(archivos_excel):
        raise ValueError(f"El Client ID {Cid} no tiene un archivo asociado.")

    archivo = archivos_excel[Cid]
    print(f"Cargando el archivo: {archivo}")

    # Cargar el archivo Excel, filtrar columnas necesarias
    data = pd.read_excel(os.path.join(ruta_archivos, archivo))
    data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].dropna()

    # Asegurar que los datos estén ordenados temporalmente (si hay una columna temporal disponible)
    # data_cleaned = data_cleaned.sort_values(by='time_column')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = -global_min / (global_max - global_min)
    scaler.scale_ = 1 / (global_max - global_min)
    data_scaled = scaler.transform(data_cleaned)

    # Dividir los datos en conjuntos de entrenamiento y prueba (80%-20%) preservando el orden temporal
    split_index = int(len(data_scaled) * 0.8)  
    train_scaled = data_scaled[:split_index]
    test_scaled = data_scaled[split_index:]

    # Preparar subconjutnos
    sequence_length = 12
    pasos_adelante = 6
    X_train, y_train = [], []
    for i in range(len(train_scaled) - sequence_length - pasos_adelante):
        X_train.append(train_scaled[i:i+sequence_length, :])
        y_train.append(train_scaled[i + sequence_length + pasos_adelante - 1, 2])

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

    # Imprimir los primeros datos de y_test
    print("Primeros datos de y_test:", y_test[:5])

    return X_train, X_test, y_train, y_test, archivo, scaler

def load_data_crosssilo(cid):
    if cid==0:
        ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data_dividido/1"
    elif cid==1:
        ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data_dividido/2"
    elif cid==2:
        ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data_dividido/3"
    elif cid==3:
        ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data_dividido/4"
    elif cid==5:
        ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data_dividido/5"
    else:
        raise ValueError(f"El Client ID {cid} no es válido")


    archivos_excel = os.listdir(ruta_archivos)

    global_min, global_max = calcular_escalado_global()

    # Cargar y concatenar todos los archivos, filtrar columnas necesarias
    dataframes = []
    for archivo in archivos_excel:
        print(f"Cargando el archivo: {archivo}")
        data = pd.read_excel(os.path.join(ruta_archivos, archivo))
        dataframes.append(data)

    data_combined = pd.concat(dataframes, ignore_index=True)

    data_cleaned = data_combined[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = -global_min / (global_max - global_min)
    scaler.scale_ = 1 / (global_max - global_min)
    data_scaled = scaler.transform(data_cleaned)

    # Dividir los datos en conjuntos de entrenamiento y prueba (80%-20%) preservando el orden temporal
    if cid==0 or cid==1 or cid==2 or cid==3:
        split_index = int(len(data_scaled) * 0.8)  # 80% para entrenamiento, 20% para prueba
    elif cid==5:
        split_index = int(len(data_scaled) * 0.8)

    train_scaled = data_scaled[:split_index]
    test_scaled = data_scaled[split_index:]

    # Preparar subconjuntos
    sequence_length = 12
    pasos_adelante = 6
    X_train, y_train = [], []
    for i in range(len(train_scaled) - sequence_length - pasos_adelante):
        X_train.append(train_scaled[i:i+sequence_length, :])
        y_train.append(train_scaled[i + sequence_length + pasos_adelante - 1, 2])

    X_test, y_test = [], []
    for i in range(len(test_scaled) - sequence_length - pasos_adelante):
        X_test.append(test_scaled[i:i+sequence_length, :])
        y_test.append(test_scaled[i + sequence_length + pasos_adelante - 1, 2])


    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print("Datos preparados:")
    print(f"  Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
    print(f"  Tamaño del conjunto de prueba: {X_test.shape[0]}")
    print(f"  Dimensiones de las entradas: {X_train.shape[1:]}")

    # Imprimir los primeros datos de y_test 
    print("Primeros datos de y_test:", y_test[:5])

    return X_train, X_test, y_train, y_test, archivos_excel, scaler



def clarke_error_grid(ref_values, pred_values, title_string):
    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(
        len(ref_values), len(pred_values))

    # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if ref_values.max() > 400 or pred_values.max() > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)))
    if ref_values.min() < 0 or pred_values.min() < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)))


    # Clear plot
    plt.clf()

    plt.style.use("seaborn-v0_8")

    # Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=1)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400) / (400))

    # Plot zone lines
    plt.plot([0, 400], [0, 400], ':', c='black')  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175 / 3, 400 / 1.2], [70, 400], '-',
             c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290], [180, 400], '-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values.iloc[i] <= 70 and pred_values.iloc[i] <= 70) or (
                pred_values.iloc[i] <= 1.2 * ref_values.iloc[i] and pred_values.iloc[i] >= 0.8 * ref_values.iloc[i]):
            zone[0] += 1  # Zone A

        elif (ref_values.iloc[i] >= 180 and pred_values.iloc[i] <= 70) or (
                ref_values.iloc[i] <= 70 and pred_values.iloc[i] >= 180):
            zone[4] += 1  # Zone E

        elif ((ref_values.iloc[i] >= 70 and ref_values.iloc[i] <= 290) and pred_values.iloc[i] >= ref_values.iloc[
            i] + 110) or ((ref_values.iloc[i] >= 130 and ref_values.iloc[i] <= 180) and (
                pred_values.iloc[i] <= (7 / 5) * ref_values.iloc[i] - 182)):
            zone[2] += 1  # Zone C
        elif (ref_values.iloc[i] >= 240 and (pred_values.iloc[i] >= 70 and pred_values.iloc[i] <= 180)) or (
                ref_values.iloc[i] <= 175 / 3 and pred_values.iloc[i] <= 180 and pred_values.iloc[i] >= 70) or (
                (ref_values.iloc[i] >= 175 / 3 and ref_values.iloc[i] <= 70) and pred_values.iloc[i] >= (6 / 5) *
                ref_values.iloc[i]):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return plt, zone


def zone_percentages(file, zone):
    total_points = zone[0] + zone[1] + zone[2] + zone[3] + zone[4]
    A_zone = float(zone[0]/total_points)*100
    B_zone = float(zone[1] / total_points) * 100
    C_zone = float(zone[2] / total_points) * 100
    D_zone = float(zone[3] / total_points) * 100
    E_zone = float(zone[4] / total_points) * 100
    percentages = [(file, A_zone, B_zone, C_zone, D_zone, E_zone)]
    total_percentages = pd.DataFrame(percentages, columns=['Model', 'A_zone', 'B_zone', 'C_zone', 'D_zone', 'E_zone'])
    return total_percentages

def zone_percentages_new(file, zone):
    total_points = zone[0] + zone[1] + zone[2] + zone[3] + zone[4]
    A_zone = float(zone[0] / total_points) * 100
    B_zone = float(zone[1] / total_points) * 100
    C_zone = float(zone[2] / total_points) * 100
    D_zone = float(zone[3] / total_points) * 100
    E_zone = float(zone[4] / total_points) * 100
    percentages = {
        'Model': file,
        'A_zone': A_zone,
        'B_zone': B_zone,
        'C_zone': C_zone,
        'D_zone': D_zone,
        'E_zone': E_zone
    }
    return percentages