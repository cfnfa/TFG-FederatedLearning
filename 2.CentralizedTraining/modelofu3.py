import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def preparacion_datos():
    ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = os.listdir(ruta_archivos)

    dataframes = []
    print(f"Encontrados {len(archivos_excel)} archivos en la ruta especificada.")

    for archivo in archivos_excel:
        print(f"Cargando el archivo: {archivo}")
        data = pd.read_excel(os.path.join(ruta_archivos, archivo))
        
        # Filtrar las columnas necesarias
        data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']]
        data_cleaned = data_cleaned.dropna()  # Eliminar filas con valores nulos
        data_cleaned['Paciente'] = archivo.split('.')[0]  # Asumiendo que el nombre del archivo es el ID del paciente
        dataframes.append(data_cleaned)

    # Concatenar los DataFrames de todos los pacientes en uno solo
    data_combined = pd.concat(dataframes, ignore_index=True)
    print("Datos combinados con éxito.")
    
    # Crear listas para datos de entrenamiento y prueba
    train_data_list = []
    test_data_list = []

    # Agrupar por paciente y dividir el conjunto de cada paciente
    for paciente in data_combined['Paciente'].unique():
        paciente_data = data_combined[data_combined['Paciente'] == paciente]
        
        # Dividir en 80% entrenamiento y 20% prueba para cada paciente
        paciente_train, paciente_test = train_test_split(paciente_data, test_size=0.2, random_state=42)
        
        train_data_list.append(paciente_train)
        test_data_list.append(paciente_test)

    # Concatenar los datos de entrenamiento y prueba de todos los pacientes
    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)

    # Normalizar los datos de los conjuntos de entrenamiento y prueba por separado
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_train.fit_transform(train_data.drop(columns=['Paciente']))
    
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    test_scaled = scaler_test.fit_transform(test_data.drop(columns=['Paciente']))

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

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print("Datos preparados. Tamaño del conjunto de entrenamiento:", X_train.shape[0], "Tamaño del conjunto de prueba:", X_test.shape[0])

    # Información adicional sobre los conjuntos de datos
    print(f"\nMuestra de las primeras 3 secuencias de entrenamiento (X) y sus etiquetas (y):")
    for i in range(3):  # Mostrar las primeras 3 muestras de entrenamiento
        print(f"Secuencia {i+1}:")
        print("X:", X_train[i])  # Muestra las características de la secuencia
        print("y:", y_train[i])  # Muestra la etiqueta correspondiente
        print("-" * 50)

    print(f"\nMuestra de las primeras 3 secuencias de prueba (X) y sus etiquetas (y):")
    for i in range(3):  # Mostrar las primeras 3 muestras de prueba
        print(f"Secuencia {i+1}:")
        print("X:", X_test[i])  # Muestra las características de la secuencia
        print("y:", y_test[i])  # Muestra la etiqueta correspondiente
        print("-" * 50)
    
    return X_train, X_test, y_train, y_test, scaler_train, scaler_test, data_combined



def training_evaluating(X_train, X_test, y_train, y_test, sequence_length):
    # Definir el modelo LSTM con activaciones ReLU
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 4), return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')
    print("Modelo compilado. Iniciando entrenamiento...")

    # Implementar EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping])

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    return y_pred, model


def visualize(y_test, y_pred, scaler_test):
    # Desnormalizar usando el escalador de prueba
    y_test_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]

    plt.figure(figsize=(10, 6))

    # Mostrar solo un día de datos (288 pasos de 5 minutos)
    plt.plot(y_test_desnormalized[:288], label='Valor real de CGM (mg/dl)', color='blue')
    plt.plot(y_pred_desnormalized[:288], label='Predicción de CGM (mg/dl)', color='red')
    plt.title('Comparación de valores reales vs predicciones (CGM en mg/dl) - 1 Día')
    plt.xlabel('Tiempo (intervalos de 5 minutos)')
    plt.ylabel('CGM (mg/dl)')
    plt.legend()
    plt.show()

    # Calcular y mostrar métricas de evaluación
    rmse = np.sqrt(mean_squared_error(y_test_desnormalized, y_pred_desnormalized))
    mae = mean_absolute_error(y_test_desnormalized, y_pred_desnormalized)
    print(f'Error Cuadrático Medio (RMSE): {rmse}')
    print(f'Error Absoluto Medio (MAE): {mae}')


# Ejecutar el flujo principal del programa
sequence_length = 12
print("Iniciando la preparación de datos...")
X_train, X_test, y_train, y_test, scaler_train, scaler_test, data_combined = preparacion_datos()

print("Iniciando el entrenamiento y la evaluación del modelo...")
y_pred, model = training_evaluating(X_train, X_test, y_train, y_test, sequence_length)

print("Visualizando resultados...")
visualize(y_test, y_pred, scaler_test)
