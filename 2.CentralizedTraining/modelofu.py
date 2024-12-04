import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def preparacion_datos():
    ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = os.listdir(ruta_archivos)

    dataframes = []
    print(f"Encontrados {len(archivos_excel)} archivos en la ruta especificada.")

    for archivo in archivos_excel:
        print(f"Cargando el archivo: {archivo}")
        data = pd.read_excel(os.path.join(ruta_archivos, archivo))
        
        # Asegúrate de que las columnas necesarias estén en el DataFrame
        data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']]  # Incluye 'Carb Input'
        
        # Aquí se podrían eliminar las filas con valores nulos si es necesario
        dataframes.append(data_cleaned)

    # Concatenar los DataFrames de todos los pacientes en uno solo
    data_combined = pd.concat(dataframes, ignore_index=True)
    print("Datos combinados con éxito.")
    print("Primeros 5 registros del DataFrame combinado:")
    print(data_combined.head())

    # Definir la longitud de la secuencia de entrada (por ejemplo, 12 datos anteriores para predecir los próximos 25 minutos)
    sequence_length = 12  # 1 hora de datos (12 muestras de 5 minutos)

    X = []
    y = []

    # En lugar de predecir 1 paso (5 minutos), vamos a predecir 5 pasos (25 minutos)
    pasos_adelante = 1  # Predicción a 25 minutos (5 pasos de 5 minutos)

    print("Preparando los datos para el modelo...")

    for i in range(len(data_combined) - sequence_length - pasos_adelante):  
        # Ahora incluimos 'Carb Input' en las características de entrada
        X.append(data_combined[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']].iloc[i:i+sequence_length].values.astype(np.float32))
        y.append(data_combined['CGM(mg/dl)'].iloc[i + sequence_length + pasos_adelante - 1].astype(np.float32))  # Predicción a 25 minutos (5 pasos después)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Datos preparados. Tamaño del conjunto de entrenamiento:", X_train.shape[0], "Tamaño del conjunto de prueba:", X_test.shape[0])
    
    return X_train, X_test, y_train, y_test


def training_evaluating(X_train, X_test, y_train, y_test, sequence_length):
    # Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 4)))  # Cambiar a 4 para incluir Carb Input
    model.add(Dense(1))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Modelo compilado. Iniciando entrenamiento...")

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluar el modelo
    loss = model.evaluate(X_test, y_test)
    print(f'Pérdida en el conjunto de prueba: {loss}')

    # Hacer predicciones
    y_pred = model.predict(X_test)

    return y_pred


def visualize(y_test, y_pred):
    # No necesitamos desnormalizar ya que eliminamos la normalización
    plt.figure(figsize=(10, 6))

    # Solo vamos a mostrar un día de datos (288 pasos de 5 minutos)
    plt.plot(y_test[:288], label='Valor real de CGM (mg/dl)', color='blue')
    plt.plot(y_pred[:288], label='Predicción de CGM (mg/dl)', color='red')
    plt.title('Comparación de valores reales vs predicciones (CGM en mg/dl) - 1 Día')
    plt.xlabel('Tiempo (intervalos de 5 minutos)')
    plt.ylabel('CGM (mg/dl)')
    plt.legend()
    plt.show()


sequence_length = 12
print("Iniciando la preparación de datos...")
X_train, X_test, y_train, y_test = preparacion_datos()

# Hacer predicciones con el modelo
print("Iniciando el entrenamiento y la evaluación del modelo...")
y_pred = training_evaluating(X_train, X_test, y_train, y_test, sequence_length)

# Visualizar los resultados sin desnormalización
print("Visualizando resultados...")
visualize(y_test, y_pred)
