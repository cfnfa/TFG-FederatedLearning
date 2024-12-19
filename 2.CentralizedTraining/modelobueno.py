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
from ClarkeErrorGrid import clarke_error_grid
from ClarkeErrorGrid import zone_percentages
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from utils import RMSE

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
    
    # Agrupar por paciente
    pacientes = data_combined['Paciente'].unique()
    train_pacientes, test_pacientes = train_test_split(pacientes, test_size=0.2)

    # Imprimir los pacientes seleccionados para prueba
    print("Pacientes seleccionados para el conjunto de prueba:", test_pacientes)

    # Filtrar los datos por pacientes para el conjunto de entrenamiento y prueba
    train_data = data_combined[data_combined['Paciente'].isin(train_pacientes)]
    test_data = data_combined[data_combined['Paciente'].isin(test_pacientes)]

    # Normalizar los datos de los conjuntos de entrenamiento y prueba por separado
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_train.fit_transform(train_data.drop(columns=['Paciente']))
    
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    test_scaled = scaler_test.fit_transform(test_data.drop(columns=['Paciente']))

    # Preparar datos de entrenamiento
    sequence_length = 6  # Longitud de secuencia
    pasos_adelante = 6  # Predicción a 30 minutos

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
    
    return X_train, X_test, y_train, y_test, scaler_train, scaler_test, data_combined



def training_evaluating(X_train, X_test, y_train, y_test, sequence_length):

    wandb.init(
    # set the wandb project where this run will be logged
    project="glucose_prediction",
    config={
        "learning_rate":0.05,
        "epochs":50,
        "batch_size": 64,
        "sequence_length": sequence_length,
        "layer_1": 128,
        "layer_2": 64,
        "activation_1": "relu",
        "activation_2": "relu",
        "dropout": 0.2
    }
    )
    # Definir el modelo LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(wandb.config.layer_1, input_shape=(sequence_length, 4), return_sequences=True, activation=wandb.config.activation_1),
        tf.keras.layers.Dropout(wandb.config.dropout),
        tf.keras.layers.LSTM(wandb.config.layer_2, activation=wandb.config.activation_2),
        tf.keras.layers.Dropout(wandb.config.dropout),
        tf.keras.layers.Dense(1)
    ])
    model.summary()

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate), loss=RMSE ,metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print("Modelo compilado. Iniciando entrenamiento...")

    # Implementar EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping, WandbMetricsLogger(), WandbModelCheckpoint("models.keras")])

    # Evaluar el modelo (no calculamos RMSE aquí)
    model.evaluate(X_test, y_test)
    #print(f'Evaluaciones en el conjunto de prueba (normalizada): {loss}')

    # Hacer predicciones
    y_pred = model.predict(X_test)

    wandb.finish()

    return y_pred, model


def visualize(y_test, y_pred, scaler_train, scaler_test):
    # Desnormalizar usando los escaladores de entrenamiento y prueba
    y_test_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]

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
    rmse_normalizado = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(mean_squared_error(y_test_desnormalized, y_pred_desnormalized))
    mae = mean_absolute_error(y_test_desnormalized, y_pred_desnormalized)
    print(f'Error Cuadrático Medio Normalizado(RMSE): {rmse_normalizado}')
    print(f'Error Cuadrático Medio (RMSE): {rmse}')
    print(f'Error Absoluto Medio (MAE): {mae}')

    # Calcular y visualizar el error de Clarke
    print("Calculando y visualizando el error de Clarke...")
    plot, zone = clarke_error_grid(pd.Series(y_test_desnormalized), pd.Series(y_pred_desnormalized), "Error de Clarke")
    plot.show()

    # Mostrar porcentajes de cada zona
    total_percentages = zone_percentages("Modelo LSTM", zone)
    print("\nPorcentajes de las zonas en el error de Clarke:")
    print(total_percentages)


sequence_length = 6
print("Iniciando la preparación de datos...")
X_train, X_test, y_train, y_test, scaler_train, scaler_test, data_combined = preparacion_datos()

# Hacer predicciones con el modelo
print("Iniciando el entrenamiento y la evaluación del modelo...")
y_pred, model = training_evaluating(X_train, X_test, y_train, y_test, sequence_length)

# Visualizar los resultados
print("Visualizando resultados...")
visualize(y_test, y_pred, scaler_train, scaler_test)
