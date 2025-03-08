"Author: Clara Fuertes Novillo"


import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ClarkeErrorGrid import clarke_error_grid
from ClarkeErrorGrid import zone_percentages
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from utils import RMSE
from scipy.stats import pearsonr
from datetime import datetime

# # función para reducir el learning rate dinamicamente
# def scheduler(epoch, lr):
#     if epoch < 5:
#         return lr
#     else:
#         return float(lr * tf.math.exp(-0.5))  # Reduce el learning rate exponencialmente

def preparacion_datos():
    ruta_archivos = "C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/final_data"
    archivos_excel = os.listdir(ruta_archivos)

    dataframes = []
    print(f"Encontrados {len(archivos_excel)} archivos en la ruta especificada.")

    for archivo in archivos_excel:
        print(f"Cargando el archivo: {archivo}")
        data = pd.read_excel(os.path.join(ruta_archivos, archivo))
 
        data_cleaned = data[['Bolus', 'Basal', 'CGM(mg/dl)', 'Carb Input']]
        data_cleaned = data_cleaned.dropna()  # Eliminar filas con valores nulos
        data_cleaned['Paciente'] = archivo.split('.')[0]  
        dataframes.append(data_cleaned)

    # Concatenar los DataFrames de todos los pacientes en uno solo
    data_combined = pd.concat(dataframes, ignore_index=True)
    print("Datos combinados con éxito.")
    
    # Agrupar por paciente
    pacientes = data_combined['Paciente'].unique()
    # Dividir el total de los pacientes en 80% para entrenamiento, 10% para validación y 10% para prueba
    train_pacientes, temp_pacientes = train_test_split(pacientes, test_size=0.2, shuffle=False)
    val_pacientes, test_pacientes = train_test_split(temp_pacientes, test_size=0.5, shuffle=False)  
    print("Pacientes seleccionados para el conjunto de entrenamiento:", train_pacientes)
    print("Pacientes seleccionados para el conjunto de validación:", val_pacientes)
    print("Pacientes seleccionados para el conjunto de prueba:", test_pacientes)

 
    train_data = data_combined[data_combined['Paciente'].isin(train_pacientes)]
    val_data = data_combined[data_combined['Paciente'].isin(val_pacientes)]
    test_data = data_combined[data_combined['Paciente'].isin(test_pacientes)]

    # Normalizar los datos de los conjuntos de entrenamiento y prueba por separado
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_train.fit_transform(train_data.drop(columns=['Paciente']))

    scaler_val = MinMaxScaler(feature_range=(0, 1))
    val_scaled = scaler_val.fit_transform(val_data.drop(columns=['Paciente']))
    
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    test_scaled = scaler_test.fit_transform(test_data.drop(columns=['Paciente']))

    
    sequence_length = 12  # Ancho de ventana
    pasos_adelante = 6  # Predicción a 30 minutos

    # Preparar datos en subconjuntos
    X_train, y_train = [], []
    for i in range(len(train_scaled) - sequence_length - pasos_adelante):  
        X_train.append(train_scaled[i:i+sequence_length, :])
        y_train.append(train_scaled[i + sequence_length + pasos_adelante - 1, 2])

    X_val, y_val = [], []
    for i in range(len(val_scaled) - sequence_length - pasos_adelante):  
        X_val.append(val_scaled[i:i+sequence_length, :])
        y_val.append(val_scaled[i + sequence_length + pasos_adelante - 1, 2])

    X_test, y_test = [], []
    for i in range(len(test_scaled) - sequence_length - pasos_adelante):  
        X_test.append(test_scaled[i:i+sequence_length, :])
        y_test.append(test_scaled[i + sequence_length + pasos_adelante - 1, 2])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print("Datos preparados. Tamaño del conjunto de entrenamiento:", X_train.shape[0], "Tamaño del conjunto de validación:", X_val.shape[0], "Tamaño del conjunto de prueba:", X_test.shape[0])
    print("Primeros datos de X_test:", X_test[:5])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_train, scaler_val, scaler_test, data_combined



def training_evaluating(X_train, X_val, X_test, y_train, y_val, y_test, sequence_length):

    wandb.init(
    # configurar el log en wandb
    project="glucose_prediction",
    config={
        "learning_rate":0.0005,
        "epochs":70,
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

    # Compilar 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate), loss=tf.keras.losses.MeanSquaredError() ,metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print("Modelo compilado. Iniciando entrenamiento...")

    # Implementar EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

   
    start_time = datetime.now()

    # Implementar LearningRateScheduler
    #lr_scheduler = LearningRateScheduler(scheduler)

    # Entrenar 
    history = model.fit(X_train, y_train, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, validation_data=(X_val, y_val), 
                        callbacks=[early_stopping, WandbMetricsLogger(), WandbModelCheckpoint("models.keras")])
    
 
    end_time = datetime.now()

    # Devuelve el tiempo total de entrenamiento
    total_training_time = (end_time - start_time).total_seconds()
    print(f"El tiempo total de entrenamiento del modelo fue de {total_training_time:.2f} segundos.")


    # Evaluar 
    model.evaluate(X_test, y_test)
    #print(f'Evaluaciones en el conjunto de prueba (normalizada): {loss}')
    y_pred = model.predict(X_test)

    #Guardar
    model_dir = "modelos_finales"
    os.makedirs(model_dir, exist_ok=True)
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    model_file = os.path.join(model_dir, f"modelo_entrenado_{fecha_actual}.keras")
    model.save(model_file)
    
    wandb.finish()

    return y_pred, model


def visualize(y_test, y_pred, scaler_train, scaler_test):
    # Desnormalizar usando los escaladores de test 
    y_test_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]


    # Graficas de predicciones vs valores reales
    plt.style.use("seaborn-v0_8-dark")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_desnormalized[:288], label='Valor real de CGM (mg/dL)', color='blue')
    plt.plot(y_pred_desnormalized[:288], label='Predicción de CGM (mg/dL)', color='orange')
    plt.title('Predicciones a 30 minutos con aprendizaje centralizado', fontweight='bold')
    plt.xlabel('Tiempo (min)')
    plt.ylabel('Concentración de glucosa sanguínea (mg/dL)')
    plt.legend()
    plt.grid(color='black', alpha=0.3)
    plt.gca().set_facecolor('#f0f0f0')
    plt.show()

    # Calcular y mostrar métricas de evaluación + clarke grid
    rmse_normalizado = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(mean_squared_error(y_test_desnormalized, y_pred_desnormalized))
    mae = mean_absolute_error(y_test_desnormalized, y_pred_desnormalized)
    pearson = pearsonr(y_test_desnormalized, y_pred_desnormalized)
    rmse_carlos = RMSE(y_test_desnormalized, y_pred_desnormalized)

    print(f'Error Cuadrático Medio Normalizado (RMSE): {rmse_normalizado}')
    print(f'Error Cuadrático Medio (RMSE): {rmse}')
    print(f'Error Cuadrático Medio (SEGÚN CARLOS): {rmse_carlos}')
    print(f'Error Absoluto Medio (MAE): {mae}')
    print(f'Correlación de Pearson (p): {pearson}')

    print("Calculando y visualizando el error de Clarke...")
    plot, zone = clarke_error_grid(pd.Series(y_test_desnormalized), pd.Series(y_pred_desnormalized), "Error de Clarke")
    plot.show()
    total_percentages = zone_percentages("Modelo LSTM", zone)
    print("\nPorcentajes de las zonas en el error de Clarke:")
    print(total_percentages)


#Preprocesamiento datos. ventana temporal de 12 pasos
sequence_length = 12
print("Iniciando la preparación de datos...")
X_train, X_val, X_test, y_train, y_val, y_test, scaler_train, scaler_val, scaler_test, data_combined = preparacion_datos()

# Entrenar y evaluar el modelo
print("Iniciando el entrenamiento y la evaluación del modelo...")
y_pred, model = training_evaluating(X_train, X_val, X_test, y_train, y_val, y_test, sequence_length)

# Visualizar resultados
print("Visualizando resultados...")
visualize(y_test, y_pred, scaler_train, scaler_test)
