"""Author: Clara Fuertes Novillo"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from task import clarke_error_grid, zone_percentages
import matplotlib.pyplot as plt
from task import load_data_crosssilo, RMSE, load_data_crossdevice, calcular_escalado_global
from scipy.stats import pearsonr


def cargar_modelo_y_predecir(ruta_modelo, X_test, scaler_test, y_test):
    # Cargar el modelo guardado
    print(f"Cargando el modelo desde: {ruta_modelo}")
    modelo_cargado=keras.saving.load_model(ruta_modelo)

    #Por si vuelvo a poner el RMSE
    #modelo_cargado=keras.saving.load_model(ruta_modelo, custom_objects={"RMSE": RMSE})
    # Generar predicciones
    print("Generando predicciones...")
    y_pred = modelo_cargado.predict(X_test)
    
    # Desnormalizar las predicciones y valores reales
    y_test_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]
    
    
    print("Primeros datos de X_test:", y_test_desnormalized[:15])
    
    # Visualizar resultados
    visualizar_predicciones(y_pred, y_test,y_test_desnormalized, y_pred_desnormalized)

    return y_pred

def visualizar_predicciones(y_pred, y_test,y_test_desnormalized, y_pred_desnormalized):
    plt.figure(figsize=(10, 6))


    plt.style.use("seaborn-v0_8")

    # Gráfica de comparación
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_desnormalized[:288], label='Valor real de CGM (mg/dL)', color='blue')
    plt.plot(y_pred_desnormalized[:288], label='Predicción de CGM (mg/dL)', color='orange')
    plt.title('Predicciones a 30 minutos con FL cross-device', fontweight='bold')
    plt.xlabel('Tiempo (min)')
    plt.ylabel('Concentración de glucosa sanguínea (mg/dL)')
    plt.legend()
    plt.grid(color='black', alpha=0.3)
    plt.gca().set_facecolor('#f0f0f0')
    plt.show()

    # Calcular y mostrar métricas de evaluación + clarke grid
    rmse_normal = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(mean_squared_error(y_test_desnormalized, y_pred_desnormalized))
    mae = mean_absolute_error(y_test_desnormalized, y_pred_desnormalized)
    pearson = pearsonr(y_test_desnormalized, y_pred_desnormalized)


    print(f'Error Cuadrático Medio (RMSE): {rmse}')
    print(f'Error Cuadrático Medio NORMALIZADO (RMSE): {rmse_normal}')
    print(f'Error Absoluto Medio (MAE): {mae}')
    
    print(f'Correlación de Pearson (p): {pearson}')

    print("Calculando y visualizando el error de Clarke...")
    plot, zone = clarke_error_grid(pd.Series(y_test_desnormalized), pd.Series(y_pred_desnormalized), "Error de Clarke")
    plot.show()

    total_percentages = zone_percentages("Modelo LSTM", zone)
    print("\nPorcentajes de las zonas en el error de Clarke:")
    print(total_percentages)



# Preparar los datos 
sequence_length = 12
print("Iniciando la preparación de datos...")
X_train, X_test, y_train, y_test, archivo, scaler= load_data_crosssilo(5)

# Ruta del modelo que queremos evaluar y visualizar 
rutamodelo = 'outputs/2024-12-29/22-32-27/model_state_acc_0.057_round_3.keras'

# Cargar el modelo y generar predicciones

print("Cargando el modelo guardado y generando predicciones...")
y_pred = cargar_modelo_y_predecir(rutamodelo, X_train, scaler, y_train)
