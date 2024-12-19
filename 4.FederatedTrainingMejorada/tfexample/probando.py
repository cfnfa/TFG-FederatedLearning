import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from ClarkeErrorGrid import clarke_error_grid, zone_percentages
import matplotlib.pyplot as plt
from task import load_data


def cargar_modelo_y_predecir(ruta_modelo, X_test, scaler_test, y_test):
    # Cargar el modelo guardado
    print(f"Cargando el modelo desde: {ruta_modelo}")
    modelo_cargado=keras.saving.load_model(ruta_modelo)
    # Generar predicciones
    print("Generando predicciones...")
    y_pred = modelo_cargado.predict(X_test)
    
    # Desnormalizar las predicciones y valores reales
    y_test_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]
    
    print("Primeros datos de X_test:", y_test[:15])
    print("Primeros datos de X_test:", y_test_desnormalized[:15])
    
    # Visualizar resultados
    visualizar_predicciones(y_test_desnormalized, y_pred_desnormalized)
    return y_pred

def visualizar_predicciones(y_test_desnormalized, y_pred_desnormalized):
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

    # # Calcular y visualizar el error de Clarke
    # print("Calculando y visualizando el error de Clarke...")
    # plot, zone = clarke_error_grid(pd.Series(y_test_desnormalized), pd.Series(y_pred_desnormalized), "Error de Clarke")
    # plot.show()

    # # Mostrar porcentajes de cada zona
    # total_percentages = zone_percentages("Modelo Cargado", zone)
    # print("\nPorcentajes de las zonas en el error de Clarke:")
    # print(total_percentages)

# Preparar los datos (asumiendo que ya tienes la función preparacion_datos)
sequence_length = 6
print("Iniciando la preparación de datos...")
X_train, X_test, y_train, y_test, archivo, scaler= load_data(11)


# Ruta del modelo guardado
PESOS = 'outputs/2024-12-19/16-29-06/model_state_acc_0.168_round_2.keras'

# Cargar el modelo y generar predicciones

print("Cargando el modelo guardado y generando predicciones...")
y_pred = cargar_modelo_y_predecir(PESOS, X_test, scaler, y_test)
