import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ClarkeErrorGrid import clarke_error_grid
from ClarkeErrorGrid import zone_percentages

from utils import RMSE
from scipy.stats import pearsonr
from datetime import datetime
from modelobueno import preparacion_datos
import keras


def visualize(y_test, X_test, ruta_modelo, scaler_train, scaler_test):

    # Cargar el modelo guardado
    print(f"Cargando el modelo desde: {ruta_modelo}")
    modelo_cargado=keras.saving.load_model(ruta_modelo, custom_objects={"RMSE": RMSE})
    # Generar predicciones
    print("Generando predicciones...")
    y_pred = modelo_cargado.predict(X_test)

    # Desnormalizar usando los escaladores de entrenamiento y prueba
    y_test_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_test), 2)), y_test.reshape(-1, 1), np.zeros((len(y_test), 1))], axis=1))[:, 2]
    y_pred_desnormalized = scaler_test.inverse_transform(np.concatenate([np.zeros((len(y_pred), 2)), y_pred, np.zeros((len(y_pred), 1))], axis=1))[:, 2]

    # Configuración gráficas
    plt.style.use("seaborn-v0_8-dark")

    # Gráfica de comparación
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

    # Calcular y mostrar métricas de evaluación + Clark grid
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

    

#Usar modelo entrenado y guardado par ahacer predicciones y visualizar resultados
print("Iniciando la preparación de datos...")
X_train, X_val, X_test, y_train, y_val, y_test, scaler_train, scaler_val, scaler_test, data_combined = preparacion_datos()

rutamodelo = 'modelos_finales/ModeloBueno1.keras'

print("Visualizando resultados...")
visualize(y_test, X_test, rutamodelo, scaler_train, scaler_test)
