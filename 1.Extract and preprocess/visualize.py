import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def visualize():
    # Ruta al directorio de archivos de entrada
    input_dir = Path('final_data')
    if not os.path.exists(input_dir):
        return

    output_dir = Path('plots')

    # Crear la carpeta 'plots' si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los archivos en el directorio de entrada
    all_data = os.listdir(input_dir)

    # Configurar un rango fijo para el eje Y (puedes ajustarlo según tus datos)
    y_min = 80  # Valor mínimo en el eje Y
    y_max = 430  # Valor máximo en el eje Y

    for file_name in all_data:
        # Crear la ruta completa al archivo Excel
        file_path = input_dir / file_name
        
        # Cargar el archivo Excel correspondiente
        data = pd.read_excel(file_path, sheet_name='Patient Data')
        
        # Convertir la columna 'Local Time' a tipo datetime
        data['Local Time'] = pd.to_datetime(data['Local Time'])

        # Extraer día
        data['Date'] = data['Local Time'].dt.date

        # Calcular la mediana de los niveles de glucosa por día
        median_daily = data.groupby('Date')['CGM(mg/dl)'].median()

        # Nombre base para los archivos de salida
        base_name = file_name.replace('.xlsx', '')  # Remover la extensión .xlsx

        # Graficar y guardar la mediana diaria
        plt.figure(figsize=(14, 6))
        plt.plot(median_daily.index, median_daily.values, marker='o', linestyle='-', color='orange')
        plt.title('Mediana diaria de nivel de glucosa sanguínea', fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Mediana de Glucosa Sanguínea (mg/dL)')
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Configurar formato de las etiquetas del eje X
        plt.xticks(rotation=45, fontsize=8)  # Tamaño de fuente reducido para evitar que se corten

        # Guardar la gráfica en la carpeta de salida
        daily_plot_path = output_dir / f'{base_name}_mediana_diaria.png'
        plt.savefig(daily_plot_path, bbox_inches='tight')
        plt.close()

    print(f'Gráficos guardados en la carpeta: {output_dir}')

visualize()
