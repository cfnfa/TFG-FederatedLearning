import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def visualize():
    # Ruta al directorio de archivos de entrada
    input_dir = Path('cleaned_excel')
    if not os.path.exists(input_dir):
        return

    output_dir = Path('plots')

    # Crear la carpeta 'plots' si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los archivos en el directorio de entrada
    all_data = os.listdir(input_dir)

    for file_name in all_data:
        # Crear la ruta completa al archivo Excel
        file_path = input_dir / file_name
        
        # Cargar el archivo Excel correspondiente
        data = pd.read_excel(file_path, sheet_name='CGM')
        
        # Convertir la columna 'Local Time' a tipo datetime
        data['Local Time'] = pd.to_datetime(data['Local Time'])

        # Extraer día, mes y año
        data['Date'] = data['Local Time'].dt.date
        data['Month'] = data['Local Time'].dt.to_period('M')
        data['Year'] = data['Local Time'].dt.year

        # Calcular la mediana de los niveles de glucosa por día, mes y año
        median_daily = data.groupby('Date')['Value'].median()
        median_monthly = data.groupby('Month')['Value'].median()
        median_yearly = data.groupby('Year')['Value'].median()

        # Nombre base para los archivos de salida
        base_name = file_name.replace('.xlsx', '')  # Remover la extensión .xlsx

        # Graficar y guardar la mediana diaria
        plt.figure(figsize=(14, 6))
        plt.plot(median_daily.index, median_daily.values, marker='o', linestyle='-', color='blue')
        plt.title(f'Mediana Diaria de Niveles de Glucosa - {base_name}')
        plt.xlabel('Fecha')
        plt.ylabel('Mediana de Glucosa (mg/dL)')
        plt.grid(True)
        plt.xticks(rotation=45)
        daily_plot_path = output_dir / f'{base_name}_mediana_diaria.png'
        plt.savefig(daily_plot_path)
        plt.close()

        # Graficar y guardar la mediana mensual
        plt.figure(figsize=(14, 6))
        plt.plot(median_monthly.index.astype(str), median_monthly.values, marker='o', linestyle='-', color='green')
        plt.title(f'Mediana Mensual de Niveles de Glucosa - {base_name}')
        plt.xlabel('Mes')
        plt.ylabel('Mediana de Glucosa (mg/dL)')
        plt.grid(True)
        plt.xticks(rotation=45)
        monthly_plot_path = output_dir / f'{base_name}_mediana_mensual.png'
        plt.savefig(monthly_plot_path)
        plt.close()

        # Graficar y guardar la mediana anual
        plt.figure(figsize=(10, 6))
        plt.plot(median_yearly.index, median_yearly.values, marker='o', linestyle='-', color='red')
        plt.title(f'Mediana Anual de Niveles de Glucosa - {base_name}')
        plt.xlabel('Año')
        plt.ylabel('Mediana de Glucosa (mg/dL)')
        plt.grid(True)
        yearly_plot_path = output_dir / f'{base_name}_mediana_anual.png'
        plt.savefig(yearly_plot_path)
        plt.close()

    print(f'Gráficos guardados en la carpeta: {output_dir}')

visualize()