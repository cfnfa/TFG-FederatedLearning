import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def visualize():
    input_dir = Path('cleaned_excel')
    if not os.path.exists(input_dir):
        return

    output_dir = Path('graficas')
    os.makedirs(output_dir, exist_ok=True)

 
    all_data = os.listdir(input_dir)

    # Configurar graficas
    y_min = 80  # Valor mínimo en el eje Y
    y_max = 430  # Valor máximo en el eje Y

    for file_name in all_data:
        file_path = input_dir / file_name
        data = pd.read_excel(file_path, sheet_name='CGM')
        
        # Convertir columna 'Local Time' a tipo datetime
        data['Local Time'] = pd.to_datetime(data['Local Time'])

        data['Date'] = data['Local Time'].dt.date
        data['Month'] = data['Local Time'].dt.to_period('M')
        data['Year'] = data['Local Time'].dt.year

        # Calcular la mediana de los niveles de glucosa por día, mes y año
        median_daily = data.groupby('Date')['Value'].median()
        median_monthly = data.groupby('Month')['Value'].median()
        median_yearly = data.groupby('Year')['Value'].median()

        
        base_name = file_name.replace('.xlsx', '')  

        # Graficas mediana diaria
        plt.figure(figsize=(14, 6))
        plt.plot(median_daily.index, median_daily.values, marker='o', linestyle='-', color='orange')
        plt.title(f'Mediana Diaria de nivel de glucosa sanguínea',fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Mediana de Glucosa Sanguínea(mg/dL)')
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.grid(True)
        plt.xticks(rotation=45,fontsize=8)
        daily_plot_path = output_dir / f'{base_name}_mediana_diaria.png'
        plt.savefig(daily_plot_path)
        plt.close()

        # Graficas mediana mensual
        plt.figure(figsize=(14, 6))
        plt.plot(median_monthly.index.astype(str), median_monthly.values, marker='o', linestyle='-', color='green')
        plt.title(f'Mediana Mensual de nivel de glucosa sanguínea',fontweight='bold')
        plt.xlabel('Mes')
        plt.ylabel('Mediana de Glucosa Sanguínea(mg/dL)')
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.grid(True)
        plt.xticks(rotation=45)
        monthly_plot_path = output_dir / f'{base_name}_mediana_mensual.png'
        plt.savefig(monthly_plot_path)
        plt.close()

        # Graficas la mediana anual
        plt.figure(figsize=(10, 6))
        plt.plot(median_yearly.index, median_yearly.values, marker='o', linestyle='-', color='blue')
        plt.title(f'Mediana Anual de nivel de glucosa sanguínea',fontweight='bold')
        plt.xlabel('Año')
        plt.ylabel('Mediana de Glucosa Sanguínea(mg/dL)')
        plt.ylim(y_min, y_max)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.grid(True)
        yearly_plot_path = output_dir / f'{base_name}_mediana_anual.png'
        plt.savefig(yearly_plot_path)
        plt.close()

    print(f'Gráficos guardados en la carpeta: {output_dir}')

visualize()