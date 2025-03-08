import pandas as pd
import os
from pathlib import Path

def merge():
    input_dir = Path('cleaned_excel')
    if not os.path.exists(input_dir):
        return

    output_dir = Path('final_data_prueba')
    os.makedirs(output_dir, exist_ok=True)

  
    all_data = os.listdir(input_dir)

    for file_name in all_data:
        file_path = input_dir / file_name
        base_name = file_name.replace('.xlsx', '')  # quitar la extensión .xlsx
        
        xls = pd.ExcelFile(file_path)

        bolus_df = pd.read_excel(xls, 'Bolus')
        basal_df = pd.read_excel(xls, 'Basal')
        cgm_df = pd.read_excel(xls, 'CGM')
        carb_df = pd.read_excel(xls, 'Carb Input')

        # Seleccionar las columnas necesarias de cada hoja
        bolus_df_selected = bolus_df[['Local Time', 'Normal']].rename(columns={'Normal': 'Bolus'})
        basal_df_selected = basal_df[['Local Time', 'Value']].rename(columns={'Value': 'Basal'})
        cgm_df_selected = cgm_df[['Local Time', 'Value']].rename(columns={'Value': 'CGM(mg/dl)'})
        carb_df_selected = carb_df[['Local Time', 'Carb Input']]

        # Asegurarse de que la columna 'Local Time' esté normalizada 
        for df in [bolus_df_selected, basal_df_selected, cgm_df_selected, carb_df_selected]:
            df['Local Time'] = pd.to_datetime(df['Local Time']).dt.floor('T')  # Redondear a minutos

        # Unir las tablas según la columna "Local Time"
        merged_df = pd.merge(bolus_df_selected, basal_df_selected, on='Local Time', how='outer')
        merged_df = pd.merge(merged_df, cgm_df_selected, on='Local Time', how='outer')
        merged_df = pd.merge(merged_df, carb_df_selected, on='Local Time', how='outer')

        # Ordenar por la columna "Local Time"
        merged_df = merged_df.sort_values(by='Local Time')

        # Limitar los valores de CGM para que estén dentro del rango de 0 a 400
        merged_df['CGM(mg/dl)'] = merged_df['CGM(mg/dl)'].clip(lower=0, upper=400)

        # Reemplazar los valores vacíos con 0
        merged_df['CGM(mg/dl)'] = merged_df['CGM(mg/dl)'].fillna(0)
        merged_df['Carb Input'] = merged_df['Carb Input'].fillna(0)
        merged_df['Bolus'] = merged_df['Bolus'].fillna(0)

        # Contar las filas antes de eliminar
        #rows_before = len(merged_df)

        # Eliminar las filas donde 'CGM(mg/dl)' es igual a 0
        #merged_df = merged_df[merged_df['CGM(mg/dl)'] != 0]

        # Contar las filas después de eliminar
        #rows_after = len(merged_df)
        #rows_deleted = rows_before - rows_after

        #print(f"Se han eliminado {rows_deleted} filas donde 'CGM(mg/dl)' es igual a 0")

        output_path = output_dir / f'{base_name}.xlsx'
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='Patient Data', index=False)

        print(f"Archivo guardado en {output_path}")

merge()
