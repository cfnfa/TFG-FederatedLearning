""" Author:Clara Fuertes """


from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os

def clean(patients):
    save_dir = 'cleaned_excel/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i in patients:
        # Ruta del archivo de entrada y del archivo de salida
        input_file_path = 'C:/Users/clara/Desktop/TFG/Ensayo1/Ensayo1/Tidepool/{}.xlsx'.format(i)
        output_file_path = os.path.join(save_dir, '{}-extracted.xlsx'.format(i))

        # Leer las pestañas que nos interesen: "Bolus" y "CGM" del archivo Excel
        bolus_df = pd.read_excel(input_file_path, sheet_name='Bolus', parse_dates=['Local Time'])
        cgm_df = pd.read_excel(input_file_path, sheet_name='CGM', parse_dates=['Local Time'])
        basal_df = pd.read_excel(input_file_path, sheet_name='Basal', parse_dates=['Local Time'])
        carbs_df = pd.read_excel(input_file_path, sheet_name='Bolus Calculator', parse_dates=['Local Time'])


        # Seleccionar las columnas deseadas
        bolus_columns = ['Local Time', 'Tidepool Data Type', 'Sub Type', 'Duration (mins)', 'Extended','Normal', 'Expected Normal']
        cgm_columns = ['Local Time', 'Tidepool Data Type', 'Value', 'Units']
        basal_columns=['Local Time', 'Tidepool Data Type','Delivery Type','Duration (mins)', 'Rate']
        carbs_columns=['Local Time','Carb Input']
        
        carbs_filtered_df= carbs_df[carbs_columns]
        bolus_filtered_df = bolus_df[bolus_columns]
        cgm_filtered_df = cgm_df[cgm_columns]
        basal_filtered_df = basal_df[basal_columns]

        # Función para redondear los minutos al múltiplo de 5 anterior
        def round_minutes(dt):
            return dt - timedelta(minutes=dt.minute % 5, seconds=dt.second, microseconds=dt.microsecond)

        # Convertir la columna 'Local Time' a datetime, redondear minutos y eliminar segundos
        for df in [bolus_filtered_df, cgm_filtered_df, basal_filtered_df, carbs_filtered_df]:
            df['Local Time'] = pd.to_datetime(df['Local Time'])
            df['Local Time'] = df['Local Time'].apply(round_minutes)

        # Crear un nuevo archivo Excel con las columnas extraídas
        with pd.ExcelWriter(output_file_path) as writer:
            bolus_filtered_df.to_excel(writer, sheet_name='Bolus', index=False)
            cgm_filtered_df.to_excel(writer, sheet_name='CGM', index=False)
            basal_filtered_df.to_excel(writer, sheet_name='Basal', index=False)
            carbs_filtered_df.to_excel(writer, sheet_name='Carb Input', index=False)


        print(f"Paciente {i} extraído y guardado en {output_file_path}")
    
