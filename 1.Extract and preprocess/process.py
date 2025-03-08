import pandas as pd
import os
import time 

def preprocess_cgm(patients):
    for i in patients:
        file_path = 'C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/cleaned_excel/{}-extracted.xlsx'.format(i)
        if not os.path.exists(file_path):
            print("NO SE HA REALIZADO EL PREPROCESAMIENTO DE CGM")
            continue

        df = pd.read_excel(file_path, sheet_name="CGM")

        # Identificar duplicados basados en la columna 'Local Time'
        duplicates = df.duplicated(subset=['Local Time'], keep=False)

        if not duplicates.any():
            print(f"No se encontraron filas duplicadas en {i}. No se requiere ningún cambio.")
            continue
        
        # Procesar sólo las filas duplicadas
        duplicated_df = df[duplicates]
        print(f"Modificando filas duplicadas en {i}:")
        print(duplicated_df)
        
        # Agrupar y calcular la media de "Value"
        grouped_df = duplicated_df.groupby(['Local Time']).agg({
            'Value': 'mean'
        }).reset_index()
        
        # Mantener los valores de la primera fila de las columnas adicionales, Fusionar los resultados y eliminar las filas originales duplicadas y agregar las filas procesadas
        first_rows = duplicated_df.groupby(['Local Time']).first().reset_index()
        merged_df = first_rows.drop(columns=['Value']).merge(grouped_df, on='Local Time')
        df = df[~duplicates].append(merged_df, ignore_index=True)
        
        # Ordenar en orden descendente según 'Local Time', Sobrescribir la pestaña excel
        df = df.sort_values(by=['Local Time'], ascending=False).reset_index(drop=True)
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
            writer.book.remove(writer.book['CGM'])
            df.to_excel(writer, sheet_name="CGM", index=False)
        
        print(f"Archivo {i} procesado (CGM) y guardado exitosamente.")


def preprocess_bolus(patients):
    for i in patients:
        file_path = f'C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/cleaned_excel/{i}-extracted.xlsx'
        if not os.path.exists(file_path):
            print("NO SE HA REALIZADO EL PREPROCESAMIENTO DE BOLUS")
            continue
 
        df = pd.read_excel(file_path, sheet_name="Bolus")
        
        # Filtrar y procesar filas con bolos de tipo 'dual/square'
        dual_square_df = df[df['Sub Type'] == 'dual/square']
        
        if dual_square_df.empty:
            print(f"No se encontraron filas con SubType 'dual/square' en {i}. No se requiere ningún cambio.")
            continue
        
        new_rows = []
        additional_rows = []
 
        for index, row in dual_square_df.iterrows():
            duration = row['Duration (mins)']
            extended = row['Extended']
            num_new_rows = int(duration / 5)
            
            # Crear una fila adicional con Normal igual a la original y Duration(min) y Extended como None
            additional_row = row.copy()
            additional_row['Duration (mins)'] = None
            additional_row['Extended'] = None
            additional_rows.append(additional_row)

            # Crear nuevas filas
            for j in range(num_new_rows):
                new_row = row.copy()
                new_row['Duration (mins)'] = None
                new_row['Extended'] = None
                new_row['Local Time'] = row['Local Time'] + pd.Timedelta(minutes=5 * j)
                new_row['Normal'] = extended / duration
                new_rows.append(new_row)
        
        # Eliminar las filas originales que se procesaron, agregar nuevas
        df = df[~df.index.isin(dual_square_df.index)]
        df = pd.concat([df, pd.DataFrame(additional_rows), pd.DataFrame(new_rows)], ignore_index=True)
        
        # Identificar duplicados basados en la columna 'Local Time'
        duplicates = df.duplicated(subset=['Local Time'], keep=False)
        
        # Procesar filas duplicadas
        if duplicates.any():
            duplicated_df = df[duplicates]
            print(f"Modificando filas duplicadas en {i}:")
            print(duplicated_df)
            
            # Agrupar y calcular la suma de "Normal"
            grouped_df = duplicated_df.groupby(['Local Time']).agg({
                'Normal': 'sum'
            }).reset_index()
            first_rows = duplicated_df.groupby(['Local Time']).first().reset_index()
            merged_df = first_rows.drop(columns=['Normal']).merge(grouped_df, on='Local Time')
            df = df[~duplicates].append(merged_df, ignore_index=True)
        
        # Ordenar según 'Local Time', sobrescribir la hoja excel
        df = df.sort_values(by=['Local Time'], ascending=False).reset_index(drop=True)
   
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
            writer.book.remove(writer.book['Bolus'])
            df.to_excel(writer, sheet_name="Bolus", index=False)
        
        print(f"Archivo {i} procesado (Bolus) y guardado exitosamente.")



def preprocess_basal(patients):
    for i in patients:
        file_path = f'C:/Users/clara/Desktop/TFG/Codigo TFG/1.Extract and preprocess/cleaned_excel/{i}-extracted.xlsx'
        if not os.path.exists(file_path):
            print(f"NO SE HA REALIZADO EL PREPROCESAMIENTO DE BASAL para {i}")
            continue

        df = pd.read_excel(file_path, sheet_name="Basal", parse_dates=['Local Time'])

        
        df['Insulin per min'] = df['Rate'] / 60
        df['End Time'] = df['Local Time'] + pd.to_timedelta(df['Duration (mins)'], unit='m')

        # Crear un rango de tiempo de 5 minutos para todo el intervalo
        start_time = df['Local Time'].min().floor('5T')
        end_time = df['End Time'].max().ceil('5T')
        time_range = pd.date_range(start=start_time, end=end_time, freq='5T')

        # Crear un DataFrame vacío con los intervalos de 5 minutos
        result = pd.DataFrame(time_range, columns=['Local Time'])
        result['Next Time'] = result['Local Time'] + pd.Timedelta(minutes=5)
        result['Value'] = 0

       
        start = time.time()

        # Para cada fila en el DataFrame original, calcular la cantidad de insulina en cada intervalo de 5 minutos
        for _, row in df.iterrows():
            start = row['Local Time']
            end = row['End Time']

            # Encontrar los intervalos de tiempo que se superponen con el rango [start, end]
            mask = (result['Local Time'] < end) & (result['Next Time'] > start)

            # Para los intervalos afectados, calcular la cantidad de insulina administrada en ese intervalo
            for i in result[mask].index:
                interval_start = result.at[i, 'Local Time']
                interval_end = result.at[i, 'Next Time']

                overlap_start = max(interval_start, start)
                overlap_end = min(interval_end, end)

                overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
                result.at[i, 'Value'] += overlap_minutes * row['Insulin per min']

        end = time.time()

        result = result.drop(columns=['Next Time'])

        # Sobrescribir la hoja 'Basal' en el archivo original
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            result.to_excel(writer, sheet_name="Basal", index=False)

        print(f"Preprocesamiento de Basal completado para {i}. ")