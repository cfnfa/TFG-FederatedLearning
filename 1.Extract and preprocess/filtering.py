import os.path
from pathlib import Path

import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt



#Este código complementa el programa EXTRACT y PREPROCESS y se encarga de realizar un filtrado 
# de los datos de glucosa de los pacientes diabéticos y generar gráficos representativos de los mismos. 

interpolate = False
if interpolate:
    figure_plots = 'plots/normal_filter/'
    csv_files = 'csv/interpolate/'
    filtered_files = 'csv/normal_filter/'
else:
    figure_plots = 'plots/causal_filter/'
    csv_files = 'csv/extrapolate/'
    filtered_files = 'csv/causal_filter/'

all_data = os.listdir(csv_files)

Path(filtered_files).mkdir(parents=True, exist_ok=True)

#Iteración sobre los archivos CSV de los pacientes:
for i in all_data:
    print('*****{}******'.format(i))
    Path(figure_plots + '{}/'.format(i)).mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_files + i)

    #Filtrado de datos y generación de gráficos:
    mov_avg = data[['clave', 'CGM_value']].copy()
    if interpolate:
        mov_avg['CGM_value'] = data['CGM_value'].rolling(5, center=True, min_periods=3).mean()  # se aplica un filtro de ventana móvil
    else:
        mov_avg['CGM_value'] = data['CGM_value'].rolling(5, center=False, min_periods=3).mean()  # se aplica un filtro de ventana móvil

    avg_data_group = mov_avg.groupby('clave')
    data_day_group = data.groupby('clave')

    #Guardado de datos filtrados y gráficos generados:
    for k in data['clave'].drop_duplicates():
        data_day = data_day_group.get_group(k).reset_index()
        avg_data_day = avg_data_group.get_group(k).reset_index()
        plt.subplot(211)
        avg_data_day['CGM_value'].plot(ylim=[0, 400], xlim=[-24, 300])
        plt.title(k)
        plt.subplot(212)
        data_day['CGM_value'].plot(ylim=[0, 400], xlim=[-24, 300], color='r')
        #plt.show()
        plt.savefig(figure_plots + '{}/{}.png'.format(i, k))
        plt.close()

    data['CGM_value'] = mov_avg['CGM_value']
    data.to_csv(filtered_files + i, na_rep='0')

