from task import  calcular_escalado_global

global_min, global_max=calcular_escalado_global()


rmse_normalizado = 0.065658

# Desnormalizar RMSE
cgm_range = global_max[2] - global_min[2]  # Calcular el rango de CGM
rmse_desnormalizado = rmse_normalizado * cgm_range

print(f"RMSE desnormalizado: {rmse_desnormalizado:.2f} mg/dl")
