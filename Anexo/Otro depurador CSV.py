import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def reconstruccion_total_v2():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("--- Reconstructor de Integridad de Datasets (v2.1) ---")
    ruta_input = filedialog.askopenfilename(title="Selecciona el archivo AVENA")
    
    if not ruta_input:
        return

    ruta_output = ruta_input.replace(".csv", "_RESTR_COMPLETO.csv")
    dataset_limpio = []
    num_cols_esperadas = 0

    try:
        # Abrimos con 'latin-1' o 'utf-8' dependiendo del origen, 
        # 'errors=replace' evita que el script se detenga por un solo byte corrupto
        with open(ruta_input, 'r', encoding='utf-8', errors='replace') as f:
            for i, linea in enumerate(f):
                # Eliminamos absolutamente todos los caracteres de control de fin de línea
                linea_procesada = linea.strip().replace('\r', '').replace('\n', '')
                
                if not linea_procesada:
                    continue
                
                columnas = linea_procesada.split(',')
                
                if i == 0:
                    cabecera = [c.strip() for c in columnas]
                    num_cols_esperadas = len(cabecera)
                    dataset_limpio.append(cabecera)
                    continue

                # Solo procesamos líneas que coincidan con la estructura de la cabecera
                if len(columnas) == num_cols_esperadas:
                    dataset_limpio.append([c.strip() for c in columnas])
                else:
                    # Intento de reparación si hay comas extras al final
                    if len(columnas) > num_cols_esperadas:
                        dataset_limpio.append([c.strip() for c in columnas[:num_cols_esperadas]])

        # Creación del DataFrame
        df_final = pd.DataFrame(dataset_limpio[1:], columns=dataset_limpio[0])
        
        # Limpieza de nombres de columnas por si acaso
        df_final.columns = df_final.columns.str.strip()

        # Conversión explícita de tipos para evitar que la última columna se pierda por ser 'object'
        if 'prod_ton' in df_final.columns:
            df_final['prod_ton'] = pd.to_numeric(df_final['prod_ton'], errors='coerce').fillna(0)

        # GUARDADO: Usamos 'lineterminator' (sin guion bajo) para compatibilidad con Pandas 2.x
        df_final.to_csv(ruta_output, index=False, encoding='utf-8', lineterminator='\n')
        
        print(f"\n[✓] Archivo reparado con éxito: {os.path.basename(ruta_output)}")
        print(f"Estructura final: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
        print(f"Columnas detectadas: {list(df_final.columns)}")

    except Exception as e:
        print(f"\n[!] Error crítico: {e}")

    root.destroy()

if __name__ == "__main__":
    reconstruccion_total_v2()