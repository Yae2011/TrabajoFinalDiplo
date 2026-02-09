# Elimina provincias-departamentos que poseen menos registros (años con datos)
# que los indicados por el ususario y guarda el archivo depurado con otro nombre

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os

def procesar_archivo_csv():
    # 1. Configuración de la interfaz oculta
    root = tk.Tk()
    root.withdraw()

    # 2. Selección del archivo
    ruta_archivo = filedialog.askopenfilename(
        title="Seleccione el archivo CSV",
        filetypes=[("Archivos CSV", "*.csv")]
    )

    if not ruta_archivo:
        return

    try:
        # Carga de datos
        df = pd.read_csv(ruta_archivo)

        # 3. Análisis previo de combinaciones
        # Generamos el conteo por combinación
        conteo_df = df.groupby(['provincia', 'departamento']).size().reset_index(name='cantidad_anios')
        
        # Métricas iniciales
        total_registros_ini = len(df)
        total_comb_ini = len(conteo_df)

        # Mostrar listado en consola para referencia del usuario
        print("--- Listado de Combinaciones ---")
        print(conteo_df.to_string(index=False))

        # 4. Solicitud de umbral
        umbral = simpledialog.askinteger(
            "Criterio de Depuración", 
            f"Total combinaciones: {total_comb_ini}\nIngrese el mínimo de años requerido:",
            parent=root, minvalue=1
        )

        if umbral is None:
            return

        # 5. Proceso de filtrado
        # Filtrar las combinaciones que cumplen el requisito
        validos = conteo_df[conteo_df['cantidad_anios'] >= umbral]
        
        # Filtrar el DataFrame original (depuración)
        df_depurado = df.merge(validos[['provincia', 'departamento']], on=['provincia', 'departamento'])

        # 6. Cálculo de métricas finales
        total_registros_fin = len(df_depurado)
        total_registros_del = total_registros_ini - total_registros_fin
        total_comb_fin = len(validos)
        total_comb_del = total_comb_ini - total_comb_fin

        # 7. Exportación y reporte final
        nombre_base, extension = os.path.splitext(ruta_archivo)
        ruta_salida = f"{nombre_base}_depurado{extension}"
        df_depurado.to_csv(ruta_salida, index=False)

        # Informe detallado
        informe = (
            f"DEPURACIÓN FINALIZADA\n"
            f"{'-'*30}\n"
            f"REGISTROS (Filas):\n"
            f"  • Total originales: {total_registros_ini}\n"
            f"  • Total eliminados: {total_registros_del}\n"
            f"  • Total remanentes: {total_registros_fin}\n\n"
            f"GEOGRAFÍA (Provincia-Departamento):\n"
            f"  • Combinaciones originales: {total_comb_ini}\n"
            f"  • Combinaciones eliminadas: {total_comb_del}\n"
            f"  • Combinaciones remanentes: {total_comb_fin}\n"
            f"{'-'*30}\n"
            f"Archivo guardado como: {os.path.basename(ruta_salida)}"
        )

        messagebox.showinfo("Informe de Procesamiento", informe)

    except Exception as e:
        messagebox.showerror("Error", f"Falla en el procesamiento: {e}")
    finally:
        root.destroy()

if __name__ == "__main__":
    procesar_archivo_csv()