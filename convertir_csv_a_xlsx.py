import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os

def procesar_carpeta():
    # Crear la ventana raíz de tkinter y ocultarla
    root = tk.Tk()
    root.withdraw()

    # 0. Solicitar al usuario un nombre (una única vez)
    nombre_usuario = simpledialog.askstring("Entrada", "Por favor, ingrese el nombre para los archivos de salida:")
    
    if not nombre_usuario:
        print("No se ingresó un nombre.")
        return

    # Seleccionar la carpeta
    carpeta_seleccionada = filedialog.askdirectory(title="Seleccionar carpeta con archivos CSV")

    if not carpeta_seleccionada:
        print("No se seleccionó ninguna carpeta.")
        return

    archivos_procesados = 0
    errores = []

    # Recorrer todos los archivos de la carpeta
    for archivo in os.listdir(carpeta_seleccionada):
        if archivo.lower().endswith(".csv"):
            ruta_completa = os.path.join(carpeta_seleccionada, archivo)
            print(f"Procesando archivo: {archivo}...")
            
            try:
                # 2. Obtener el número de año de los primeros cuatro caracteres
                periodo_str = archivo[:4]
                
                if not periodo_str.isdigit():
                    errores.append(f"{archivo}: El nombre no comienza con 4 dígitos.")
                    continue
                
                periodo = int(periodo_str)

                # 1. Guardar el archivo como un dataframe utilizando el punto y coma como separador
                df = pd.read_csv(ruta_completa, sep=';')

                # 2. Agregar una primera columna llamada "periodo"
                df.insert(0, 'periodo', periodo)

                # 3. Guardar el archivo en formato XLSX
                # Nombre formado por el número de año seguido de un espacio y luego el nombre indicado por el usuario
                nombre_base_salida = f"{periodo_str} {nombre_usuario}"
                
                # Guardar XLSX
                ruta_salida_xlsx = os.path.join(carpeta_seleccionada, nombre_base_salida + ".xlsx")
                df.to_excel(ruta_salida_xlsx, index=False)
                
                # Guardar CSV (UTF-8, separador coma)
                ruta_salida_csv = os.path.join(carpeta_seleccionada, nombre_base_salida + ".csv")
                df.to_csv(ruta_salida_csv, sep=',', encoding='utf-8', index=False)

                print(f"  -> Guardado como XLSX y CSV.")
                archivos_procesados += 1

            except Exception as e:
                errores.append(f"{archivo}: {str(e)}")

    # Mostrar resumen
    mensaje = f"Proceso finalizado.\nArchivos procesados correctamente: {archivos_procesados}"
    if errores:
        mensaje += "\n\nErrores encontrados:\n" + "\n".join(errores)
        messagebox.showwarning("Resultados con advertencias", mensaje)
    else:
        messagebox.showinfo("Éxito", mensaje)

if __name__ == "__main__":
    procesar_carpeta()
