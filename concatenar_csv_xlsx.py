import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os

def concatenar_archivos():
    # Crear la ventana raíz de tkinter y ocultarla
    root = tk.Tk()
    root.withdraw()

    # 1. Solicitar carpeta de trabajo
    carpeta_seleccionada = filedialog.askdirectory(title="Seleccionar carpeta con archivos CSV a concatenar")

    if not carpeta_seleccionada:
        print("No se seleccionó ninguna carpeta.")
        return

    # 2. Solicitar nombre del archivo de salida
    nombre_salida_base = simpledialog.askstring("Nombre de Salida", "Ingrese el nombre para el archivo final (sin extensión):")
    
    if not nombre_salida_base:
        print("No se ingresó un nombre de salida.")
        return

    # 3. Listar y ordenar archivos CSV
    archivos_csv = [f for f in os.listdir(carpeta_seleccionada) if f.lower().endswith('.csv')]
    archivos_csv.sort() # Orden alfabético

    if not archivos_csv:
        messagebox.showwarning("Atención", "No se encontraron archivos CSV en la carpeta seleccionada.")
        return

    print(f"Encontrados {len(archivos_csv)} archivos CSV. Iniciando concatenación...")

    dfs = []
    for archivo in archivos_csv:
        ruta_completa = os.path.join(carpeta_seleccionada, archivo)
        print(f"Leyendo: {archivo}")
        try:
            # Intentar leer con separador coma (estándar) o punto y coma si falla, 
            # o simplemente asumir uno. El usuario no especificó el separador de entrada,
            # pero en el script anterior usamos punto y coma.
            # Para ser robustos, intentaremos detectar o usar el motor de python.
            # Asumiremos punto y coma dado el contexto anterior, o coma.
            # Probaremos primero con coma, si da error de parseo, punto y coma.
            # O mejor, usaremos 'sep=None' con engine='python' para autodetectar, 
            # aunque es más lento, es más seguro.
            df = pd.read_csv(ruta_completa, sep=None, engine='python')
            dfs.append(df)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")

    if not dfs:
        messagebox.showerror("Error", "No se pudieron leer los archivos CSV.")
        return

    # 4. Concatenar (elimina encabezados intermedios automáticamente al usar pd.concat y dfs separados)
    df_final = pd.concat(dfs, ignore_index=True)

    # 5. Limpieza de datos: Rellenar vacíos y convertir a numérico
    # Reemplazar celdas con solo espacios o vacías por 0
    df_final.replace(r'^\s*$', 0, regex=True, inplace=True)
    
    # Rellenar valores NaN/None con ceros
    df_final.fillna(0, inplace=True)

    # Convertir columnas numéricas
    # Iterar sobre las columnas y tratar de convertir a numérico
    for col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='ignore')

    # Filtrar registros donde departamento es "Enmascarado"
    if 'departamento' in df_final.columns:
        df_final = df_final[df_final['departamento'] != 'Enmascarado']

    # 6. Guardar archivos
    ruta_salida_csv = os.path.join(carpeta_seleccionada, nombre_salida_base + ".csv")
    ruta_salida_xlsx = os.path.join(carpeta_seleccionada, nombre_salida_base + ".xlsx")

    try:
        # Guardar CSV (UTF-8, separador coma)
        df_final.to_csv(ruta_salida_csv, index=False, sep=',', encoding='utf-8')
        
        # Guardar XLSX
        df_final.to_excel(ruta_salida_xlsx, index=False)

        mensaje = f"Proceso finalizado con éxito.\n\nArchivos creados:\n- {nombre_salida_base}.csv\n- {nombre_salida_base}.xlsx"
        print(mensaje)
        messagebox.showinfo("Éxito", mensaje)

    except Exception as e:
        messagebox.showerror("Error al guardar", f"Ocurrió un error al guardar los archivos:\n{str(e)}")

if __name__ == "__main__":
    concatenar_archivos()
