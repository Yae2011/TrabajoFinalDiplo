import pandas as pd
import os
import unicodedata
import tkinter as tk
from tkinter import filedialog, messagebox

def normalizar_texto(texto):
    """
    Elimina tildes, convierte la Ñ en N y pasa a mayúsculas.
    Mantiene la integridad de datos nulos o no textuales.
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    
    # Normalización NFKD para descomponer caracteres con tildes
    texto_normalizado = unicodedata.normalize('NFKD', texto)
    # Filtrado de caracteres de combinación (acentos)
    texto_sin_tildes = "".join([c for c in texto_normalizado if not unicodedata.combining(c)])
    
    # Reemplazo de caracteres específicos y conversión a mayúsculas
    return texto_sin_tildes.replace('ñ', 'n').replace('Ñ', 'N').upper()

def seleccionar_y_procesar():
    # Inicializar la raíz de tkinter y ocultarla para usar solo los diálogos
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # 1. Selección del archivo original
    file_path = filedialog.askopenfilename(
        title="Seleccione el archivo CSV para procesar y sobreescribir",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )

    if not file_path:
        return

    try:
        # 2. Carga de datos con detección de codificación
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep=';', encoding='latin1')

        # 3. Procesamiento de contenido (excluyendo cabeceras)
        # Identificamos solo columnas de tipo objeto (strings)
        columnas_texto = df.select_dtypes(include=['object']).columns
        for col in columnas_texto:
            df[col] = df[col].apply(normalizar_texto)
            
        # Nota: Se ha eliminado la línea que normalizaba df.columns 
        # para preservar la primera línea original.

        # 4. Sobreescritura del archivo original
        # Se utiliza la misma ruta 'file_path' para el guardado
        df.to_csv(file_path, sep=';', index=False, encoding='utf-8')
        
        messagebox.showinfo("Éxito", f"Archivo sobreescrito correctamente:\n{file_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante el procesamiento: {e}")
    finally:
        root.destroy()

if __name__ == "__main__":
    seleccionar_y_procesar()