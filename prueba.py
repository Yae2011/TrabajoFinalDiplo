import gradio as gr
import pandas as pd
import numpy as np
import os
import base64
# import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import StringIO
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL # Modelo Seasonal-Trend con LOESS para descomposición de series cortas
from statsmodels.tsa.stattools import acf, pacf # Funciones de Autocorrelación y de Autocorrelación Parcial
from statsmodels.tsa.stattools import adfuller # Test de Dickey-Fuller Aumentado
from statsmodels.tsa.arima.model import ARIMA # Modelo ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox # Prueba Ljung-Box para verificar: residuos = ruido blanco
import statsmodels.api as sm # Modelos estadísticos
import pmdarima as pm # Modelo AUTO-ARIMA
import scipy.stats as stats
import seaborn as sns
import pywt
from scipy.signal import stft, welch


# YAE: comienzo bibliotecas 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# YAE: fin bibliotecas 
# region CONSTANTES Y DICCIONARIOS
# --- Constantes ---
DATA_PATH = "./Datasets"
IMAGE_PATH = "./Images"
FILE_MAP = {
    "ARROZ": "arroz_depurado.csv",
    "AVENA": "avena_depurado.csv",
    "GIRASOL": "girasol_depurado.csv",
    "MAÍZ": "maiz_depurado.csv",
    "POROTO": "poroto_depurado.csv",
    "SOJA": "soja_depurado.csv",
    "SORGO": "sorgo_depurado.csv",
    "TRIGO": "trigo_depurado.csv",
    "YERBA MATE": "yerbamate_depurado.csv",
    "VARIABLES": "variables_nacionales.csv"
}
KEY_COLUMNS = ['cultivo', 'periodo', 'provincia', 'departamento']
MIN_REG = 20 # Cantidad mínima de registros para cada serie temporal
NO_EXISTE = 999 # Para indicar que no se aplicó la prueba ADF o que no se calcularon grados de diferenciación en las series
YEAR_MIN = 1930 # Primer año de la serie de datos
YEAR_MAX = 2024 # ültimo año de la serie de datos 


# region FUNCIONES DE LECTURA DE ARCHIVOS
def get_file_path(dataset_type):
    filename = FILE_MAP.get(dataset_type)
    if not filename:
        return None
    return os.path.join(DATA_PATH, filename)

def load_data(dataset_type):
    path = get_file_path(dataset_type)
    if not path or not os.path.exists(path):
        return pd.DataFrame(), ["Archivo no encontrado"]
    
    try:
        df = pd.read_csv(path, encoding='utf-8', skipinitialspace=True, sep=',')
        df.columns = df.columns.str.strip()
        return df, list(df['provincia'].unique())
    except Exception as e:
        return pd.DataFrame(), [f"Error cargando: {e}"]
    
def load_vars(file_vars):
    path = get_file_path(file_vars)
    if not path or not os.path.exists(path):
        return ["Archivo no encontrado"]
    
    try:
        df = pd.read_csv(path, encoding='utf-8', skipinitialspace=True, sep=',')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        return [f"Error cargando: {e}"]

# endregion

# YAE: ############################################################
# region FUNCIONES PARA LA PESTAÑA "BOSQUES ALEATORIOS"
# #################################################################
def rf_on_cultivo(cultivo):

    if cultivo == "Elegir cultivo..." or cultivo is None:
            return gr.update(choices=[], value=None)
    
    df, provincias = load_data(cultivo)

    if df.empty:
        return gr.update(choices=[], value=None)

    provincias_sorted = sorted([str(p) for p in provincias])

        
    return gr.update(
        choices= ["Elegir provincia..."] + provincias_sorted,
        value="Elegir provincia..."
    )

# #################################################################
# endregion  FUNCIONES PARA LA PESTAÑA "BOSQUES ALEATORIOS" 
# YAE: ############################################################


###### INTERFACE GRADIO ######

with gr.Blocks(title="Análisis de Cultivos") as app:
    #YAE_GRF: ###############################
    ###### PESTAÑA BOSQUES ALEATORIOS
         with gr.Tab("Bosques Aleatorios"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE BOSQUES ALEATORIOS", elem_classes="title-text")

            with gr.Row():
            # COLUMNA IZQUIERDA (COMBOS)
                with gr.Column( elem_classes="custom-tab", min_width=200):
                     gr.HTML("CULTIVO", elem_classes="info-display-7")
                     cultivo_rf = gr.Dropdown(label="", 
                                            choices=["Elegir cultivo..."] + list(FILE_MAP.keys()),
                                            value="Elegir cultivo...", elem_classes="custom-dropdown-small")
                     gr.HTML("PROVINCIA", elem_classes="info-display-7")
                     provincia_rf = gr.Dropdown(label="", 
                                            choices=[""],
                                            value=None, 
                                            interactive=True, 
                                            elem_classes="custom-dropdown-small")
                     

                # COLUMNA DERECHA (SUBPESTAÑA)
                with gr.Column(scale=3):
                    #SUBPESTAÑA RF: 
                    with gr.Tab("Algoritmo") as rfPredecir:
                        with gr.Row():
                            with gr.Column(elem_classes="custom-tab-2", scale=20):   
                                 gr.HTML(value="PREDICCION", elem_classes="info-display-2")
                            with gr.Column(min_width=150):
                                 btnPredecir_rf = gr.Button("Predecir")
                        salida_rf = gr.Plot(label="Predicción Random Forest")  

                    #SUBPESTAÑA DataSet:
                    with gr.Tab("DataSet") as rfEda:
                        with gr.Row():
                            with gr.Column(elem_classes="custom-tab-2", scale=20):   
                                 gr.HTML(value="CONTENIDO DE LAS VARIABLES", elem_classes="info-display-2")
                            with gr.Column(min_width=150):
                                btnEda_rf = gr.Button("Mostrar Datos")
                        tableOut_rf = gr.Dataframe(interactive=False, max_height=335)

            # Eventos YAE
            cultivo_rf.change(
                               rf_on_cultivo,
                               inputs=cultivo_rf,
                               outputs=provincia_rf
                           ) 
         # fin BOSQUES ALEATORIOS
        #YAE_GRF: ###############################
if __name__ == "__main__":
    # Ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Carpeta de imágenes
    images_folder = os.path.join(current_dir, "Images")
    if not os.path.exists(images_folder):
        print(f"Advertencia: La carpeta {images_folder} no existe.")
    # Lanzamos la aplicación
        # allowed_paths DEBE incluir las rutas absolutas de las carpetas que contienen recursos
    app.launch(
            allowed_paths=[current_dir, images_folder]
        )