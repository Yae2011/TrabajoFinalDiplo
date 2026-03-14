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

#YAE: ###############################
        ###### PESTAÑA BOSQUES ALEATORIOS
with gr.Tab("Bosques Aleatorios"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE BOSQUES ALEATORIOS", elem_classes="title-text")
with gr.Row():
            # COLUMNA IZQUIERDA (COMBOS)
             gr.HTML("PROVINCIA", elem_classes="info-display-7")
             provincia_rf = gr.Dropdown(label="", 
                                            choices=["Elegir provincia", "BS AS"],
                                            value="Elegir provincia")
                    
btn_rf = gr.Button("Predecir")

# COLUMNA DERECHA (GRAFICOS)
with gr.Column(scale=3):
                    salida_rf = gr.Plot(label="Predicción Random Forest")  

            # Eventos YAE
            #cultivo_rf.select(
             #                   rf_on_cultivo,
              #                  inputs=cultivo_rf,
               #                 outputs=provincia_rf
                #            ) 