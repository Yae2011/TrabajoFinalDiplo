import gradio as gr
import pandas as pd
import numpy as np
import os
import base64
import math
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
import scipy.stats as stats
import seaborn as sns
import json
import unicodedata

# comienzo prg MAF
import json
import plotly.express as px
import unicodedata
# fin prg MAF


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
}
IMAGE_MAP = {
    "ARROZ": "arroz.png",
    "AVENA": "avena.png",
    "GIRASOL": "girasol.png",
    "MAÍZ": "maiz.png",
    "POROTO": "poroto.png",
    "SOJA": "soja.png",
    "SORGO": "sorgo.png",
    "TRIGO": "trigo.png",
    "YERBA MATE": "yerbamate.png",
}
KEY_COLUMNS = ['cultivo', 'periodo', 'provincia', 'departamento']
MIN_REG = 20 # Cantidad mínima de registros para cada serie temporal
NO_EXISTE = 999 # Para indicar que no se aplicó la prueba ADF o que no se calcularon grados de diferenciación en las series
YEAR_MIN = 1930 # Primer año de la serie de datos
YEAR_MAX = 2024 # ültimo año de la serie de datos 

# Se cargan las descripciones de las variables de los datasets en un diccionario
# para títulos de gráficos de evolución de cultivos
variables = os.path.join(DATA_PATH, "Nombres_Largos.csv")
df_vars = pd.read_csv(variables, header=None, encoding='latin-1', sep=',')
dict_nlargos = df_vars.set_index(df_vars.columns[0])[df_vars.columns[1]].to_dict()
variables = os.path.join(DATA_PATH, "Nombres_Cortos.csv")
df_vars = pd.read_csv(variables, header=None, encoding='latin-1', sep=',')
dict_ncortos = df_vars.set_index(df_vars.columns[0])[df_vars.columns[1]].to_dict()
# endregion


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
        df = pd.read_csv(path, encoding='utf-8', sep=',')
        return df, list(df['provincia'].unique())
    except Exception as e:
        return pd.DataFrame(), [f"Error cargando: {e}"]

# endregion


# region FUNCIONES PARA LA PESTAÑA "EDA"
def tab_EDA_on_load(dataset_type, automatico, interactivo):
    df, provincias = load_data(dataset_type)
    
    if df.empty:
        m_inic = "EL DATASET DE CULTIVOS SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)
    
    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]

    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, prov_first, dpto_first, sector, ambito, True, interactivo)

    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, pd.DataFrame(), gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), \
            gr.update(value=m_inic), gr.update(visible=False), \
            pd.DataFrame(), pd.DataFrame(), \
            None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True), \
            gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
            gr.Checkbox(value=False), gr.Checkbox(value=False)
            
def tab_EDA_on_dataset_change(dataset_type, automatico, interactivo):
    df, provincias = load_data(dataset_type)
    
    if df.empty:
        m_inic = "EL DATASET DE CULTIVOS SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)
    
    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]

    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, prov_first, dpto_first, sector, ambito, True, interactivo)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        # Al actualizar el dataset, la lista "provincia" muestra la primera provincia
        # y la lista "departamento" muestra el primer departamento. La lista "indicador"
        # se actualiza recién luego de presionar el botón "Mostrar Datos"
        return df, pd.DataFrame(), gr.update(choices=provincias_sorted, value=prov_first), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)

def tab_EDA_on_provincia_change(df, df_filtered, dataset_type, provincia, automatico, interactivo):

    # Se arma el listado ordenado de departamentos de la provincia
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    
    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, provincia, dpto_first, sector, ambito, True, interactivo)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)

def tab_EDA_on_departamento_change(df, df_filtered, dataset_type, provincia, departamento, 
                                   automatico, interactivo):

    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True, interactivo)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(value=departamento), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)

def tab_EDA_on_opcion_change(df, df_filtered, dataset_type, provincia, departamento, sector, ambito, 
                             automatico, interactivo):

    if automatico:
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True, interactivo)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(value=departamento), \
                gr.update(value=sector), gr.update(value=ambito), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)

def tab_EDA_create_boxplot_graph(df, interactivo):
    # df: dataset filtrado con columnas con "nombres originales"

    if df is None or df.empty:
        return None
    
    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None

    if interactivo:
        fig = go.Figure()

        for col in cols_to_plot:
            # Se obtiene el nombre corto del indicador
            label = dict_ncortos.get(col, col)
            
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=label,
                boxpoints='outliers',
                fillcolor='blue',
                line=dict(color='black', width=1)
            ))

        fig.update_layout(
            title="DISTRIBUCIÓN DE ESTUDIANTES POR CATEGORÍA",
            template="plotly_white",
            showlegend=False,
            # Dimensiones del gráfico tratando de que se aproximen al figsize=(10, 4)
            # width=1000, 
            height=500,
            xaxis=dict(
                tickangle=90,
                showline=True, # Línea del eje X (borde inferior)
                tickfont=dict(size=10, family="Arial, sans-serif", color="black"),
                linewidth=1,
                linecolor='black',
                mirror=True # Para el recuadro del gráfico
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.5,
                showline=True, # Línea del eje Y (borde izquierdo)
                linewidth=1,
                linecolor='black',
                mirror=True # Para el recuadro del gráfico
            )
        )

    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Se excluyen los valores nan
        data_values = []
        headers = []
        for col in cols_to_plot:
            data_values.append(df[col].dropna())
            headers.append(col)

        new_labels = [dict_ncortos.get(h, h) for h in headers]
        # Se crea el gráfico
        box = ax.boxplot(data_values, patch_artist=True,
                        tick_labels=new_labels,
                        #tick_labels=headers, 
                        medianprops=dict(color="white", linewidth=1.5))
        
        # Cajas color celeste
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            
        ax.set_title("DISTRIBUCIÓN DE ESTUDIANTES POR CATEGORÍA")
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
    
    return fig

def tab_EDA_create_evolution_graph(df, indicador, serie=True, med_glob=True, tend=True, 
                                   med_mov=False, sd_mov=False, tipo_mov=4, interactivo=False,
                                   h_estatico=5, h_interactivo=350):
    """
    - df: dataset filtrado con columnas con nombres originales ['periodo', indicador]
    - indicador: nombre original de columna
    - serie: muestra la serie de datos
    - med_glob: muestra la media global
    - tend: muestra la tendencia
    - med_mov: muestra la media móvil
    - sd_mov: muestra la SD móvil
    - tipo_mov: ventana para media y SD móviles (2: k=2; 3: k=3 hacia atrás; 4: k=3 centrado; otro: k=2)
    - interactivo: True para Plotly (Web), False para Matplotlib (Estático)
    h_estatico: altura relativa para el gráfico estático en Matplotlib
    h_interactivo: altura en pixeles para el gráfico interactivo con Plotly
    """

    # 1. Validación y Preparación de Datos
    if df is None or df.empty:
        return None
        
    df_sorted = df.sort_values('periodo').reset_index(drop=True)
    x_data = df_sorted['periodo']
    y_data = df_sorted[indicador]

    # Título dinámico (puedes ajustar el diccionario de nombres aquí)
    titulo = f"{dict_nlargos[indicador].upper()}"

    # Configuración de la ventana para cálculos móviles
    if tipo_mov == 3:
        k, centro, lab_m = 3, False, "3 atrás"
    elif tipo_mov == 4:
        k, centro, lab_m = 3, True, "3 centrado"
    else:
        k, centro, lab_m = 2, False, "2"

    # --- OPCIÓN 1: GRÁFICO INTERACTIVO (PLOTLY) ---
    if interactivo:
        fig = go.Figure()

        # Serie principal
        if serie:
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='lines+markers', name='Cultivo',
                line=dict(color='green', width=3),
                marker=dict(color='red', size=8, line=dict(width=1, color='darkred'))
            ))

        # Media Global (Línea horizontal)
        if med_glob:
            media_v = y_data.mean()
            fig.add_hline(y=media_v, line_dash="dash", line_color="skyblue", 
                          annotation_text="Media Global")

        # Línea de Tendencia (Regresión Lineal)
        if tend:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=x_data, y=p(x_data), mode='lines', name='Tendencia',
                line=dict(color='orange', dash='dash', width=2)
            ))

        # Cálculos móviles (Media y Desviación Estándar)
        if med_mov or sd_mov:
            y_med_mov = y_data.rolling(window=k, center=centro).mean()
            
            if med_mov:
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_med_mov, mode='lines', 
                    name=f'Media Móvil ({lab_m})', line=dict(color='purple', dash='dot')
                ))
            
            if sd_mov:
                y_sd_mov = y_data.rolling(window=k, center=centro).std()
                # Banda de variabilidad (Sombreado)
                fig.add_trace(go.Scatter(
                    x=pd.concat([x_data, x_data[::-1]]),
                    y=pd.concat([y_med_mov + y_sd_mov, (y_med_mov - y_sd_mov)[::-1]]),
                    fill='toself', fillcolor='rgba(148, 103, 189, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Banda SD ({lab_m})', hoverinfo='skip'
                ))

        # Ajustes estéticos de Plotly
        fig.update_layout(
            title=titulo,
            # xaxis_title="Año",
            # yaxis_title="Estudiantes",
            template="plotly_white",
            hovermode="x unified",
                        # Dimensiones del gráfico tratando de que se aproximen al figsize=(10, 4)
            # width=1000, 
            height=h_interactivo,
            xaxis=dict(
                tickmode='linear',
                tick0=2011, dtick=1, # range=[2010.5, 2024.5],
                showline=True, # Línea del eje X (borde inferior)
                tickfont=dict(size=10, family="Arial, sans-serif", color="black"),
                linewidth=1,
                linecolor='black',
                mirror=True # Para el recuadro del gráfico
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.5,
                showline=True, # Línea del eje Y (borde izquierdo)
                linewidth=1,
                linecolor='black',
                mirror=True # Para el recuadro del gráfico
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # --- OPCIÓN 2: GRÁFICO ESTÁTICO (MATPLOTLIB) CON LIBERACIÓN DE MEMORIA ---
    else:
        # Creamos la figura explícitamente
        fig, ax = plt.subplots(figsize=(12, h_estatico))

        try:
            
            if serie:
                ax.plot(x_data, y_data, label="Cultivo", marker='o', linewidth=3, 
                        color='green', markerfacecolor='red', markeredgecolor='red')

            if med_glob:
                media_v = y_data.mean()
                ax.axhline(y=media_v, color='skyblue', linestyle='--', linewidth=2, label='Media Global')

            if tend:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), color='orange', linestyle='--', linewidth=2, label='Tendencia')

            if med_mov or sd_mov:
                y_med_mov = y_data.rolling(window=k, center=centro).mean()
                if med_mov:
                    ax.plot(x_data, y_med_mov, color='purple', linestyle='--', label=f'Media Móvil ({lab_m})')
                if sd_mov:
                    y_sd_mov = y_data.rolling(window=k, center=centro).std()
                    ax.fill_between(x_data, y_med_mov - y_sd_mov, y_med_mov + y_sd_mov, 
                                    color='purple', alpha=0.2, label=f'SD Móvil ({lab_m})')

            # Configuración final del eje y cuadrícula
            ax.set_title(titulo)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            plt.tight_layout()
            
            return fig
            
        finally:
            # Aquí liberamos la memoria del backend de Matplotlib
            # Gradio ya ha procesado el objeto 'fig' antes de que este bloque termine.
            plt.close(fig)

def tab_EDA_create_normal_dist_graph(df, indicador):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre original de columna

    # Se crea la figura para el gráfico
    fig, ax = plt.subplots(figsize=(6, 3))
    
    try:
        # Se extraen los datos y se eliminan valores nulos para el cálculo estadístico
        y_data = df[indicador].dropna()
        
        if len(y_data) < 2:
            return fig

        # Se calculan los parámetros de la distribución: Media (mu) y Desviación Estándar (sigma)
        mu = np.mean(y_data)
        sigma = np.std(y_data)
        
        # Se genera el eje x (rango de 4 sigmas para cubrir la base de la campana)
        # Si sigma es 0, evitamos error de división por cero
        if sigma == 0:
            ax.text(0.5, 0.5, "Desviación Estándar es 0", ha='center', va='center')
            return fig
            
        x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        # Función de Densidad de Probabilidad (PDF) Normal
        y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma)**2)

        # Se grafica la curva de la distribución normal
        ax.plot(x_norm, y_norm, color='green', linewidth=3, label='Distribución Normal')

        # Se sombrea el área correspondiente a 1 Desviación Estándar (SD)
        # Representa aproximadamente el 68.2% de la probabilidad
        x_fill = np.linspace(mu - sigma, mu + sigma, 100)
        y_fill = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fill - mu) / sigma)**2)
        ax.fill_between(x_fill, y_fill, color='red', alpha=0.3, label='Área SD (±1σ)')

        # Se grafica la línea vertical representativa de la media
        ax.axvline(x=mu, color='red', linestyle='--', linewidth=2, label=f'Media: {mu:.2f}')

        # Configuración de títulos y etiquetas utilizando el diccionario de nombres largos
        titulo = f"DISTRIBUCIÓN NORMAL: {dict_nlargos[indicador].upper()}"
        ax.set_title(titulo)
        ax.set_xlabel("CULTIVO")
        ax.set_ylabel("DENSIDAD")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Se agrega la leyenda para identificar media y área de dispersión
        ax.legend()
        
        plt.tight_layout() 
        
        return fig
    
    except Exception as e:
        print(f"Error al generar gráfico de distribución normal: {e}")
        return fig
        
    finally:
        # Se cierra la figura para liberar memoria del backend de Matplotlib.
        # Gradio ya ha procesado el objeto 'fig' antes de este cierre.
        plt.close(fig)

def tab_EDA_create_histogram_graph(df, indicador):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre original de columna

    # Creación de la figura
    fig, ax = plt.subplots(figsize=(6, 3))
    
    try:
        data = df[indicador].dropna()
        
        # Generación del histograma con Seaborn
        sns.histplot(data, bins='auto', kde=True, ax=ax, 
            color='red', edgecolor='red', alpha=1,
            shrink=0.8 # Columnas más angostas
        )
        
        # Modificación de la línea de densidad
        if ax.lines:
            line = ax.lines[0]
            line.set_color('blue')
            line.set_linewidth(3)
        
        # Configuración de títulos y etiquetas
        titulo = f"DISTRIBUCIÓN DE FRECUENCIAS: {dict_nlargos[indicador].upper()}"
        ax.set_title(titulo)
        ax.set_xlabel("CULTIVO")
        ax.set_ylabel("FRECUENCIA (AÑOS)")
        
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout() 
        
        return fig
    
    except Exception as e:
        print(f"Error al generar el histograma: {e}")
        return None
        
    finally:
        # Liberación de memoria para compatibilidad con Gradio/Streamlit
        plt.close(fig)

def tab_EDA_create_all_graphs(df, indicador, serie, mg, tend, mm, sd, interactivo=False):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre descriptivo corto de columna

    if df is None or df.empty or indicador is None:
        return None, None, None
    
    # Si no hay columna "período"
    if 'periodo' not in df.columns:
        return None, None, None

    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Se busca la clave original (columna) en el diccionario a partir del valor descriptivo
    indicador = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    # Si el indicador no está en las columnas numéricas del df
    if indicador not in numeric_cols:
        return None, None
    
    fig1 = tab_EDA_create_evolution_graph(df, indicador, serie, mg, tend, mm, sd, 
                                          tipo_mov=4, interactivo=interactivo)
    fig2 = tab_EDA_create_histogram_graph(df, indicador)
    fig3 = tab_EDA_create_normal_dist_graph(df, indicador)
    
    return fig1, fig2, fig3

def tab_EDA_options_graph(df, indicador, serie, mg, tend, mm, sd, interactivo=False):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre descriptivo corto de columna

    if df is None or df.empty or indicador is None:
        return None
    
    # Si no hay columna "período"
    if 'periodo' not in df.columns:
        return None

    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Se busca la clave original (columna) en el diccionario a partir del valor descriptivo
    indicador = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    # Si el indicador no está en las columnas numéricas del df
    if indicador not in numeric_cols:
        return None
    
    fig = tab_EDA_create_evolution_graph(df, indicador, serie, mg, tend, mm, sd, 4, interactivo)
   
    return fig

def tab_EDA_graph_interactive(df, indicador, serie, mg, tend, mm, sd, interactivo=False):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre descriptivo corto de columna

    if df is None or df.empty or indicador is None:
        return None, None
    
    # Si no hay columna "período"
    if 'periodo' not in df.columns:
        return None, None

    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Se busca la clave original (columna) en el diccionario a partir del valor descriptivo
    indicador = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    # Si el indicador no está en las columnas numéricas del df
    if indicador not in numeric_cols:
        return None, None
    
    fig1 = tab_EDA_create_boxplot_graph(df, interactivo)
    fig2 = tab_EDA_create_evolution_graph(df, indicador, serie, mg, tend, mm, sd, 
                                          tipo_mov=4, interactivo=interactivo)
   
    return fig1, fig2
    
def tab_EDA_create_next_all_graphs(df, indicador, serie, mg, tend, mm, sd, interactivo=False):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre descriptivo corto de columna

    # Se obtiene la lista de variables (nombre de columnas numéricas) del df
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Se elimina la columna numérica del 'periodo'
    indicadores_cols = [c for c in numeric_cols if c != 'periodo']
    # Se busca la clave original (columna) en el diccionario a partir del nombre corto descriptivo
    indicador = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    # Se obtiene el índice del indicador actual (nombre corto)
    indice_actual = indicadores_cols.index(indicador)
    # Se obtiene el nombre del indicador siguiente (nombre corto)
    indice_sig = (indice_actual + 1) % len(indicadores_cols)
    nuevo_indicador = indicadores_cols[indice_sig]
    
    # Se genera el gráfico de evolución para el indicador siguiente
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador, serie, mg, tend, mm, sd, 
                                        tipo_mov=4, interactivo=interactivo)

    # Se genera el gráfico de histograma para el indicador siguiente
    fig2 = tab_EDA_create_histogram_graph(df, nuevo_indicador)

    # Se genera el gráfico de distribución normal para el indicador siguiente
    fig3 = tab_EDA_create_normal_dist_graph(df, nuevo_indicador)

    # Se renombra el nuevo indicador con el nombre corto del diccionario para
    # colocarlo en la lista desplegable
    indicador_ncorto = dict_ncortos.get(nuevo_indicador)

    return gr.update(value=indicador_ncorto), fig1, fig2, fig3

def tab_EDA_create_prev_all_graphs(df, indicador, serie, mg, tend, mm, sd, interactivo=False):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre descriptivo corto de columna

    # Se obtiene la lista de variables (nombre de columnas numéricas) del df
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Se elimina la columna numérica del 'periodo'
    indicadores_cols = [c for c in numeric_cols if c != 'periodo']
    # Se busca la clave original (columna) en el diccionario a partir del valor descriptivo
    indicador = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    # Se obtiene el índice del indicador actual
    indice_actual = indicadores_cols.index(indicador)
    # Se obtiene el nombre del indicador anterior
    indice_ant = (indice_actual - 1) % len(indicadores_cols)
    nuevo_indicador = indicadores_cols[indice_ant]
    
    # Se genera el gráfico de evolución para el indicador siguiente
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador, serie, mg, tend, mm, sd, 
                                          tipo_mov=4, interactivo=interactivo)

    # Se genera el gráfico de histograma para el indicador siguiente
    fig2 = tab_EDA_create_histogram_graph(df, nuevo_indicador)

    # Se genera el gráfico de distribución normal para el indicador siguiente
    fig3 = tab_EDA_create_normal_dist_graph(df, nuevo_indicador)

    # Se renombra el nuevo indicador con el nombre corto del diccionario para
    # colocarlo en la lista desplegable
    indicador_ncorto = dict_ncortos.get(nuevo_indicador)

    return gr.update(value=indicador_ncorto), fig1, fig2, fig3


    if df is None or df.empty:
        return pd.DataFrame() # Retorna DF vacío si no hay datos
    
    if not provincia or not departamento:
        return pd.DataFrame()

    # Filtrado inicial
    res_df = df[(df['provincia'] == provincia) & (df['departamento'] == departamento)].copy()
    
    if res_df.empty:
        return pd.DataFrame(columns=df.columns) # Retorna estructura original vacía

    # Definición de dimensiones de agrupación
    group_cols = list(key_columns)
    
    # Identificación de métricas (solo numéricas)
    numeric_cols = [col for col in res_df.columns 
                    if col not in group_cols and pd.api.types.is_numeric_dtype(res_df[col])]
    
    # Agregación
    final_df = res_df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    
    # Limpieza final para Gradio
    # Gradio a veces tiene problemas con tipos de datos complejos o NaNs en la visualización
    final_df = final_df.fillna(0)
    
    # Reordenar columnas para que las dimensiones precedan a las métricas
    final_cols = group_cols + numeric_cols
    return final_df[final_cols]

def get_filtered_subset(df, cultivo, provincia, departamento, key_columns):

    if df is None or df.empty:
        return pd.DataFrame() # Retorna DF vacío si no hay datos
    
    if not provincia or not departamento:
        return pd.DataFrame()

    # Filtrado inicial
    res_df = df[(df['provincia'] == provincia) & (df['departamento'] == departamento)].copy()
    
    if res_df.empty:
        # return pd.DataFrame(columns=df.columns) # Retorna estructura original vacía
        return pd.DataFrame()

    # Definición de dimensiones de agrupación
    group_cols = list(key_columns)
    
    # Identificación de métricas (solo numéricas)
    numeric_cols = [col for col in res_df.columns 
                    if col not in group_cols and pd.api.types.is_numeric_dtype(res_df[col])]
    
    # Agregación
    final_df = res_df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    
    # Limpieza final para Gradio
    # Gradio a veces tiene problemas con tipos de datos complejos o NaNs en la visualización
    final_df = final_df.fillna(0)
    print(final_df)
    
    # Reordenar columnas para que las dimensiones precedan a las métricas
    final_cols = group_cols + numeric_cols
    return final_df[final_cols]

def tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, automatico, interactivo):

    # Se arma el listado ordenado de departamentos de la provincia
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    # if departamento is None: # Cuando se cambió la provincia pero no se seleccionó un departamento
    #    departamento = dptos_sorted[0]

    filtered = get_filtered_subset(df, provincia, departamento, KEY_COLUMNS)
    
    if filtered.empty:
        info_text = f" CULTIVO {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): SIN REGISTROS"
        return df, filtered, gr.update(value=provincia), gr.update(value=departamento), \
            gr.update(value=sector), gr.update(value=ambito), \
            info_text, gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), \
            None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=True), gr.Button(interactive=True), gr.update(visible=True), \
            gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
            gr.Checkbox(value=False), gr.Checkbox(value=False)

    # Columnas para mostrar del dataset
    all_cols = list(filtered.columns)
    cols_to_show = [c for c in all_cols if c not in ['provincia', 'departamento']]

    # Se arma el listado de las variables numéricas del dataset, excluyendo "periodo"
    # y se guarda la primera variable de la lista
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    indicadores_originales = cols_to_plot
    # Se renombran los indicadores (nombres de columnas numéricas) con los nombres cortos del diccionario
    indicadores = [dict_ncortos.get(col, col) for col in indicadores_originales]
    indicador_first = indicadores[0]

    # Data 1: Mensaje informativo sobre registros y campos
    info_text = f" CULTIVO {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): {len(filtered)} REGISTROS  -  {len(cols_to_show)} CAMPOS"
   
    # Data 2: Calcular estadísticas del dataset filtrado
    stats = (
            filtered.drop(columns=['periodo'], errors='ignore')
            .describe()
            .round(2)
            .reset_index()
            .rename(columns={'index': 'Medida'})
            )

    # Se renombran las columnas con los nombres cortos
    # El parámetro 'errors="ignore"' evita conflictos si una clave del diccionario no está en el DF
    stats = stats.rename(columns=dict_ncortos)

    # Data 3: Obtener el dataset final con las columnas para mostrar, 
    # renombrando las columnas con los nombres cortos
    final_df = filtered[cols_to_show].rename(columns=dict_ncortos)
  
    # Data 4: Generar gráfico de cajas
    fig_boxplot = tab_EDA_create_boxplot_graph(final_df, interactivo)
    
    # Data 5: Generar gráfico de serie temporal con la variable numérica indicada
    fig_evolution = tab_EDA_create_evolution_graph(filtered, indicadores_originales[0], serie=True, 
                                    med_glob=True, tend=True, med_mov=False, sd_mov=False, 
                                    tipo_mov=4, interactivo=interactivo)
    
    # Data 6: Generar gráfico de histograma con la variable numérica indicada
    fig_histogram = tab_EDA_create_histogram_graph(filtered, indicadores_originales[0])

    # Data 7: Generar gráfico de distribución normal con la variable numérica indicada
    fig_normal_dist = tab_EDA_create_normal_dist_graph(filtered, indicadores_originales[0])

    if automatico:
        mostrar_boton = False
    else:
        mostrar_boton = True

    return df, filtered, gr.update(value=provincia), gr.update(choices=dptos_sorted, value=departamento), \
            gr.update(value=sector), gr.update(value=ambito), \
            info_text, gr.update(visible=True), stats, final_df, \
            fig_boxplot, fig_evolution, fig_histogram, fig_normal_dist, \
            gr.Dropdown(choices=indicadores, value=indicador_first, interactive=True), \
            gr.Button(interactive=True), gr.Button(interactive=True), gr.update(visible=mostrar_boton), \
            gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
            gr.Checkbox(value=False), gr.Checkbox(value=False)

def tab_EDA_on_checkbox(dataset_type, provincia, departamento, sector, ambito, automatico, interactivo):
    df, provincias = load_data(dataset_type)

    if df.empty:
        m_inic = "EL DATASET DE CULTIVO SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), \
                None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True), \
                gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
                gr.Checkbox(value=False), gr.Checkbox(value=False)

    if automatico: # Se marcó la casilla "Automático", se muestra toda la data y se oculta el botón "Mostrar datos"
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True, interactivo)
    else: # Se desmarcó la casilla "Automático"; se limpia toda la data y se muestra el botón "Mostrar datos"
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, pd.DataFrame(), gr.update(value=provincia), \
            gr.update(value=departamento), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), \
            gr.update(value=m_inic), \
            gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), \
            None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True), \
            gr.Checkbox(value=True), gr.Checkbox(value=True), gr.Checkbox(value=True), \
            gr.Checkbox(value=False), gr.Checkbox(value=False)
                          

# endregion FUNCIONES PARA LA PESTAÑA "EDA"

# comienzo prg MAF
def normalize_str(s):
    if not isinstance(s, str): return ""
    s = s.upper().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def tab_Dashboard_load_geojsons():
    try:
        p_path = "Datasets/ProvinciasArgentina.geojson"
        if os.path.exists(p_path):
            with open(p_path, "r", encoding="utf-8") as f:
                prov = json.load(f)

        # Inicio MAF: Carga de GeoJSON segmentado en 5 partes
        dept = {"type": "FeatureCollection", "features": []}
        for i in range(1, 6):
            part_path = f"Datasets/DepartamentosArgentina_part{i}.geojson"
            if os.path.exists(part_path):
                with open(part_path, "r", encoding="utf-8") as f:
                    part_data = json.load(f)
                    if 'features' in part_data:
                        dept['features'].extend(part_data['features'])
        
        if not dept['features']:
            dept = None
        # Fin MAF

        return prov, dept
    except Exception as e:
        print(f"Error cargando GeoJSONs: {e}")
        return None, None

GEOJSON_PROV, GEOJSON_DEPT = tab_Dashboard_load_geojsons()

# mapeo para coordenadas
DEPT_COORDS = {}
if GEOJSON_DEPT:
    for feat in GEOJSON_DEPT['features']:
        props = feat['properties']
        p_name = normalize_str(props.get('provincia', ''))
        d_name = normalize_str(props.get('departamento', ''))
        # Obtener centroide (promedio simple de coordenadas)
        try:
            geom = feat['geometry']
            if geom['type'] == 'Polygon':
                coords = np.array(geom['coordinates'][0])
            elif geom['type'] == 'MultiPolygon':
                coords = np.array(geom['coordinates'][0][0])
            else:
                continue
            centroid = coords.mean(axis=0)
            DEPT_COORDS[(p_name, d_name)] = (centroid[1], centroid[0]) # lat, lon
        except:
            continue

def tab_Dashboard_get_summary(df, indicador, year_min, year_max):

    if df.empty:
        return "<div style='text-align: center; color: gray; padding: 10px;'>Seleccione filtros para ver indicadores</div>"
    
    ind_label = dict_ncortos.get(indicador, indicador)
    total = df[indicador].sum()
    avg = df[indicador].mean()
    
    # Provincia con mayor valor
    prov_sums = df.groupby('provincia')[indicador].sum()
    max_prov = prov_sums.idxmax() if not prov_sums.empty else "N/A"
    
    html = f"""
    <div style='display: flex; flex-direction: column; align-items: center; background: rgba(31, 119, 180, 0.05); border-radius: 12px; border: 1px solid rgba(31, 119, 180, 0.2); padding: 10px; margin-bottom: 10px;'>
        <p style='margin: 0 0 10px 0; font-size: 14px; font-weight: bold; color: #1f77b4;'>PERIODO ANALIZADO: {year_min} - {year_max}</p>
        <div style='display: flex; justify-content: space-around; width: 100%;'>
            <div style='text-align: center;'>
                <p style='margin:0; font-size: 14px; color: #666;'>TOTAL {ind_label.upper()}</p>
                <p style='margin:0; font-size: 24px; font-weight: bold; color: #1f77b4;'>{total:,.0f}</p>
            </div>
            <div style='text-align: center;'>
                <p style='margin:0; font-size: 14px; color: #666;'>RENDIMIENTO PROMEDIO</p>
                <p style='margin:0; font-size: 24px; font-weight: bold; color: #1f77b4;'>{avg:,.2f}</p>
            </div>
            <div style='text-align: center;'>
                <p style='margin:0; font-size: 14px; color: #666;'>PROVINCIA LÍDER</p>
                <p style='margin:0; font-size: 24px; font-weight: bold; color: #1f77b4;'>{max_prov}</p>
            </div>
        </div>
    </div>
    """
    return html



def get_image_base64(image_name):
    try:
        path = os.path.join("Images", image_name)
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except:
        pass
    return None

def tab_Dashboard_update_map(df, indicador, top_n, cultivo):
    if df.empty or not GEOJSON_PROV:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para mostrar", showarrow=False, font_size=20)
        fig.update_layout(template="plotly_white")
        return fig

    # Datos de coropleta: suma por provincia
    df_prov = df.groupby('provincia')[indicador].sum().reset_index()
    # Normalizar nombres de provincias para coincidencia
    df_prov['prov_norm'] = df_prov['provincia'].apply(normalize_str)
    
    # Debemos asegurar que las características de GeoJSON también tengan nombres normalizados 
    # para el mapeo de ubicaciones si es necesario.
    # Las características de GEOJSON_PROV tienen "provincia": "Formosa", etc.
    # Vamos a crear un mapeo en el geojson para que coincida con nuestros nombres normalizados.
    temp_geojson = json.loads(json.dumps(GEOJSON_PROV))
    for feat in temp_geojson['features']:
        feat['properties']['prov_match'] = normalize_str(feat['properties'].get('provincia', ''))

    fig = go.Figure()

    # Capa de Coropleta
    fig.add_trace(go.Choropleth(
        geojson=temp_geojson,
        locations=df_prov['prov_norm'],
        z=df_prov[indicador],
        featureidkey="properties.prov_match",
        colorscale="Blues",
        marker_opacity=0.6,
        marker_line_width=0.5,
        marker_line_color="white",
        name="Provincias",
        colorbar=dict(title=dict_ncortos.get(indicador, indicador), x=0)
    ))

    # Capa de puntos para Departamentos (Mapa de burbujas)
    if 'departamento' in df.columns:
        df_dept = df.groupby(['provincia', 'departamento'])[indicador].mean().reset_index()
        df_dept['prov_norm'] = df_dept['provincia'].apply(normalize_str)
        df_dept['dept_norm'] = df_dept['departamento'].apply(normalize_str)
        
        lats, lons, sizes, texts = [], [], [], []
        # Calcular total para porcentajes
        total_val = df_dept[indicador].sum() if not df_dept.empty else 1
        if total_val == 0: total_val = 1
        
        for _, row in df_dept.iterrows():
            key = (row['prov_norm'], row['dept_norm'])
            if key in DEPT_COORDS:
                coord = DEPT_COORDS[key]
                lats.append(coord[0])
                lons.append(coord[1])
                val = row[indicador]
                sizes.append(val)
                pct = (val / total_val) * 100
                texts.append(f"<b>{row['departamento']}</b><br>{row['provincia']}<br>{dict_ncortos.get(indicador, indicador)}: {val:,.2f}<br>Participación: {pct:.2f}%")
        
        if lats:
            # Escalar tamaños para mejor visibilidad
            max_val = max(sizes) if sizes else 1
            scaled_sizes = [max(5, (s / max_val) * 25) for s in sizes]
            
            fig.add_trace(go.Scattergeo(
                lat=lats,
                lon=lons,
                marker=dict(
                    size=scaled_sizes,
                    color="#1f77b4",
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=texts,
                hoverinfo="text",
                name="Departamentos"
            ))


    fig.update_geos(

        fitbounds="locations",
        visible=False,
        projection_type="mercator"
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f"Mapa de {dict_ncortos.get(indicador, indicador)}", x=0.5, font=dict(size=18, color='#1f77b4'))
    )
    
    # Actualizar etiqueta de la barra de colores si existe en la primera traza
    # Agregar imagen del cultivo en la esquina
    img_name = IMAGE_MAP.get(cultivo)
    if img_name:
        b64_img = get_image_base64(img_name)
        if b64_img:
            fig.add_layout_image(
                dict(
                    source=b64_img,
                    xref="paper", yref="paper",
                    x=0.98, y=0.98,
                    sizex=0.2, sizey=0.2,
                    xanchor="right", yanchor="top",
                    layer="above"
                )
            )

    # Agregar lista de ranking en la esquina inferior derecha (Provincia o Departamento)
    is_single_prov = df['provincia'].nunique() == 1
    rank_col = 'departamento' if is_single_prov else 'provincia'
    rank_title = "Ranking Departamentos" if is_single_prov else "Ranking Provincias"
    
    df_rank = df.groupby(rank_col)[indicador].sum().reset_index().sort_values(by=indicador, ascending=False)
    
    list_items = []
    for i, (_, row) in enumerate(df_rank.iterrows()):
        item_name = row[rank_col]
        val = row[indicador]
        # Resaltar el top N
        if i < top_n:
            list_items.append(f"● <b>{item_name}</b>: {val:,.0f}")
        else:
            list_items.append(f"○ {item_name}: {val:,.0f}")
    
    # Limitar la lista si es muy larga para evitar sobrecargar el mapa (ej. top 15)
    display_items = list_items[:15]
    if len(list_items) > 15:
        display_items.append("...")
        
    ranking_text = f"<b>{rank_title}</b><br>" + "<br>".join(display_items)
    
    fig.add_annotation(
        text=ranking_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="#1f77b4",
        borderwidth=1,
        borderpad=4
    )

    return fig


def tab_Dashboard_update_charts(df, indicador, top_n):
    if df.empty:
        return None, None
    
    ind_label = dict_ncortos.get(indicador, indicador)

    # Gráfico de Torta: Top N Provincias
    df_top = df.groupby('provincia')[indicador].sum().sort_values(ascending=False).head(top_n).reset_index()
    fig_pie = px.pie(df_top, values=indicador, names='provincia', 
                    title=f"Distribución Top {top_n} Provincias ({ind_label})",
                    color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
    
    # Gráfico de Evolución: A lo largo de los años
    df_year = df.groupby('periodo')[indicador].sum().reset_index()
    fig_evo = px.area(df_year, x='periodo', y=indicador, 
                     title=f"Evolución Histórica: {ind_label}",
                     color_discrete_sequence=['#1f77b4'])
    
    # Agregar Línea de Tendencia
    if len(df_year) > 1:
        z = np.polyfit(df_year['periodo'], df_year[indicador], 1)
        p = np.poly1d(z)
        df_year['trend'] = p(df_year['periodo'])
        
        fig_evo.add_trace(go.Scatter(
            x=df_year['periodo'], 
            y=df_year['trend'],
            mode='lines',
            name='Tendencia',
            line=dict(color='red', width=2, dash='dot')
        ))

    fig_evo.update_layout(
        margin=dict(t=40, b=20, l=20, r=20), 
        height=350, 
        template="plotly_white",
        yaxis_title=ind_label,
        xaxis_title="Año"
    )
    
    return fig_pie, fig_evo






def tab_Dashboard_on_cultivo_change(cultivo):
    df, provincias = load_data(cultivo)
    
    if df.empty:
        return gr.update(choices=["Todas"], value="Todas"), gr.update(choices=["Todos"], value="Todos"), gr.update(value=1970), gr.update(value=2024)
    
    provs = sorted([str(p) for p in df['provincia'].unique() if p])
    year_min, year_max = int(df['periodo'].min()), int(df['periodo'].max())
    
    return gr.update(choices=["Todas"] + provs, value="Todas"), \
           gr.update(choices=["Todos"], value="Todos"), \
           gr.update(minimum=year_min, maximum=year_max, value=year_min), \
           gr.update(minimum=year_min, maximum=year_max, value=year_max)





def tab_Dashboard_on_provincia_change(cultivo, provincia):
    df, _ = load_data(cultivo)
    if df.empty or provincia == "Todas":
        return gr.update(choices=["Todos"], value="Todos")
    
    depts = sorted([str(d) for d in df[df['provincia'] == provincia]['departamento'].unique() if d])
    return gr.update(choices=["Todos"] + depts, value="Todos")

def tab_Dashboard_run(cultivo, provincia, departamento, year_min, year_max, indicador, top_n):
    df, _ = load_data(cultivo)
    if df.empty:
        return "<p>No hay datos</p>", go.Figure(), None, None
    
    # Filtrado
    df_f = df[(df['periodo'] >= year_min) & (df['periodo'] <= year_max)]

    if provincia != "Todas":
        df_f = df_f[df_f['provincia'] == provincia]
        if departamento != "Todos":
            df_f = df_f[df_f['departamento'] == departamento]
    
    summary = tab_Dashboard_get_summary(df_f, indicador, year_min, year_max)
    fig_map = tab_Dashboard_update_map(df_f, indicador, top_n, cultivo)
    fig_pie, fig_evo = tab_Dashboard_update_charts(df_f, indicador, top_n)
    
    return summary, fig_map, fig_pie, fig_evo


# fin prg MAF

# region FUNCIONES PARA LA PESTAÑA "SERIES TEMPORALES"

def tab_ST_on_cult_change_OBSOLETA(dataset_type):
    df, provincias = load_data(dataset_type)

    if df.empty:
        msg = "Sin datos"
        return (pd.DataFrame(), # Dataset vacío
                ## Campos vacíos: Prov., Depto., Indicador, Tendencia
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.Plot(visible=False),
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.Plot(visible=False),
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.Plot(visible=False),
                ## Info de las series: Inicio, Final, C/Rges, S/Regs
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                ## Datasets filtrados se devuelven vacíos
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                ## Datasets diferenciados se devuelven vacíos
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                ## Sección Descomposición de las Series
                # Descripción de filtros de las tres series
                msg, msg, msg,
                # Gráficos de descomposición de las tres series
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretación de la descomposición de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección Diferenciación de las Series y Prueba ADF
                # Descripción de filtros de las tres serie
                msg, msg, msg, 
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 1
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 2
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 3
                gr.update(value=NO_EXISTE), # Variable 1 de estado para grado de diferenciación
                gr.update(value=NO_EXISTE), # Variable 2 de estado para grado de diferenciación
                gr.update(value=NO_EXISTE), # Variable 3 de estado para grado de diferenciación
                # Gráficos de diferenciación, interpretación y resultado prueba ADF
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Interpretación de la descomposición de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección ACF y PACF
                # Descripción de filtros de las tres series
                msg, msg, msg, 
                # Funciones ACF
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretaciones ACF
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Funciones PACF
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretaciones PACF
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q' para las tres series
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Descripción de filtros de las tres series
                msg, msg, msg,
                # Info de parámetros
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Gráficos de predicciones
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Tablas de predicciones
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Gráficos de residuos
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Info de residuos
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                )

    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]

    # Listado de las variables numéricas del dataset, excluyendo "periodo"
    # y se guarda la primera variable de la lista
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    indicadores_originales = cols_to_plot
    # Se renombran los indicadores (nombres de columnas numéricas) con los nombres cortos del diccionario
    indicadores = [dict_ncortos.get(col, col) for col in indicadores_originales]
    # Se guarda el nombre corto del primer indicador
    indicador_first = indicadores[0]

    # Se filtra el dataset de cultivo
    filtered = get_filtered_subset(df, prov_first, dpto_first, KEY_COLUMNS)

    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    # Al actualizar el dataset, se muestra la primera provincia, el primer departamento,
    # el primer indicador y el gráfico correspodiente.
    # Se hace para las tres series temporales
    msg = ("<b>"
           f"PROVINCIA: {prov_first.upper()}<br>"
           f"DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"INDICADOR: {dict_nlargos[indicadores_originales[0]].upper()}"
           "</b>")
    return (df, # Dataset cargado de cultivo
                ## Campos por defecto Serie 1: Prov., Depto., Indicador, Tendencia
                gr.update(choices=provincias_sorted, value=prov_first),
                gr.update(choices=dptos_sorted, value=dpto_first),
                gr.update(choices=indicadores, value=indicador_first),
                gr.Plot(visible=False),
                ## Campos por defecto Serie 2: Prov., Depto., Indicador, Tendencia
                gr.update(choices=provincias_sorted, value=prov_first),
                gr.update(choices=dptos_sorted, value=dpto_first),
                gr.update(choices=indicadores, value=indicador_first),
                gr.Plot(visible=False),
                ## Campos por defecto Serie 3: Prov., Depto., Indicador, Tendencia
                gr.update(choices=provincias_sorted, value=prov_first),
                gr.update(choices=dptos_sorted, value=dpto_first),
                gr.update(choices=indicadores, value=indicador_first),
                gr.Plot(visible=False),
                ## Info de las series (inicialmente las tres series iguales)
                inicio, final, regs, regsno,
                inicio, final, regs, regsno,
                inicio, final, regs, regsno,
                # Datasets filtrados con campos por defecto
                filtered, filtered, filtered,
                ## Datasets diferenciados se devuelven vacíos
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                ## Sección Descomposición de las Series
                # Descripción de filtros de las tres series
                msg, msg, msg,
                # Gráficos de descomposición de las tres series
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretación de la descomposición de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección Diferenciación de las Series y Prueba ADF
                # Descripción de filtros de las tres seriess
                msg, msg, msg,
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 1
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 2
                gr.update(visible=True, value=0), # Opciones de grado de dif. para Serie 3
                gr.update(value=NO_EXISTE), # Variable 1 de estado para grado de diferenciación
                gr.update(value=NO_EXISTE), # Variable 2 de estado para grado de diferenciación
                gr.update(value=NO_EXISTE), # Variable 3 de estado para grado de diferenciación
                # Gráficos de diferenciación de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Resultados de la prueba ADF de las tres seris
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección ACF y PACF
                # Descripción de filtros de las tres series
                msg, msg, msg,
                # Gráficos de ACF de las tres series
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretación de la ACF de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Gráficos de PACF de las tres series
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Interpretación de la PACF de las tres series
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q' para las tres series
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Descripción de filtros de las tres series
                msg, msg, msg,
                # Info de parámetros
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Gráficos de predicciones
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Tablas de predicciones
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                # Gráficos de residuos
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False),
                # Info de residuos
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                )

def tab_ST_on_cult_change_all(cult1, cult2, cult3):

    # Esta función se ejecuta cada vez que se entra a la pestaña "SERIES TEMPORALES";
    # las series se reinician con el primer cultivo de la lista (se indica manualmente: ARROS)
    # y con la primera provincia, la primera zona y la primera variable,
    # eso lo hace automáticamente la función tab_ST_on_cult_change()

    cult1 = "ARROZ"
    cult2 = "ARROZ"
    cult3 = "ARROZ"

    dfa, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32 = tab_ST_on_cult_change(cult1)

    dfb, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31, b32 = tab_ST_on_cult_change(cult2)

    dfc, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32 = tab_ST_on_cult_change(cult3)

    return dfa, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, \
            a11, a12, a13, a14, a15, a16, a17, a18, a19, \
            a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, \
            dfb, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
            b11, b12, b13, b14, b15, b16, b17, b18, b19, \
            b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31, b32, \
            dfc, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, \
            c11, c12, c13, c14, c15, c16, c17, c18, c19, \
            c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32

def tab_ST_on_cult_change(dataset_type):
    df, provincias = load_data(dataset_type)

    if df.empty:
        msg = "Sin datos"
        return (pd.DataFrame(), # Dataset vacío
                ## Campos vacíos: Cultivo, Prov., Depto., Indicador, Tendencia
                gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.Plot(visible=False),
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                ## Dataset filtrado se devuelvn vacío
                pd.DataFrame(),
                ## Dataset diferenciado se devuelve vacío
                pd.DataFrame(),
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(visible=False),
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                gr.update(visible=False),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )

    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]

    # Listado de las variables numéricas del dataset, excluyendo "periodo"
    # y se guarda la primera variable de la lista
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    indicadores_originales = cols_to_plot
    # Se renombran los indicadores (nombres de columnas numéricas) con los nombres cortos del diccionario
    indicadores = [dict_ncortos.get(col, col) for col in indicadores_originales]
    # Se guarda el nombre corto del primer indicador
    indicador_first = indicadores[0]

    # Se filtra el dataset de cultivo
    filtered = get_filtered_subset(df, dataset_type, prov_first, dpto_first, KEY_COLUMNS)

    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    # Al actualizar el dataset, se muestra la primera provincia, el primer departamento
    # y el primer indicador.
    msg = ("<b>"
           f"CULTIVO: {dataset_type.upper()}<br>"
           f"PROVINCIA: {prov_first.upper()}<br>"
           f"DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"INDICADOR: {dict_nlargos[indicadores_originales[0]].upper()}"
           "</b>")

    return (df, # Dataset cargado de cultivo
                ## Campos por defecto de la serie: Cultivo, Prov., Depto., Indicador, Tendencia
                gr.update(value=dataset_type),
                gr.update(choices=provincias_sorted, value=prov_first),
                gr.update(choices=dptos_sorted, value=dpto_first),
                gr.update(choices=indicadores, value=indicador_first),
                gr.Plot(visible=False),
                # Info de la serie
                inicio, final, regs, regsno,
                ## Dataset filtrado
                filtered,
                ## Dataset diferenciads se devuelve vacío
                pd.DataFrame(),
                ## Sección de Descomposición de la Serie
                # Información de filtros
                msg,
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                msg,
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                msg,
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                msg, gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )

def tab_ST_on_prov_change(df, cultivo, provincia, indicador):
    
    if not df.empty:
        # Se arma el listado ordenado de departamentos de la provincia
        # y se guarda el primer departamento de la lista
        dptos = df[df['provincia'] == provincia]['departamento'].unique()
        dptos_sorted = sorted([str(d) for d in dptos if d is not None])
        dpto_first = dptos_sorted[0]
        
        # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
        # se debe convertir a su nombre original
        ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de cultivo
    if not df.empty:
        filtered = get_filtered_subset(df, cultivo, provincia, dpto_first, KEY_COLUMNS)
    
    if df.empty or filtered.empty:
        return (pd.DataFrame(), pd.DataFrame(),
                # Lista de departamentos
                gr.update(choices=[], value=None),
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                # Gráfico de tendencia
                gr.Plot(visible=False), 
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(visible=False),
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                gr.update(visible=False),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )
    
    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    msg = ("<b>"
           f"CULTIVO: {cultivo.upper()}<br>"
           f"PROVINCIA: {provincia.upper()}<br>DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return (filtered, pd.DataFrame(),
                # Lista de departamentos
                gr.update(choices=dptos_sorted, value=dpto_first),
                # Info de la serie
                inicio, final, regs, regsno,
                # Gráfico de tendencia
                gr.Plot(visible=False), 
                ## Sección de Descomposición de la Serie
                # Información de filtros
                msg,
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                msg,
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                msg,
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                msg, gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )

def tab_ST_on_option_change(df, cultivo, provincia, departamento, indicador):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de cultivo
    if not df.empty:
        filtered = get_filtered_subset(df, cultivo, provincia, departamento, KEY_COLUMNS)
    
    if df.empty or filtered.empty:
        return (pd.DataFrame(), pd.DataFrame(),
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                # Gráfico de tendencia
                gr.Plot(visible=False), 
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(visible=False),
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                gr.update(visible=False),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )
    
    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    msg = ("<b>"
           f"CULTIVO: {cultivo.upper()}<br>"
           f"PROVINCIA: {provincia.upper()}<br>"
           f"DEPARTAMENTO: {departamento.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return (filtered, pd.DataFrame(),
                # Info de la serie
                inicio, final, regs, regsno,
                # Gráfico de tendencia
                gr.Plot(visible=False), 
                ## Sección de Descomposición de la Serie
                # Información de filtros
                msg,
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                msg,
                # Grado de diferenciación
                gr.update(value=0, visible=True),
                # Gráfico de diferenciación
                gr.Plot(visible=False),
                # Resultado prueba ADF
                gr.update(visible=False),
                # Variable de estado de grados de diferenciación
                gr.update(value=NO_EXISTE),
                ## Sección de ACF y PACF
                # Información de filtros
                msg,
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                msg, gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False)
                )

def tab_ST_on_graph_change(filtered1, filtered2, filtered3, ind1, ind2, ind3,
                         serie, mg, tend, mm, sd):
    # Función que genera tres gráficos de tendencia invocando a la
    # función tab_EDA_create_evolution_graph:
    # df1, df2, df3: datasets filtrados con nombres originales de columnas
    # ind1, ind2, ind3: indicadores actuales con nombres descriptivos cortos

    # Como loa parámetroa "ind1", "ind2", "ind3", se reciben con el nombre descriptivo corto
    # se deben convertir a su nombre original
    ind_orig1 = next((k for k, v in dict_ncortos.items() if v == ind1), ind1)
    ind_orig2 = next((k for k, v in dict_ncortos.items() if v == ind2), ind2)
    ind_orig3 = next((k for k, v in dict_ncortos.items() if v == ind3), ind3)

    mmov =  0 < mm < 4
    tipo = mm + 1

    graph1 = tab_EDA_create_evolution_graph(filtered1, ind_orig1, serie, mg, tend, mmov, sd, 
                                            tipo, interactivo=False, h_estatico=5.15)
    graph2 = tab_EDA_create_evolution_graph(filtered2, ind_orig2, serie, mg, tend, mmov, sd, 
                                            tipo, interactivo=False, h_estatico=5.15)
    graph3 = tab_EDA_create_evolution_graph(filtered3, ind_orig3, serie, mg, tend, mmov, sd, 
                                            tipo, interactivo=False, h_estatico=5.15)

    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
                graph1, graph2, graph3

def tab_ST_stl_decomp(df, indicador):
    """
    Se aplica la descomposición STL (Seasonal-Trend decomposition using LOESS).
    Este enfoque se destaca por tres ventajas clave:
    * Robustez ante anomalías: STL ignora valores atípicos mediante un ajuste robusto, 
      evitando que eventos excepcionales desvíen la tendencia real.
    * Integridad de la serie: a diferencia de las medias móviles, no existe pérdida de datos 
      en los extremos; STL calcula la tendencia y los residuos para el periodo completo 
      mediante LOESS (regresiones locales ponderadas en ventanas móviles).
    * Flexibilidad local: STL no asume un crecimiento lineal, lo que le permite adaptarse a 
      cambios de dirección en la trayectoria de los datos.
    Además, permite ajustar la ventana de suavizado para capturar variaciones cíclicas multianuales, 
    superando la limitación de la falta de estacionalidad mensual en datos anuales.

    Parámetros de la función:
    - df: dataset filtrado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    """
    
    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    df = df.sort_values('periodo').reset_index(drop=True)
    serie = df[ind_orig]
    nobs = len(serie)
    marcasx = nobs if nobs <= 10 else 10

    # Se entrena el modelo STL. En datos anuales con N=14, el 'period' suele fijarse en un ciclo 
    # hipotético o se usa period=2 (para capturar la variabilidad bianual mínima).
    # Robust indica que el modelo debe ignorar las anomalías o años atípicos.
    res = STL(serie, period=2, robust=True).fit()
    
    df['tendencia'] = res.trend
    df['estacional'] = res.seasonal
    df['residuo'] = res.resid
    
    # Gráfico interactivo
    fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=False, # Cambiado a False para mostrar años en cada fila
            vertical_spacing=0.07, # Aumentamos un poco el espacio para que no se encimen los títulos
            subplot_titles=(
                f"<b>SERIE ORIGINAL</b>", 
                f"<b>TENDENCIA (SUAVIZADO LOESS)</b>", 
                f"<b>COMPONENTE ESTACIONAL/CÍCLICO</b>", 
                f"<b>RESIDUOS (RUIDO)</b>"
            )
        )
    
    titulo = f"DESCOMPOSICIÓN STL DE: {dict_nlargos[ind_orig].upper()}"
    # Datos originales
    fig.add_trace(go.Scatter(x=df['periodo'], y=serie, name="Original", 
                             line=dict(color='black')), row=1, col=1)
    # Tendencia
    fig.add_trace(go.Scatter(x=df['periodo'], y=df['tendencia'], name="Tendencia", 
                             line=dict(color='blue')), row=2, col=1)
    # Ciclos
    fig.add_trace(go.Scatter(x=df['periodo'], y=df['estacional'], name="Estacional", 
                             line=dict(color='green')), row=3, col=1)
    # Residuos
    fig.add_trace(go.Bar(x=df['periodo'], y=df['residuo'], name="Residuos", 
                         marker_color='red'), row=4, col=1)
    
    fig.update_layout(
        height=800,
        autosize=True, # Para que el gráfico responda al contenedor
        margin=dict(l=10, r=10, t=50, b=10, pad=0),
        # title_text=f"<b>{titulo}</b>",
        # title_font=dict(size=14, color='black', family='Arial'),
        showlegend=False,
        template="plotly_white"
    )
    fig.update_annotations(font=dict(size=12, color='black', family='Arial Black'))
    fig.update_xaxes(
        tickmode='auto', # 'linear',
        # tick0=2011, 
        # dtick=10, # Salto entre etiquetas
        nticks=10, # Número máximo de etiquetas en el eje x
        tickangle=0,
        # range=[2010.5, 2024.5],
        tickfont=dict(size=10, color='black', family='Arial Black'),
        tickformat='d',
        showticklabels=True # Fuerza la visibilidad en todos los subgráficos
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color='black', family='Arial Black'),
    )

    # Métricas
    variabilidad_residuo = (df['residuo'].std() / serie.mean()) * 100
    puntos_atipicos = df[np.abs(df['residuo']) > 2 * df['residuo'].std()]
    
    # Interpretación sencilla de la tendencia
    cambio_total = res.trend.iloc[-1] - res.trend.iloc[0]
    umbral_estabilidad = 0.01  # 1% de variación como "Estabilidad"
    if abs(cambio_total) < umbral_estabilidad * res.trend.mean():
        interpretacion_tendencia = "estabilidad (sin tendencia significativa)"
    else:
        interpretacion_tendencia = "crecimiento" if cambio_total > 0 else "decrecimiento"

    # Interpretación de la Variabilidad No Explicada (Ruido)
    if variabilidad_residuo < 5:
        desc_ruido = "baja. El modelo explica casi la totalidad de los datos; los factores externos son mínimos."
    elif variabilidad_residuo < 15:
        desc_ruido = "moderada. Existen fluctuaciones aleatorias o eventos puntuales que afectan el cultivo."
    else:
        desc_ruido = "alta. La serie es altamente impredecible; hay una fuerte influencia de variables no capturadas por la tendencia."

    reporte = (f"<b>Variabilidad No Explicada (Ruido): </b>"
                f"{variabilidad_residuo:.2f}% respecto a la media. La variabilidad es {desc_ruido}<br>"
                f"<b>Tendencia: </b>"
                f"La serie presenta {interpretacion_tendencia} del cultivo.<br>"
                f"<b>Años con Desviaciones Significativas: </b>"
                f"{puntos_atipicos['periodo'].tolist() if not puntos_atipicos.empty else 'Ninguno'}<br>"
                )
    
    return fig, reporte

def tab_ST_stl_decomp_all(df1, df2, df3, var1, var2, var3):

    fig1, desc1 = tab_ST_stl_decomp(df1, var1)
    fig2, desc2 = tab_ST_stl_decomp(df2, var2)
    fig3, desc3 = tab_ST_stl_decomp(df3, var3)

    return gr.update(value = fig1, visible = True), \
            gr.update(value = desc1, visible = True), \
            gr.update(value = fig2, visible = True), \
            gr.update(value = desc2, visible = True), \
            gr.update(value = fig3, visible = True), \
            gr.update(value = desc3, visible = True)

def tab_ST_ACF(df, indicador, grado_dif):
    """
    Calcula la función de autocorrelación (ACF),
    genera un gráfico interactivo y analiza la ciclicidad.
    - df: dataset ya diferenciado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    - grado_dif: grado de diferenciación (0 = no diferenciada; 1, 2, 3, 4 = grado de dif.; NO_EXISTE = sin prueba ADF)
    IMPORTANTE: la serie debe ser estacionaria!!!
    """

    # Se verifica que el dataframe con la serie diferenciada ya exista
    if df.empty:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie aún no fue diferenciada. Debe aplicarse la diferenciación (si corresponde) y "
                    "la prueba ADF para verificar estacionariedad, para luego graficar "
                    "la FUNCIÓN DE AUTOCORRELACIÓN.</div>"
                    )
        return None, reporte, NO_EXISTE
    
    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if grado_dif == NO_EXISTE:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie debe ser estacionaria para que la función de AUTOCORRELACIÓN tenga relevancia estadística. "
                    "Debe aplicarse previamente la prueba ADF para verificar estacionariedad "
                    "y diferenciar la serie si corresponde.</div>"
                    )
        return None, reporte, NO_EXISTE
    
    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    df = df.sort_values('periodo').reset_index(drop=True)
    serie = df[ind_orig].dropna()

    n_obs = len(df)
    # lags = 6 # Desfases (lags); para 14 datos, lags=6 para mantener potencia estadística
    lags = int((n_obs / 2)) - 1
    valores_acf = acf(serie, nlags=lags)
    conf_interval = 1.96 / np.sqrt(n_obs)

    # Se verifica que los retardos sean proporcionales al número de observaciones
    # La restricción es nlags < n_obs / 2
    if lags >= int(n_obs / 2) or lags < 1:
        reporte = (f"<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    f"El tamaño de la serie ({n_obs} observaciones) es demasiado pequeño "
                    f"para calcular la FUNCIÓN DE AUTOCORRELACIÓN.</div>")
        return None, reporte, NO_EXISTE

    # Identificación del lag más significativo; se excluye lag=0 que siempre es 1
    lag_max = np.argmax(valores_acf[1:]) + 1
    val_max = valores_acf[lag_max]
    
    # Identificación del valor 'q' (Parámetro ARIMA), es el último lag donde el valor absoluto supera el IC
    lags_significativos = np.where(np.abs(valores_acf[1:]) > conf_interval)[0]
    if len(lags_significativos) > 0:
        # q es el último lag significativo (se suma 1 porque empezamos en index 0)
        q_sugerido = int(lags_significativos[-1] + 1)
    else:
        # Valor de referencia si no hay significancia
        q_sugerido = 0

    # Construcción del gráfico interactivo
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(lags + 1)),
        y=valores_acf,
        # name='Autocorrelación'
    ))
    
    # Intervalos de confianza (aproximación 95%)
    conf_interval = 1.96 / np.sqrt(n_obs)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="orange", annotation_text="IC 95%")
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="orange", annotation_text="IC 95%")
    
    fig.update_xaxes(
        # tick0=0, 
        # dtick=1, 
        tickfont=dict(size=10, color='black', family='Arial Black'),
        tickformat='d',
        tickmode='auto', # 'linear',
        nticks=10, # Número máximo de etiquetas en el eje x
        tickangle=0,
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color='black', family='Arial Black'),
    )
    fig.update_layout(
        # title=f"{dict_nlargos[ind_orig].upper()}",
        title="Función de Autocorrelación (ACF)",
        height=400,
        autosize=True, # Para que el gráfico responda al contenedor
        margin=dict(l=10, r=10, t=50, b=10, pad=0),
        xaxis_title="Desplazamiento (Años)",
        # yaxis_title="Coeficiente de Correlación",
        template="plotly_white"
    )

    # Reporte estadístico
    es_significativo = "es estadísticamente significativo" if abs(val_max) > conf_interval else "no es estadísticamente significativo"
    reporte = (
        f"<b>Coeficiente de Correlación más alto: {val_max:.4f}</b> en el desplazamiento de {lag_max} años. "
        f"Existe una probabilidad alta de que los patrones se repitan cada <b>{lag_max} años</b>. "
        f"Este valor {es_significativo} bajo un intervalo de confianza del 95%.<br>"
        f"Valor sugerido para el coeficiente <i>q</i> del modelo ARIMA: {q_sugerido}<br><br>"
        "<i>La ACF busca el máximo valor positivo para identificar la periodicidad de la serie. "
        "Un coeficiente alto indica el intervalo de tiempo en el que los patrones tienden a repetirse "
        "(estacionalidad), indicando de concordancia entre el pasado y el presente.</i><br><br><br>"
        )
    
    return fig, reporte, q_sugerido

def tab_ST_PACF(df, indicador, grado_dif):
    """
    Calcula la función de autocorrelación parcial (PACF), genera un gráfico interactivo
    y analiza la influencia directa de los retardos.
    - df: dataset ya diferenciado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    - grado_dif: grado de diferenciación (0 = no diferenciada; 1, 2, 3, 4 = grado de dif.; NO_EXISTE = sin prueba ADF)
    IMPORTANTE: la serie debe ser estacionaria!!!
    """

    # Se verifica que el dataframe con la serie diferenciada ya exista
    if df.empty:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie aún no fue diferenciada. Debe aplicarse la diferenciación (si corresponde) y "
                    "la prueba ADF para verificar estacionariedad, para luego graficar "
                    "la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL.</div>"
                    )
        return None, reporte, NO_EXISTE
    
    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if grado_dif == NO_EXISTE:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie debe ser estacionaria para que la función de AUTOCORRELACIÓN PARCIAL tenga relevancia estadística. "
                    "Debe aplicarse previamente la prueba ADF para verificar estacionariedad "
                    "y diferenciar la serie si corresponde.</div>"
                    )
        return None, reporte, NO_EXISTE
    
    # Preparación de datos (idéntica a la función anterior)
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    df = df.sort_values('periodo').reset_index(drop=True)
    serie = df[ind_orig].dropna()

    n_obs = len(serie)
    lags = max(1, (n_obs // 2) - 1) # Retardo mínimo igual a 1 para poder graficar
    
    # Se verifica que los retardos sean proporcionales al número de observaciones
    # Statsmodels requiere que nlags < n_obs // 2
    if lags >= (n_obs / 2) or n_obs < 4:
        reporte = (f"<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    f"Muestra insuficiente: {n_obs} observaciones. "
                    f"No se puede calcular la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL.</div>")
        return None, reporte, NO_EXISTE
    
    # Cálculo de PACF usando el método de Yule-Walker para muestras pequeñas
    # nlags debe ser menor a n_obs/2
    valores_pacf = pacf(serie, nlags=lags, method='ywm')
    
    # Identificación del lag más significativo (excluyendo lag 0)
    lag_max = np.argmax(np.abs(valores_pacf[1:])) + 1
    val_max = valores_pacf[lag_max]
    
    # Determinación de 'p'
    conf_interval = 1.96 / np.sqrt(n_obs)
    
    # Identificación de p: último lag significativo (fuera de las bandas de confianza)
    lags_significativos = np.where(np.abs(valores_pacf[1:]) > conf_interval)[0]
    
    if len(lags_significativos) > 0:
        # Se suma 1 porque el índice 0 del slice corresponde al lag 1
        p_sugerido = int(lags_significativos[-1] + 1)
    else:
        # Valor de referencia si no hay autocorrelación parcial significativa
        p_sugerido = 0

    # Construcción del gráfico interactivo
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(lags + 1)),
        y=valores_pacf,
        marker_color='orange',
        name='PACF'
    ))
    
    # Intervalos de confianza (Aproximación de Bartlett/Quenouille para PACF)
    conf_interval = 1.96 / np.sqrt(n_obs)
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="black", annotation_text="IC 95%")
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="black", annotation_text="IC 95%")
    
    fig.update_xaxes(
        # tick0=0, 
        # dtick=1, 
        tickfont=dict(size=10, color='black', family='Arial Black'),
        tickformat='d',
        tickmode='auto', # 'linear',
        nticks=10, # Número máximo de etiquetas en el eje x
        tickangle=0,
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color='black', family='Arial Black'),
    )
    fig.update_layout(
        height=400,
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10, pad=0),
        template="plotly_white",
        title="Función de Autocorrelación Parcial (PACF)",
        xaxis_title="Desplazamiento (Años)",
    )
    
    # Reporte estadístico para PACF
    es_significativo = "es estadísticamente significativo" if abs(val_max) > conf_interval else "no es estadísticamente significativo"
    reporte = (
        f"<b>PACF más alta: {val_max:.4f}</b> en el lag de {lag_max} año(s). "
        f"Esto indica que el valor actual tiene una relación directa con lo ocurrido hace {lag_max} año(s), "
        f"eliminando efectos intermedios. Este valor {es_significativo} (IC 95%).</b><br>"
        f"Valor sugerido para el coeficiente <i>p</i> del modelo ARIMA: {p_sugerido}<br><br>"
        "<i>La PACF utiliza el valor absoluto más alto para detectar la influencia directa "
        "de un periodo previo, eliminando efectos intermedios. Se consideran tanto valores "
        "positivos como negativos, ya que ambos definen la estructura del modelo predictivo</i><br><br>"
        "<b>Para ambas funciones, los retardos (lags) se limitan al 50% de la muestra para evitar "
        "correlaciones falsas. Solo los valores fuera del IC 95% son estructurales, el resto son ruido "
        "aleatorio. En la PACF, se aplicó el método Yule-Walker para asegurar estabilidad matemática "
        "ante la escasez de datos.</b>"
    )

    return fig, reporte, p_sugerido

def tab_ST_ACF_PACF_all(df1, df2, df3, var1, var2, var3, level1, level2, level3):

    ### IMPORTANTE: las series var1, var2 y var3 deben ser estacionarias
    ### para que la función de autocorrelación y la función de autocorrelación parcial
    ### tengan relevancia estadística.

    fig1a, desc1a, q1 = tab_ST_ACF(df1, var1, level1)
    fig2a, desc2a, q2 = tab_ST_ACF(df2, var2, level2)
    fig3a, desc3a, q3 = tab_ST_ACF(df3, var3, level3)

    fig1b, desc1b, p1 = tab_ST_PACF(df1, var1, level1)
    fig2b, desc2b, p2 = tab_ST_PACF(df2, var2, level2)
    fig3b, desc3b, p3 = tab_ST_PACF(df3, var3, level3)

    return (# ACF de la Serie 1: gráfico, interpretación, q sugerido para ARIMA
            gr.update(value = fig1a, visible = fig1a is not None),
            gr.update(value = desc1a, visible = True),
            q1,
            # ACF de la Serie 2: gráfico, interpretación, q sugerido para ARIMA
            gr.update(value = fig2a, visible = fig2a is not None),
            gr.update(value = desc2a, visible = True),
            q2,
            # ACF de la Serie 3: gráfico, interpretación, q sugerido para ARIMA
            gr.update(value = fig3a, visible = fig3a is not None),
            gr.update(value = desc3a, visible = True),
            q3,
            # PACF de la Serie 1: gráfico, interpretación, p sugerido para ARIMA
            gr.update(value = fig1b, visible = fig1b is not None),
            gr.update(value = desc1b, visible = True),
            p1,
            # PACF de la Serie 2: gráfico, interpretación, p sugerido para ARIMA
            gr.update(value = fig2b, visible = fig2b is not None),
            gr.update(value = desc2b, visible = True),
            p2,
            # PACF de la Serie 3: gráfico, interpretación, p sugerido para ARIMA
            gr.update(value = fig3b, visible = fig3b is not None),
            gr.update(value = desc3b, visible = True),
            p3
            )

def tab_ST_diff_ADF(df, indicador, grado, graficar=True, diferenciar=True, aplicar_adf=True):
    """
    Realiza la diferenciación/integración de una serie y opcionalmente aplica el test ADF.
    - df: dataset original con columnas ['periodo', indicadores_originales]
    - indicador: nombre corto del indicador
    - grado: nivel de diferenciación/integración
    - graficar: True para generar el gráfico Plotly
    - diferenciar: True para diferenciar, False para integrar
    - aplicar_adf: True para ejecutar el test de Dickey-Fuller Aumentado
    """

    # Se asume antes de la prueba que la serie no es estacionaria
    level = NO_EXISTE

    # Recuperación del nombre original mediante el diccionario global
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    # Preparación de datos
    df_result = df.sort_values('periodo').copy()
    series_pasos = []
    nombres_pasos = []
    
    # Serie inicial (Grado 0)
    current_serie = df_result[ind_orig].copy()
    series_pasos.append(current_serie.values)
    nombres_pasos.append("Serie Original")
    
    # Proceso de transformación (diferenciación o integración)
    for i in range(1, grado + 1):
        if diferenciar:
            current_serie = current_serie.diff()
            label = f"Dif. Grado {i}"
        else:
            current_serie = current_serie.cumsum()
            label = f"Int. Grado {i}"
        series_pasos.append(current_serie.values)
        nombres_pasos.append(label)
    
    # Actualización del DataFrame con el resultado final (se eliminan NaNs para el DF de salida)
    df_result[ind_orig] = current_serie

    # Generación del Gráfico
    fig = None
    if graficar:
        fig = go.Figure()
        
        for idx, serie_val in enumerate(series_pasos):
            # Se agregan todos los trazos al mismo objeto Figure
            fig.add_trace(
                go.Scatter(
                    x=df_result['periodo'], 
                    y=serie_val, 
                    mode='lines+markers', 
                    name=nombres_pasos[idx],
                    visible=True  # Todas las series visibles simultáneamente
                )
            )
        
        # Configuración estética y títulos
        op_name = "Diferenciación" if diferenciar else "Integración"
        fig.update_layout(
            # title=f"Análisis de {op_name}: {indicador} (Pasos 0 a {grado})",
            # xaxis_title="Años",
            # yaxis_title="Valor / Diferencia",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            ),
            hovermode="x unified", # Muestra todos los valores al pasar el mouse por el eje X
            height=300,
            autosize=True, # Para que el gráfico responda al contenedor
            margin=dict(l=10, r=10, t=50, b=10, pad=0),
        )
        fig.update_xaxes(
            tickfont=dict(size=10, color='black', family='Arial Black'),
            tickformat='d',
            tickmode='auto',
            nticks=10, # Número máximo de etiquetas en el eje x
            tickangle=0,
        )
        fig.update_yaxes(
            tickfont=dict(size=10, color='black', family='Arial Black'),
    )

    # Aplicación de Test ADF (sobre la serie final transformada) ---
    html_output = None
    if aplicar_adf:
        serie_test = current_serie.dropna()
        n_obs = len(serie_test)
        
        # Verificación de serie constante
        if n_obs > 1 and np.all(serie_test == serie_test.iloc[0]):
            html_output = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> Error en el Análisis:</strong><br>
                La serie temporal es constante. La varianza es 0, lo que impide calcular el estadístico ADF.
            </div>
            """
        elif n_obs > 4: # Mínimo de observaciones para una prueba con lags
            res = adfuller(serie_test, autolag='AIC')
            test_stat, p_value, lags_used, crit_values = res[0], res[1], res[2], res[4]
            
            es_estacionaria = p_value < 0.05
            color_status = "green" if es_estacionaria else "#d93025"
            conclusion_msg = "ESTACIONARIA" if es_estacionaria else "NO ESTACIONARIA"
            if es_estacionaria:
                level = grado # Serie estacionaria con grado de diferenciación (0, 1, 2, 3, 4)
            html_output = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #ffffff;">
                <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                    <tbody>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 10px; font-size: 16px !important; font-weight: bold;">Estadístico de la Prueba</td>
                            <td style="border: 1px solid #ddd; padding: 10px; text-align: center; font-size: 16px !important; font-weight: bold;">{test_stat:.4f}</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 10px; font-size: 16px !important; font-weight: bold;">P-Valor (Nivel de significancia)</td>
                            <td style="border: 1px solid #ddd; padding: 10px; text-align: center; font-size: 16px !important; font-weight: bold;">{p_value:.4f}</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 10px; font-size: 16px !important; font-weight: bold;">Rezagos utilizados (AIC)</td>
                            <td style="border: 1px solid #ddd; padding: 10px; text-align: center; font-size: 16px !important; font-weight: bold;">{lags_used}</td>
                        </tr>
                    </tbody>
                </table>
                <div style="background-color: #f1f3f4; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 10px 0; color: #5f6368;">Valores Críticos:</h4>
                    <code>1% (Confianza 99%): {crit_values['1%']:.4f}</code><br>
                    <code>5% (Confianza 95%): {crit_values['5%']:.4f}</code><br>
                    <code>10% (Confianza 90%): {crit_values['10%']:.4f}</code>
                </div>
                <div style="padding: 15px; border-left: 5px solid {color_status}; background-color: {color_status}10; margin-bottom: 20px;">
                    <strong style="color: {color_status}; font-size: 18px !important;">Conclusión: {conclusion_msg}</strong>
                    <p style="margin: 5px 0 0 0; color: #3c4043; font-size: 14px !important;">
                        A un nivel de confianza del 95%, el p-valor de {p_value:.4f} indica que 
                        {'se rechaza' if es_estacionaria else 'no se puede rechazar'} la hipótesis nula de raíz unitaria, 
                        con {grado} grados de diferenciación.
                    </p>
                </div>
            </div>
            """
        else:
            html_output = "<div>Datos insuficientes para realizar el test ADF.</div>"

    # Retorno de los tres elementos para la interfaz
    return df_result, \
            (gr.update(value=fig, visible=True) if fig else gr.update(visible=False)), \
            (gr.update(value=html_output, visible=True) if aplicar_adf else gr.update(visible=False)), \
            level

def tab_ST_diff_ADF_all(df1, df2, df3, var1, var2, var3, level1, level2, level3):
    """
    Realiza la diferenciación de las tres series y aplica el test ADF.
    - df1, df2, df3: datasets filtrados con columnas ['periodo', indicadores_originales]
    - var1, var2, var3: nombres cortos de los indicadores
    - level1, level2, level3: niveles de diferenciación/integración
    """
        
    dfdiff1, fig1, desc1, difflevel1 = tab_ST_diff_ADF(df1, var1, level1, 
                                           graficar=True, diferenciar=True, aplicar_adf=True)
    dfdiff2, fig2, desc2, difflevel2 = tab_ST_diff_ADF(df2, var2, level2, 
                                           graficar=True, diferenciar=True, aplicar_adf=True)
    dfdiff3, fig3, desc3, difflevel3 = tab_ST_diff_ADF(df3, var3, level3, 
                                           graficar=True, diferenciar=True, aplicar_adf=True)

    return dfdiff1, fig1, desc1, difflevel1, \
            dfdiff2, fig2, desc2, difflevel2, \
            dfdiff3, fig3, desc3, difflevel3

def tab_ST_on_level_change():
    
    return (gr.Plot(visible=False), # Gráfico de series diferenciadas
            gr.update(visible=False), # Estadísticos e informe de Prueba ADF
            gr.update(value=NO_EXISTE), # Variable para grado de dif. (NO_EXISTE = no estacionaria)
            gr.Plot(visible=False), # Gráfico de ACF
            gr.update(visible=False), # Informe de ACF
            gr.Plot(visible=False), # Gráfico de PACF
            gr.update(visible=False), # Informe de PACF
            gr.update(value=NO_EXISTE), # Variable 'p' de ARIMA
            gr.update(value=NO_EXISTE), # Variable 'q' de ARIMA
            gr.update(visible=False), # Informe de ARIMA
            gr.Plot(visible=False), # Gráfico de ARIMA predicciones
            gr.update(visible=False), # Tabla de ARIMA predicciones
            gr.Plot(visible=False), # Gráfico de ARIMA residuos
            gr.update(visible=False) # Informe de ARIMA residuos
    )

def tab_ST_ARIMA_all(df1, df2, df3, var1, var2, var3, p1, p2, p3, d1, d2, d3, q1, q2, q3):

    '''
    IMPORTANTE: los datasetes df1, df2 y df3 deben ser los ORIGINALES ya filtrados por provincial,
                departamento, sector y ámbito, pero SIN DIFERENCIACIÓN.
                Si la diferenciación fue necesaria para lograr ESTACIONARIEDAD, verificada con la
                prueba de ADF, queda indicada con el valor de d > 0 obtenido previamente.
    IMPORTANTE: todos los parámatros del modelo ARIMA(q, d, p) deben estar calculados:
    - d1, d2, d3: grados de diferenciación para que la prueba ADF arroje ESTACIONARIEDAD
    - q1, q2, q3: obtenidos en la función de autocorrelación ACF
    - p1, p2, p3: obtenidos en la función de autocorrelación parcial PACF
    '''

    desc1, fig1a, predic1, fig1b, resid1 = tab_ST_ARIMA(df1, var1, p1, d1, q1, n = 5)
    desc2, fig2a, predic2, fig2b, resid2 = tab_ST_ARIMA(df2, var2, p2, d2, q2, n = 5)
    desc3, fig3a, predic3, fig3b, resid3 = tab_ST_ARIMA(df3, var3, p3, d3, q3, n = 5)

    return (# ARIMA de la Serie 1: valores de (q, d, p), gráfico principal, gráficos de residuos
            gr.update(value = desc1, visible = True),
            gr.update(value = fig1a, visible = fig1a is not None),
            gr.update(value = predic1, visible = True),
            gr.update(value = fig1b, visible = fig1b is not None),
            gr.update(value = resid1, visible = True),
            # ARIMA de la Serie 2: valores de (q, d, p), gráfico principal, gráficos de residuos
            gr.update(value = desc2, visible = True),
            gr.update(value = fig2a, visible = fig2a is not None),
            gr.update(value = predic2, visible = True),
            gr.update(value = fig2b, visible = fig2b is not None),
            gr.update(value = resid2, visible = True),
            # ARIMA de la Serie 3: valores de (q, d, p), gráfico principal, gráficos de residuos
            gr.update(value = desc3, visible = True),
            gr.update(value = fig3a, visible = fig3a is not None),
            gr.update(value = predic3, visible = True),
            gr.update(value = fig3b, visible = fig3b is not None),
            gr.update(value = resid3, visible = True),
            )

def tab_ST_ARIMA(df, indicador, p, d, q, n):
   
    """
    Entrena un modelo ARIMA y genera pronósticos integrados a la escala original.
    
    df: Dataset original (no diferenciado) que contiene la serie de tiempo.
    indicador: Nombre de la variable objetivo en el DataFrame (con su nombre descriptivo corto).
    d: Orden de integración. Número de diferenciaciones necesarias para 
       alcanzar la estacionariedad, validado previamente mediante la prueba ADF.
    p: Orden del componente Autorregresivo (AR). Se obtiene analizando la PACF 
       (Autocorrelación Parcial) sobre la serie estacionaria. Se define por el 
       último rezago (lag) cuya correlación es ESTADÍSTICAMENTE SIGNIFICATIVA. 
       Signo positivo: indica persistencia; Signo negativo: indica reversión a la media.
       Se identifica como un modelo AR puro si la PACF muestra un corte abrupto 
       ('cut-off') y la ACF decae gradualmente.
    q: Orden del componente de Media Móvil (MA). Se obtiene analizando la ACF 
       (Autocorrelación Simple). Representa la relación entre el valor actual y 
       los errores de pronóstico pasados (choques). Se define por el último rezago 
       ESTADÍSTICAMENTE SIGNIFICATIVO. Signo positivo: los choques se propagan 
       en la misma dirección; Signo negativo: los choques tienden a compensarse.
       Se identifica como un modelo MA puro si la ACF muestra un corte abrupto 
       y la PACF decae gradualmente.
    n: Número de periodos futuros a predecir.

    NOTA: 
    - Modelo AR puro: ARIMA(p, d, 0)
    - Modelo MA puro: ARIMA(0, d, q)
    - Modelo MIXTO: ARIMA(p, d, q). Ocurre cuando tanto la ACF como la PACF 
      presentan decaimientos graduales o cuando ambas muestran persistencia 
      significativa que requiere ser modelada conjuntamente. si ambas funciones
      decaen gradualmente (sin un 'cut-off' claro), es más adecuado un modelo ARMA(p, q),
      porque los componentes AR y MA trabajan juntos para explicar la estructura de la serie.
    """

    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if d == NO_EXISTE:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie debe ser estacionaria para aplicar el modelo ARIMA. "
                    "Debe aplicarse la diferenciación (si corresponde) y "
                    "la prueba ADF para verificar ESTACIONARIEDAD.</div>"
                    )
        return reporte, None, None, None, None
    
    # Se verifica que el dataframe con la serie diferenciada ya exista
    if p == NO_EXISTE or q == NO_EXISTE:
        reporte = ("<div style='font-size: 16px !important; color: #FF0000 !important; font-weight: bold !important;'>"
                    "Faltan parámetros para el modelo ARIMA. El coeficiente "
                    "<i style='color: #FF0000;'>p</i> se obtiene con la FUNCIÓN DE AUTOCORRELACIÓN (ACF) y "
                    "el coeficiente <i style='color: #FF0000;'>q</i> con la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL (PACF).</div>"
                    )
        return reporte, None, None, None, None
    
    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    # Se convierte 'periodo' a formato datetime y establecerlo como índice con frecuencia,
    # porque así lo requiere el modelo ARIMA
    df = df.copy()
    # 1. Convertir a datetime
    df['periodo'] = pd.to_datetime(df['periodo'])
    df = df.sort_values('periodo').set_index('periodo')
    
    # 2. Convertir a PeriodIndex para manejar frecuencias sin día específico
    # Esto elimina el ValueWarning de statsmodels
    df.index = pd.DatetimeIndex(df.index).to_period() 
    
    serie = df[ind_orig].dropna()

    # Validación de longitud mínima de la serie (Regla general: T > p + q + 1)
    if len(serie) < (p + q + 2):
        reporte = ("<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    "La serie no tiene suficientes datos para los valores de "
                    "<i>p</i>  y <i>q</i> calculados.<br>"
                    f"<i>p</i> = {p}; <i>q</i> = {q}.</div>"
                    )
        return reporte, None, None, None, None

    # Ajuste del Modelo ARIMA (p, d, q)
    try:
        # La serie que se le pasa al modelo ARIMA es la ORIGINAL; el parámetro 'd' le indica que debe
        # diferenciarla y al hacer la predicción los valores son integrados a la escala orginal.
        # ARIMA significa: AR = auto-regresivo; I = integrado (para desdiferenciar); MA = medias móviles.
        # enforce_stationarity=False y enforce_invertibility=False ayudan a evitar los UserWarnings
        # de parámetros iniciales, permitiendo que el optimizador encuentre la solución.
        modelo = ARIMA(serie, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
        resultado = modelo.fit()
    except Exception as e:
        return f"Error al ajustar el modelo: {str(e)}", None, None, None, None

    # --- CÁLCULOS ESTADÍSTICOS ---
    n_obs = len(serie)
    k_params = p + q + 1 
    gl = n_obs - k_params
    
    aic_val = resultado.aic
    bic_val = resultado.bic
    aicc_val = aic_val + (2 * k_params**2 + 2 * k_params) / (n_obs - k_params - 1) if (n_obs - k_params - 1) > 0 else np.nan

    residuos_raw = resultado.resid
    # RMSE y MAPE
    rmse_val = np.sqrt(np.mean(residuos_raw**2))
    mape_val = np.mean(np.abs(residuos_raw / serie)) * 100

    # Test Jarque-Bera con Interpretación
    jb_test = resultado.test_normality(method='jarquebera')
    jb_p_value = jb_test[0, 1]
    jb_status = "Normal" if jb_p_value > 0.05 else "No normal"
    jb_color = "#28B463" if jb_p_value > 0.05 else "#CB4335"

    if k_params <= 2: complejidad = "Baja"
    elif k_params <= 4: complejidad = "Media"
    else: complejidad = "Alta"

    if n_obs < 15 or gl < 10: riesgo = "Muy Alto"
    elif n_obs < 25 or gl < 20: riesgo = "Alto"
    elif n_obs < 50: riesgo = "Aceptable"
    else: riesgo = "Bajo"
    
    color_riesgo = {"Bajo": "#28B463", "Aceptable": "#F1C40F", "Alto": "#E67E22", "Muy Alto": "#CB4335"}[riesgo]

    metricas_dict = {
        "Cantidad de Observaciones (n)": f"{n_obs}",
        "Grados de Libertad (GL)": f"{gl}",
        "AIC (Akaike)": f"{aic_val:.2f}",
        "AICc (Akaike Corregido)": f"{aicc_val:.2f}",
        "BIC (Bayesiano)": f"{bic_val:.2f}",
        "RMSE (Raíz Error Cuadrático Medio)": f"{rmse_val:,.2f}",
        "MAPE (Error Porcentual)": f"{mape_val:.2f}%",
        "Normalidad Errores (Jarque-Bera)": f"<b style='color: {jb_color};'>{jb_status}</b> (p={jb_p_value:.4f})",
        "Complejidad del Modelo": complejidad,
        "Nivel de Riesgo en Estimación": f"<span style='color: {color_riesgo};'>{riesgo}</span>"
    }

    # 1. Obtener la tabla de coeficientes como HTML
    html_str = resultado.summary().tables[1].as_html()

    # 2. Leer la tabla usando StringIO para evitar el FutureWarning
    try:
        df_coef = pd.read_html(StringIO(html_str), header=0, index_col=0)[0]
        df_coef_t = df_coef.transpose()
        coef_html = df_coef_t.to_html(classes='table_arima', border=0)
    except Exception as e:
        coef_html = f"<p>Error al procesar la tabla: {e}</p>"

    filas_info_html = ""
    for metrica, valor in metricas_dict.items():
        filas_info_html += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid black; background-color: #F2F4F4; font-weight: bold; text-align: left;'>{metrica}</td>
                <td style='padding: 8px; border: 1px solid black; background-color: #FFFFFF; font-weight: bold; text-align: right;'>{valor}</td>
            </tr>
        """

    reporte = f"""
        <div style='font-family: Arial; font-size: 14px; overflow-x: auto;'>
            <h4 style='color: #000000;'>Robustez y Coeficientes ARIMA: <i>p</i> = {p}, <i>d</i> = {d}, <i>q</i> = {q}</h4>
            
            <table style='width: 100%; border-collapse: collapse; margin-bottom: 20px; border: 1px solid black;'>
                <tr style='background-color: #F2F4F4;'>
                    <th style='padding: 10px; border: 1px solid black; text-align: center; color: black;'>Parámetro / Métrica</th>
                    <th style='padding: 10px; border: 1px solid black; text-align: center; color: black;'>Valor Estimado</th>
                </tr>
                {filas_info_html}
            </table>

            <style>
                .table_arima {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 10px; 
                    background-color: #FFFFFF !important;
                }}
                .table_arima th {{ 
                    background-color: #F2F4F4 !important; 
                    color: #000000 !important;
                    padding: 12px; 
                    text-align: center; 
                    border: 1px solid black; 
                    font-size: 15px;
                    font-weight: bold;
                }}
                .table_arima td {{ 
                    background-color: #FFFFFF !important;
                    color: #000000 !important;
                    padding: 10px; 
                    border: 1px solid black; 
                    text-align: right; 
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    font-weight: bold;
                }}
            </style>
            {coef_html}
        </div>
        """

    # Ejecución de la prueba de Ljung-Box sobre los residuos
    residuos = resultado.resid
    lb_test = acorr_ljungbox(residuos, lags=[10], return_df=True)
    p_value_lb = lb_test['lb_pvalue'].iloc[0]

    # Determinación del estado de ruido blanco
    es_ruido_blanco = p_value_lb > 0.05
    color_status = "#28B463" if es_ruido_blanco else "#CB4335"
    mensaje_lb = "Residuos independientes (Ruido Blanco)" if es_ruido_blanco else "Residuos Autocorrelacionados"

    reporte_lb = f"""
    <div style='font-family: Arial; font-size: 14px; margin-top: 10px; border-top: 1px solid #ddd; padding-top: 10px;'>
        <b style='color: #000000;'>Validación de Residuos (Prueba Ljung-Box):</b><br>
        Estadístico p-value: <span style='color: {color_status}; font-weight: bold;'>{p_value_lb:.4f}</span><br>
        Estado: <b style='color: {color_status};'>{mensaje_lb}</b>
        <p style='font-size: 12px; color: #566573;'>
            <i>Nota: Un p-value > 0.05 indica que los residuos son independientes, 
            lo cual es un requisito para un modelo ARIMA robusto.</i><br>
        </p>
    </div>
    """

    # Predicción de 'n' pasos futuros
    forecast_obj = resultado.get_forecast(steps=n)
    pronostico = forecast_obj.predicted_mean
    intervalos = forecast_obj.conf_int()

    anios_hist = [str(anio) for anio in range(YEAR_MIN, YEAR_MAX + 1)]
    anios_fut = [str(YEAR_MAX + i) for i in range(1, n+1)]

    """
    # Extraemos los años directamente del índice de la serie (que ya es PeriodIndex)
    # Esto garantiza que len(anios_hist) == len(serie)
    anios_hist = serie.index.astype(str).tolist()
    # Para los años futuros, partimos del último año presente en la serie
    ultimo_anio = int(anios_hist[-1])
    anios_fut = [str(ultimo_anio + i) for i in range(1, n + 1)]
    """

    df_plot_hist = pd.DataFrame({
        'Año': anios_hist, 
        'Valor': serie.values.flatten()
    })
    df_plot_pred = pd.DataFrame({
        'Año': anios_fut,
        'Valor': pronostico.values.flatten(),
        'Inf': intervalos.iloc[:, 0].values,
        'Sup': intervalos.iloc[:, 1].values
    })

    # Se agrega al df histórico el primer dato del df predicho para que no quede un salto en el gráfico
    # Se extrae la primera fila df_plot_pred con [[:1]] para mantenerlo como DataFrame
    primer_dato_pred = df_plot_pred.iloc[0:1].copy()
    # Se concatena al final de df_plot_hist (solo las columnas 'Año' y 'Valor')
    df_plot_hist = pd.concat([
        df_plot_hist, 
        primer_dato_pred[['Año', 'Valor']]
    ], ignore_index=True) # Para que no mantenga el índice 0 original y tome el índice que corresponda


    fig_prediccion = go.Figure()

    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_hist['Año'], y=df_plot_hist['Valor'],
        mode='lines+markers', name='Histórico (2011-2024)',
        line=dict(color='#1f77b4', width=3), showlegend=False
    ))

    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_pred['Año'], y=df_plot_pred['Valor'],
        mode='lines+markers', name='Predicción (2025+)',
        line=dict(color='#FF7F0E', width=3), showlegend=False
    ))

    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_pred['Año'].tolist() + df_plot_pred['Año'].tolist()[::-1],
        y=df_plot_pred['Sup'].tolist() + df_plot_pred['Inf'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
    ))

    fig_prediccion.update_xaxes(
        tickmode='linear', tick0=0, dtick=1, 
        tickfont=dict(size=12, color='black', family='Arial'), tickformat='d',
    )
    fig_prediccion.update_yaxes(tickfont=dict(size=12, color='black', family='Arial Black'))
    
    titulo = f"CULTIVO: {dict_nlargos[ind_orig].upper()}"
    fig_prediccion.update_layout(
        title={'text': f"<b>{titulo}</b>", 'font': {'size': 14, 'color': 'black'}, 'xanchor': 'left'},
        height=400, autosize=True, margin=dict(l=10, r=10, t=50, b=10, pad=0),
        xaxis=dict(type='category', tickangle=-45), template="plotly_white", hovermode="x unified"
    )

    tabla_pred_html = df_plot_pred.to_html(
        classes='table_arima', border=0, index=False, justify='center',
        formatters={'Predicción': lambda x: f"{x:,.2f}", 'Inf': lambda x: f"{x:,.2f}", 'Sup': lambda x: f"{x:,.2f}"}
    )

    tabla_pred = f"""
        <div style='font-family: Arial; font-size: 14px; overflow-x: auto; margin-top: 20px;'>
            <h4 style='color: #000000;'>Valores Pronosticados (2025 en adelante)</h4>
            <style>
                .table_arima {{ width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #FFFFFF !important; }}
                .table_arima th {{ background-color: #F2F4F4 !important; color: #000000 !important; padding: 12px; text-align: center; border: 1px solid black; font-size: 15px; font-weight: bold; }}
                .table_arima td {{ background-color: #FFFFFF !important; color: #000000 !important; padding: 10px; border: 1px solid black; text-align: right; font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }}
                .table_arima tr:hover {{ background-color: #F2F4F4 !important; }}
            </style>
            {tabla_pred_html}
        </div>
    """

    fig_residuos = make_subplots(
        rows=4, cols=1, 
        subplot_titles=('Residuos Estandarizados', 
                        'Histograma y Densidad (KDE)', 
                        'Probabilidad Normal (Q-Q Plot)', 
                        'Correlograma (ACF)'),
        vertical_spacing=0.08
    )

    # Residuos estandarizados
    fig_residuos.add_trace(go.Scatter(y=residuos, 
                                      mode='lines', 
                                      name='Residuos'), 
                                      row=1, col=1)
    fig_residuos.add_hline(y=0, line_dash="dash", 
                           line_color="black", 
                           row=1, col=1)
    
    # Histograma y Densidad
    fig_residuos.add_trace(go.Histogram(x=residuos, 
                                        nbinsx=30, 
                                        name='Hist.', 
                                        histnorm='probability density'), 
                                        row=2, col=1)
    
    # Probabilidad Normal
    qq = stats.probplot(residuos, dist="norm")
    fig_residuos.add_trace(go.Scatter(x=qq[0][0],
                                      y=qq[0][1], 
                                      mode='markers', 
                                      name='Q-Q'), 
                                      row=3, col=1)
    fig_residuos.add_trace(go.Scatter(x=qq[0][0], 
                                      y=qq[0][0]*qq[1][0] + qq[1][1], 
                                      mode='lines', 
                                      line=dict(color='red')), 
                                      row=3, col=1)

    # Correlograma
    acf_res = sm.tsa.stattools.acf(residuos, nlags=20)
    fig_residuos.add_trace(go.Bar(x=list(range(len(acf_res))), 
                                  y=acf_res, 
                                  name='ACF Residuos'), 
                                  row=4, col=1)

    # Área de graficación
    fig_residuos.update_layout(height=1000, 
                               autosize=True, 
                               margin=dict(l=20, r=20, t=40, b=10, pad=0), 
                               template="plotly_white", 
                               showlegend=False)

    # Configuración de Títulos de Subplots (Arial Black, 14pt)
    fig_residuos.update_annotations(font=dict(family="Arial Black", size=14))

    # Configuración de Ejes (Arial Black, 14pt) y Cuadrícula Negra
    fig_residuos.update_xaxes(
        tickfont=dict(family="Arial Black", size=12),
        showgrid=True, gridcolor='gray',
        linecolor='gray', mirror=True
    )
    fig_residuos.update_yaxes(
        tickfont=dict(family="Arial Black", size=12),
        showgrid=True, gridcolor='gray',
        linecolor='gray', mirror=True
    )

    return reporte, fig_prediccion, tabla_pred, fig_residuos, reporte_lb

def tab_ST_serie_complete(df, metodo):
    """
    Completa años faltantes dentro de un rango, generando los valores 
    para las columnas numéricas correspondientes, en base al método 
    de imputación indicado.
    Argumentos:
    - df: dataframe con columnas: cultivo, periodo, provincia, departamento 
          y 4 variables numéricas.
    - método: 1 - interpolación lineal;
              2 - interpolacion polinómica;
              3 - media móvil (k = 2 hacia atrás);
              4 - media móvil (k = 3 hacia atrás);
              5 - media móvila (k = 3 centrado).
    """
    print(df)

    # 1. Identificación dinámica de columnas y rango
    cols_fijas = ['cultivo', 'periodo', 'provincia', 'departamento']
    # Identificamos las variables numéricas por exclusión
    cols_num = [c for c in df.columns if c not in cols_fijas]
    
    anio_min = int(df['periodo'].min())
    anio_max = int(df['periodo'].max())
    rango_completo = np.arange(anio_min, anio_max + 1)

    # 2. Reindexación y expansión de la serie
    # Ordenamos para asegurar que la interpolación sea cronológica
    df_res = df.sort_values(['provincia', 'departamento', 'periodo'])
    
    # Creamos un nuevo DataFrame con todas las combinaciones posibles de años por grupo
    df_res = (
        df_res.set_index(['provincia', 'departamento', 'periodo'])
        .groupby(level=['provincia', 'departamento'])
        .apply(lambda x: x.reindex(pd.MultiIndex.from_product(
            [x.index.get_level_values(0).unique(), 
             x.index.get_level_values(1).unique(), 
             rango_completo],
            names=['provincia', 'departamento', 'periodo']
        )))
        .reset_index(level=[0, 1], drop=True)
        .reset_index()
    )

    # 3. Propagación de etiquetas constantes (incluye el cultivo redundante)
    df_res[['cultivo', 'provincia', 'departamento']] = (
        df_res.groupby(['provincia', 'departamento'])[['cultivo', 'provincia', 'departamento']]
        .ffill().bfill()
    )
    print(df)   
    # 4. Lógica de Imputación por grupos
    grupos = df_res.groupby(['provincia', 'departamento'])

    if metodo == 1: # Interpolación lineal
        df_res[cols_num] = grupos[cols_num].transform(lambda x: x.interpolate(method='linear'))
        
    elif metodo == 2: # Interpolación polinómica (orden 2)
        df_res[cols_num] = grupos[cols_num].transform(
            lambda x: x.interpolate(method='polynomial', order=2) if x.count() > 2 else x.interpolate(method='linear')
        )
        
    elif metodo == 3: # Media móvil k=2 (ventana 3)
        df_res[cols_num] = df_res[cols_num].fillna(grupos[cols_num].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        
    elif metodo == 4: # Media móvil k=3 (ventana 4)
        df_res[cols_num] = df_res[cols_num].fillna(grupos[cols_num].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean())
        )
        
    elif metodo == 5: # Media móvil k=3 centrada
        df_res[cols_num] = df_res[cols_num].fillna(grupos[cols_num].transform(
            lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
        )

    # 5. Extrapolación final para cubrir los extremos (bordes de la serie)
    df_res[cols_num] = grupos[cols_num].transform(lambda x: x.ffill().bfill())

    # Se obtiene información de la serie
    inicio = pd.to_numeric(df_res['periodo']).min()
    final = pd.to_numeric(df_res['periodo']).max()
    regs = df_res['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    return df_res, regs, regsno

def tab_ST_on_imputacion_change(df, imp_option):

    df_imput, num1, num2 = tab_ST_serie_complete(df, imp_option)

    return df_imput, num1, num2, \
            gr.Button(interactive=False), \
            gr.Button(interactive=True)



# endregion FUNCIONES PARA LA PESTAÑA "SERIES TEMPORALES"


# region FUNCIONES PARA IMAGENES/VIDEOS EN BASE64 E INCLUSIÓN EN CÓDIGO CSS
# Función para convertir imagen/video a Base64
def media_to_base64(media_path):
    if not os.path.exists(media_path):
        return ""
    with open(media_path, "rb") as media_file:
        return base64.b64encode(media_file.read()).decode('utf-8')

# Se lee el contenido del CSS externo para agregar
# las líneas correspondientes a las imágenes de fondo
css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        base_css = f.read()
else:
    base_css = ""
    print("Advertencia: style.css no encontrado.")

# Transformación de imágenes PNG / videos MP4 a BASE64
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path_1 = os.path.join(current_dir, "Images", "App_bg.png")
img_path_2 = os.path.join(current_dir, "Images", "Title_bg.png")
img_path_3 = os.path.join(current_dir, "Images", "Container_bg.png")
img_path_4 = os.path.join(current_dir, "Images", "header_bg.png")
img_path_5 = os.path.join(current_dir, "Images", "portrait_bg.png")
vid_path_1 = os.path.join(current_dir, "Images", "portrait_bg.mp4")
fondo_app = media_to_base64(img_path_1)
fondo_titulo = media_to_base64(img_path_2)
fondo_contenedor = media_to_base64(img_path_3)
fondo_encabezado = media_to_base64(img_path_4)
fondo_portada = media_to_base64(img_path_5)
video_portada = media_to_base64(vid_path_1)

# Se agregan al CSS leído, las líneas que aplican las imágenes de fondo en BASE64,
# que no pueden insertarse direcatmente en "style.css"
extra_css = f"""
.gradio-container {{ background-image: url('data:image/png;base64,{fondo_app}') !important; }}
.title-tab {{ background-image: url('data:image/png;base64,{fondo_titulo}') !important; }}
.custom-tab {{ background-image: url('data:image/png;base64,{fondo_contenedor}') !important; }}
.custom-tab-2 {{ background-image: url('data:image/png;base64,{fondo_contenedor}') !important; }}
.header-tab {{ background-image: url('data:image/png;base64,{fondo_encabezado}') !important; }}
.portrait-bg-1 {{ background-image: url('data:image/png;base64,{fondo_portada}') !important; }}
.portrait-bg-2 {{ background-image: url('data:image/png;base64,{fondo_titulo}') !important; }}
"""
custom_css = base_css + extra_css

# Componente HTML para el video
portada_video = f'''
<div>
    <video autoplay loop muted playsinline class="video-bg">
        <source src="data:video/mp4;base64,{video_portada}" type="video/mp4">
        El navegador no soporta la etiqueta de video.
    </video>
</div>
'''
# endregion FUNCIONES PARA IMAGENES/VIDEOS EN BASE64 E INCLUSIÓN EN CÓDIGO CSS




###### INTERFACE GRADIO ######

with gr.Blocks(title="Análisis de Cultivos") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Almacenamiento para el dataset elegido
    dataset_state = gr.State(pd.DataFrame())
    # Almacenamiento para el dataset elegido y filtrado por campos clave
    dataset_filter_state = gr.State(pd.DataFrame())
    # Almacenamiento para los tres datasets elegidos, en pestaña ST
    dataset_state_1 = gr.State(pd.DataFrame())
    dataset_state_2 = gr.State(pd.DataFrame())
    dataset_state_3 = gr.State(pd.DataFrame())
    # Almacenamiento para los tres datasets filtrados, en pestaña ST
    dataset_filter_state_1 = gr.State(pd.DataFrame())
    dataset_filter_state_2 = gr.State(pd.DataFrame())
    dataset_filter_state_3 = gr.State(pd.DataFrame())
    # Almacenamiento para los tres datasets completados con valores imputados
    dataset_imput_state_1 = gr.State(pd.DataFrame())
    dataset_imput_state_2 = gr.State(pd.DataFrame())
    dataset_imput_state_3 = gr.State(pd.DataFrame())
    # Almacenamiento para los tres datasets diferenciados, en pestaña ST
    dataset_diff_state_1 = gr.State(pd.DataFrame())
    dataset_diff_state_2 = gr.State(pd.DataFrame())
    dataset_diff_state_3 = gr.State(pd.DataFrame())
    # Para guardar el grado de diferenciación, usamos 9 para indicar que no hubo diferenciación
    # Los grados de diferenciación pueden ser: 0, 1, 2, 3, 4
    # Sirven también para el parámetro 'd' en el modelo ARIMA(p, d, q)
    level_diff_state_1 = gr.State(value=NO_EXISTE)
    level_diff_state_2 = gr.State(value=NO_EXISTE)
    level_diff_state_3 = gr.State(value=NO_EXISTE)
    # Para guardar los otros parámetros del modelo ARIMA (p, d, q)
    ARIMA_p_1 = gr.State(value=NO_EXISTE)
    ARIMA_p_2 = gr.State(value=NO_EXISTE)
    ARIMA_p_3 = gr.State(value=NO_EXISTE)
    ARIMA_q_1 = gr.State(value=NO_EXISTE)
    ARIMA_q_2 = gr.State(value=NO_EXISTE)
    ARIMA_q_3 = gr.State(value=NO_EXISTE)


    gr.Row(elem_classes="header-tab")
    

    ###### ESTRUCTURA DE PESTAÑAS
    with gr.Tabs():


        ###### PESTAÑA INICIO
        with gr.Tab("Inicio"):
            with gr.Row():
                with gr.Column(scale=8, elem_classes="portrait-bg-video"): # elem_classes="portrait-bg-1"):
                    gr.HTML(portada_video)
                    gr.HTML("ANÁLISIS <br> DE SERIES<br>TEMPORALES<br>"
                            "DE LOS<br>PRINCIPALES<br>CULTIVOS<br>"
                            "DE LA<br>REPÚBLICA ARGENTINA", elem_classes="portrait-title")
                with gr.Column(scale=2, elem_classes="portrait-bg-2"):
                    gr.HTML("Aplicación de algoritmos de Machine Learning <br>"
                            "a las Bases de Datos Abiertas <br>"
                            "de la Dirección Nacional de Agricultura<br>"
                            "del Ministerio de Agricultura, Ganadería y Pesca<br>"
                            "de la República Argentina,<br>"
                            "para el análisis de series temporales de los cultivos más importantes "
                            "en las principales regiones productoras del país.",
                            elem_classes="portrait-subtitle")
        

        ###### PESTAÑA DASHBOARD
        # comienzo prg MAF
        with gr.Tab("Dashboard") as tab_Dashboard:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;CULTIVOS DE ARGENTINA - Dashboard", elem_classes="title-text")



            
            with gr.Row():
                with gr.Column(scale=10):
                    db_summary = gr.HTML(value="<p style='text-align:center'>Cargando resumen...</p>")
                
            with gr.Row():
                with gr.Column(scale=4):
                    db_map = gr.Plot(show_label=False)
                
                with gr.Column(scale=1, elem_classes="custom-tab"):
                    db_cultivo = gr.Radio(label="Cultivo", 
                                          choices=list(FILE_MAP.keys()), 
                                          value="SOJA",
                                          elem_classes="custom-radio")
                    
                    db_provincia = gr.Dropdown(label="Provincia", 
                                               choices=["Todas"], 
                                               value="Todas",
                                               elem_classes="custom-dropdown")
                    
                    db_departamento = gr.Dropdown(label="Departamento", 
                                                  choices=["Todos"], 
                                                  value="Todos",
                                                  elem_classes="custom-dropdown")
                    
                    db_year_min = gr.Slider(label="Año Inicio", 
                                            minimum=1900, maximum=2025, 
                                            value=2000, step=1,
                                            elem_classes="custom-slider")
                    
                    db_year_max = gr.Slider(label="Año Fin", 
                                            minimum=1900, maximum=2025, 
                                            value=2024, step=1,
                                            elem_classes="custom-slider")

                    
                    db_metric = gr.Dropdown(label="Métrica a Visualizar", 
                                            choices=[("Rendimiento [Kg/Ha]", "rend_kgxha"), 
                                                     ("Producción [Tn]", "prod_ton"), 
                                                     ("Sup. Cosechada [Ha]", "sup_cos_ha"), 
                                                     ("Sup. Sembrada [Ha]", "sup_sem_ha")],
                                            value="rend_kgxha",
                                            elem_classes="custom-dropdown")
                    
                    db_top_n = gr.Slider(label="Top N Provincias", 
                                         minimum=3, maximum=10, 
                                         value=3, step=1)
                    
                    db_btn_refresh = gr.Button("📊 ACTUALIZAR PANEL", variant="primary", elem_classes="custom-button")

            with gr.Row():
                with gr.Column():
                    db_chart_pie = gr.Plot(show_label=False)
                with gr.Column():
                    db_chart_evo = gr.Plot(show_label=False)

            # Eventos
            db_cultivo.change(
                fn=tab_Dashboard_on_cultivo_change,
                inputs=[db_cultivo],
                outputs=[db_provincia, db_departamento, db_year_min, db_year_max]
            )


            
            db_provincia.change(
                fn=tab_Dashboard_on_provincia_change,
                inputs=[db_cultivo, db_provincia],
                outputs=[db_departamento]
            )
            
            db_btn_refresh.click(
                fn=tab_Dashboard_run,
                inputs=[db_cultivo, db_provincia, db_departamento, db_year_min, db_year_max, db_metric, db_top_n],
                outputs=[db_summary, db_map, db_chart_pie, db_chart_evo]
            )
            
            tab_Dashboard_row = gr.Row(visible=False) # Dummy to avoid select errors if needed, but original didn't have select
            
            tab_Dashboard.select(
                fn=tab_Dashboard_run,
                inputs=[db_cultivo, db_provincia, db_departamento, db_year_min, db_year_max, db_metric, db_top_n],
                outputs=[db_summary, db_map, db_chart_pie, db_chart_evo]
            )


        # fin prg MAF
        
        """
        ###### PESTAÑA EDA
        with gr.Tab("EDA") as tab_EDA:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;VISUALIZACIÓN DE DATOS DE LAS JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
            with gr.Row():
                with gr.Column(min_width=180, # scale=1, 
                               elem_classes="custom-tab"):
                    tipo_cultivo = gr.Radio(
                        label="Tipo de Cultivo", 
                        # choices=["Por Curso", "Por Edad", "Por Población", "Por Trayectoria"],
                        choices=["Por Curso", "Por Población", "Por Trayectoria"],
                        value="Por Curso",
                        elem_classes="custom-radio"
                    )
        
                    provincia = gr.Dropdown(label="Provincia", choices=[],
                                                elem_classes="custom-dropdown")
                    departamento = gr.Dropdown(label="Departamento", choices=[],
                                                elem_classes="custom-dropdown")
            
                    sector = gr.Radio(label="Sector", choices=["Estatal", "Privado", "Ambos"], value="Ambos", elem_classes="custom-radio")
                    ambito = gr.Radio(label="Ámbito", choices=["Urbano", "Rural", "Ambos"], value="Ambos", elem_classes="custom-radio")
                    
                    chk_mostrar = gr.Checkbox(label="Datos automáticos", value=False, elem_classes="custom-checkbox")
                    chk_interactivo = gr.Checkbox(label="Gráficos interactivos", value=False, elem_classes="custom-checkbox")
                    btn_mostrar = gr.Button("Mostrar Datos", variant="primary", visible=True, elem_classes="custom-button")
        
                with gr.Column(scale=20):
                    with gr.Row(elem_classes="custom-tab"):
                        info_label = gr.HTML(value="DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS", elem_classes="info-display-1")
                    
                    # Componente que contiene toda la información a mostrar y que alterna visible = True/False
                    with gr.Column(visible=False, elem_classes="special-panel") as data_dataset:
                        with gr.Row(elem_classes="custom-tab"):
                            with gr.Column():
                                gr.HTML(value="ESTADÍSTICAS DEL DATASET", elem_classes="info-display-2")
                                stats_table = gr.Dataframe(interactive=False, max_height=335)
                            with gr.Column():
                                gr.HTML(value="CONTENIDO DEL DATASET", elem_classes="info-display-2")
                                output_table = gr.Dataframe(interactive=False, max_height=335)
                    
                        with gr.Row(elem_classes="custom-tab"):
                            output_plot_box = gr.Plot(show_label=False)
                        
                        with gr.Row(elem_classes="custom-tab"):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column(min_width=150, scale=1):
                                        with gr.Row():
                                            indicador = gr.Dropdown(label="Indicador", choices=[],
                                                        interactive=False,
                                                        allow_custom_value=True,
                                                        elem_classes="custom-dropdown")
                                        with gr.Row(elem_classes="no-stack"):
                                            btn_anterior = gr.Button("<", variant="primary",
                                                                interactive=False,
                                                                elem_classes="custom-button2",
                                                                scale=1, min_width=50)
                                            btn_siguiente = gr.Button(">", variant="primary",
                                                                interactive=False,
                                                                elem_classes="custom-button2",
                                                                scale=1, min_width=50)
                                        with gr.Row():
                                            with gr.Column():
                                                chk_serie = gr.Checkbox(label="Serie", value=True, elem_classes="custom-checkbox-2")
                                                chk_mg = gr.Checkbox(label="Media Global", value=True, elem_classes="custom-checkbox-2")
                                                chk_tend = gr.Checkbox(label="Tendencia", value=True, elem_classes="custom-checkbox-2")
                                                chk_mm = gr.Checkbox(label="Media Móvil", value=False, elem_classes="custom-checkbox-2")
                                                chk_sd = gr.Checkbox(label="SD Móvil", value=False, elem_classes="custom-checkbox-2")
                                    with gr.Column(scale=20):
                                        output_plot_evolution = gr.Plot(show_label=False)
                                with gr.Row():
                                    output_plot_histogram = gr.Plot(show_label=False)
                                    output_plot_normal_dist = gr.Plot(show_label=False)


            tipo_cultivo.change(
                fn = tab_EDA_on_dataset_change,
                inputs = [tipo_cultivo, chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            provincia.change(
                fn = tab_EDA_on_provincia_change,
                inputs = [dataset_state, dataset_filter_state, tipo_cultivo, provincia, 
                        chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            departamento.change(
                fn = tab_EDA_on_departamento_change,
                inputs = [dataset_state, dataset_filter_state, tipo_cultivo, provincia, departamento, 
                        chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            sector.change(
                fn = tab_EDA_on_opcion_change,
                inputs = [dataset_state, dataset_filter_state, tipo_cultivo, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            ambito.change(
                fn = tab_EDA_on_opcion_change,
                inputs = [dataset_state, dataset_filter_state, tipo_cultivo, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            btn_mostrar.click(
                fn = tab_EDA_show_data,
                inputs = [dataset_state, tipo_cultivo, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            indicador.change(
                fn = tab_EDA_create_all_graphs,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution, output_plot_histogram, output_plot_normal_dist]
            )

            btn_anterior.click(
                fn = tab_EDA_create_prev_all_graphs,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )
            
            btn_siguiente.click(
                fn = tab_EDA_create_next_all_graphs,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )

            chk_serie.change(
                fn = tab_EDA_options_graph,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution]
            )

            chk_mg.change(
                fn = tab_EDA_options_graph,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution]
            )

            chk_tend.change(
                fn = tab_EDA_options_graph,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution]
            )

            chk_mm.change(
                fn = tab_EDA_options_graph,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution]
            )

            chk_sd.change(
                fn = tab_EDA_options_graph,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_evolution]
            )

            chk_mostrar.select(
                fn = tab_EDA_on_checkbox,
                inputs = [tipo_cultivo, provincia, departamento, sector, ambito, 
                        chk_mostrar, chk_interactivo],
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            chk_interactivo.select(
                fn = tab_EDA_graph_interactive,
                inputs = [dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs = [output_plot_box, output_plot_evolution]
            )

            tab_EDA.select(
                fn = tab_EDA_on_load, 
                inputs = [tipo_cultivo, chk_mostrar, chk_interactivo], 
                outputs = [dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
        """

        ###### PESTAÑA SERIES TEMPORALES
        with gr.Tab("Series Temporales") as tab_ST:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;COMPARACIÓN DE SERIES TEMPORALES", elem_classes="title-text")
            

            # region SECCIÓN 1: SELECCIÓN DE LAS TRES SERIES TEMPORALES
            ### SECCIÓN 1: SELECCIÓN DE LAS TRES SERIES TEMPORALES A COMPARAR
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;1. SELECCIÓN DE LAS SERIES TEMPORALES A COMPARAR", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    tend_button = gr.Button("Graficar", variant="primary", visible=True, 
                                               elem_classes="custom-button3")
            
            with gr.Row():
                with gr.Column():
                    with gr.Row(elem_classes="custom-tab", min_height=350):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("CULTIVO", elem_classes="info-display-4")
                                cult1 = gr.Dropdown(label="", 
                                                    choices=["ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                    "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                    value="ARROZ", elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("PROVINCIA", elem_classes="info-display-4")
                                prov1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("ZONA", elem_classes="info-display-4")
                                dep1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("INDICADOR", elem_classes="info-display-4")
                                var1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Inicio", elem_classes="info-display-4")
                                inicio1 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("Final", elem_classes="info-display-4")
                                final1 = gr.HTML(elem_classes="info-display-5")
                            with gr.Row():
                                gr.HTML("C/regs.", elem_classes="info-display-4")
                                regs1 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("S/regs.", elem_classes="info-display-4")
                                regsno1 = gr.HTML(elem_classes="info-display-5")
                            with gr.Row():
                                imp1_button = gr.Button("Imputar", variant="primary", visible=True, 
                                            elem_classes="custom-button3")
                                noimp1_button = gr.Button("No Imputar", variant="primary", visible=True, 
                                            elem_classes="custom-button3", interactive=False)
                            with gr.Row():
                                imp1_option = gr.Radio(label="Imputación",
                                                choices=["Interpolación lineal", 
                                                "Interpolación polinómica",
                                                "Media móvil (k=2)",
                                                "Media móvil (k=3 atrás)",
                                                "Meida móvil (k=3 cent.)"],
                                                value="Interpolación lineal", type="index", 
                                                show_label=False,
                                                elem_classes="custom-radio")
                        with gr.Column(scale=20):
                            with gr.Row(visible=False) as tend1_area:
                                tend1 = gr.Plot(show_label=False)

                    with gr.Row(elem_classes="custom-tab", min_height=350):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("CULTIVO", elem_classes="info-display-4")
                                cult2 = gr.Dropdown(label="", 
                                                    choices=["ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                    "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                    value="ARROZ", elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("PROVINCIA", elem_classes="info-display-4")
                                prov2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("ZONA", elem_classes="info-display-4")
                                dep2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("INDICADOR", elem_classes="info-display-4")
                                var2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Inicio", elem_classes="info-display-4")
                                inicio2 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("Final", elem_classes="info-display-4")
                                final2 = gr.HTML(elem_classes="info-display-5")
                            with gr.Row():
                                gr.HTML("C/regs.", elem_classes="info-display-4")
                                regs2 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("S/regs.", elem_classes="info-display-4")
                                regsno2 = gr.HTML(elem_classes="info-display-5")
                        with gr.Column(scale=20):
                            with gr.Row(visible=False) as tend2_area:
                                tend2 = gr.Plot(show_label=False)

                    with gr.Row(elem_classes="custom-tab", min_height=350):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("CULTIVO", elem_classes="info-display-4")
                                cult3 = gr.Dropdown(label="", 
                                                    choices=["ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                    "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                    value="ARROZ", elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("PROVINCIA", elem_classes="info-display-4")
                                prov3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("ZONA", elem_classes="info-display-4")
                                dep3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("INDICADOR", elem_classes="info-display-4")
                                var3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Inicio", elem_classes="info-display-4")
                                inicio3 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("Final", elem_classes="info-display-4")
                                final3 = gr.HTML(elem_classes="info-display-5")
                            with gr.Row():
                                gr.HTML("C/regs.", elem_classes="info-display-4")
                                regs3 = gr.HTML(elem_classes="info-display-5")
                                gr.HTML("S/regs.", elem_classes="info-display-4")
                                regsno3 = gr.HTML(elem_classes="info-display-5")
                        with gr.Column(scale=20):
                            with gr.Row(visible=False) as tend3_area:
                                tend3 = gr.Plot(show_label=False)

                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(min_width=20):
                                    gr.HTML("Opciones de gráficos:", elem_classes="title-group-2")
                                with gr.Column(min_width=45):
                                    graph_serie = gr.Checkbox(label="Serie", value=True, 
                                                        elem_classes="custom-checkbox")
                                with gr.Column(min_width=45):
                                    graph_mg = gr.Checkbox(label="Media", value=False, 
                                                    elem_classes="custom-checkbox")
                                with gr.Column(min_width=45):
                                    graph_tend = gr.Checkbox(label="Tendencia", value=True, 
                                                        elem_classes="custom-checkbox")
                                with gr.Column(scale=5):
                                    graph_mm = gr.Radio(label="Media Móvil:", choices=["No", "k = 2 atrás",
                                                            "k = 3 atrás", "k = 3 centrado"],
                                                            value="No", type="index", 
                                                            elem_classes=["custom-radio", "radio-horizontal"])
                                with gr.Column(min_width=45):
                                    graph_sd = gr.Checkbox(label="SD Móvil", value=False, 
                                                    elem_classes="custom-checkbox")
                            
            # endregion SECCIÓN 1: SELECCIÓN DE LAS TRES SERIES TEMPORALES

            # region SECCION 2: DESCOMPOSICIÓN DE LAS TRES SERIES TEMPORALES
            ### SECCIÓN 2: DESCOMPOSICIÓN DE LAS TRES SERIES TEMPORALES APLICANDO EL MÉTODO STL
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;2. DESCOMPOSICIÓN DE LAS SERIES - MÉTODO STL (SEASONAL-TREND DECOMPOSITION USING LOESS) PARA SERIES CORTAS", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    STL_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")
            
            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        STL_desc1 = gr.HTML("Descomposición de la Serie 1", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():                        
                            STL_graph1 = gr.Plot(show_label=False, visible=False)
                            STL_info1 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        STL_desc2 = gr.HTML("Descomposición de la Serie 2", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            STL_graph2 = gr.Plot(show_label=False, visible=False)
                            STL_info2 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        STL_desc3 = gr.HTML("Descomposición de la Serie 3", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            STL_graph3 = gr.Plot(show_label=False, visible=False)
                            STL_info3 = gr.HTML("Interpretación", visible=False)
            # endregion SECCION 2: DESCOMPOSICIÓN DE LAS TRES SERIES TEMPORALES

            # region SECCIÓN 3: DIFERENCIACIÓN DE LAS SERIES Y PRUEBA ADF
            ### CÁLCULO DEL TEST ADF PARA ESTACIONARIEDAD DE LAS TRES SERIES TEMPORALES
            ### APLICANDO PREVIAMENTE LOS GRADOS DE DIFERENCIACIÓN INDICADOS
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;3. DIFERENCIACIÓN DE LAS SERIES Y PRUEBA DE DICKEY-FÜLLER AUMENTADA (ADF) PARA VERIFICAR ESTACIONARIEDAD", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    ADF_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")

            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        ADF_desc1 = gr.HTML("Prueba ADF para la Serie 1", elem_classes="info-display-3")
                    with gr.Row():
                        leveldiff1 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                    with gr.Row():
                        diff_graph1 = gr.Plot(show_label=False, visible=False)
                    with gr.Row():
                        with gr.Column():                        
                            ADF_info1 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        ADF_desc2 = gr.HTML("Prueba ADF para la Serie 2", elem_classes="info-display-3")
                    with gr.Row():
                        leveldiff2 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                    with gr.Row():
                        diff_graph2 = gr.Plot(show_label=False, visible=False)                    
                    with gr.Row():
                        with gr.Column():
                            ADF_info2 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        ADF_desc3 = gr.HTML("Prueba ADF para la Serie 3", elem_classes="info-display-3")
                    with gr.Row():
                        leveldiff3 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                    with gr.Row():
                        diff_graph3 = gr.Plot(show_label=False, visible=False)
                    with gr.Row():
                        with gr.Column():
                            ADF_info3 = gr.HTML("Interpretación", visible=False)
            # endregion SECCIÓN 3: DIFERENCIACIÓN DE LAS SERIES Y PRUEBA ADF

            # region SECCIÓN 4: CÁLCULO DE AUTOCORRELACIONES DE LAS SERIES
            ### SECCIÓN 4: CÁLCULO DE AUTOCORRELACIONES DE LAS TRES SERIES TEMPORALES
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;4. FUNCIONES DE AUTOCORRELACIÓN (ACF) Y DE AUTOCORRELACIÓN PARCIAL (PACF) DE LAS SERIES (DEBEN SER ESTACIONARIAS)", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    ACF_PACF_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")

            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        ACF_desc1 = gr.HTML("Autocorrelación de la Serie 1", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():                        
                            ACF_graph1 = gr.Plot(show_label=False, visible=False)
                            ACF_info1 = gr.HTML("Interpretación ACF", visible=False)
                            PACF_graph1 = gr.Plot(show_label=False, visible=False)
                            PACF_info1 = gr.HTML("Interpretación PACF", visible=False)

                with gr.Column():
                    with gr.Row():
                        ACF_desc2 = gr.HTML("Autocorrelación de la Serie 2", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            ACF_graph2 = gr.Plot(show_label=False, visible=False)
                            ACF_info2 = gr.HTML("Interpretación ACF", visible=False)
                            PACF_graph2 = gr.Plot(show_label=False, visible=False)
                            PACF_info2 = gr.HTML("Interpretación PACF", visible=False)

                with gr.Column():
                    with gr.Row():
                        ACF_desc3 = gr.HTML("AutocorrelacióN de la Serie 3", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            ACF_graph3 = gr.Plot(show_label=False, visible=False)
                            ACF_info3 = gr.HTML("Interpretación ACF", visible=False)
                            PACF_graph3 = gr.Plot(show_label=False, visible=False)
                            PACF_info3 = gr.HTML("Interpretación PACF", visible=False)
            # endregion SECCIÓN 4: CÁLCULO DE AUTOCORRELACIONES DE LAS SERIES

            # region SECCIÓN 5: APLICACIÓN DEL MODELO ARIMA
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20): 
                    gr.HTML("&nbsp;&nbsp;5. APLICACIÓN DEL MODELO PREDICTIVO ARIMA",
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    ARIMA_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")
                    
            with gr.Row(elem_classes="custom-tab"):
                gr.HTML("El modelo ARIMA(p, d, q) es adecuado para series con tendencia clara o que requieren "
                        "diferenciación para ser estacionarias. Usa el PACF para el componente autoregresivo (p); "
                        "el grado de diferenciación para el parámetro (d) y la ACF para el componente "
                        "de media móvil (q). Si solo el primer o segundo retardo (lag) en ACF y PACF son significativos "
                        "luego de la diferenciación, el modelo más adecuado es ARIMA(p, d, q).",
                        elem_classes="info-display-2a")
           
            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        ARIMA_desc1 = gr.HTML("ARIMA de la Serie 1", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():                        
                            ARIMA_info1 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                            ARIMA_graph1 = gr.Plot(show_label=False, visible=False)
                            ARIMA_preds1 = gr.HTML("Tabla de Valores Predichos", visible=False)
                            ARIMA_graph1_resids = gr.Plot(show_label=False, visible=False)
                            ARIMA_resids1 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                with gr.Column():
                    with gr.Row():
                        ARIMA_desc2 = gr.HTML("ARIMA de la Serie 2", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            ARIMA_info2 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                            ARIMA_graph2 = gr.Plot(show_label=False, visible=False)
                            ARIMA_preds2 = gr.HTML("Tabla de Valores Predichos", visible=False)
                            ARIMA_graph2_resids = gr.Plot(show_label=False, visible=False)
                            ARIMA_resids2 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                with gr.Column():
                    with gr.Row():
                        ARIMA_desc3 = gr.HTML("ARIMA de la Serie 3", elem_classes="info-display-3")
                    with gr.Row():
                        with gr.Column():
                            ARIMA_info3 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                            ARIMA_graph3 = gr.Plot(show_label=False, visible=False)
                            ARIMA_preds3 = gr.HTML("Tabla de Valores Predichos", visible=False)
                            ARIMA_graph3_resids = gr.Plot(show_label=False, visible=False)
                            ARIMA_resids3 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)
                            

            # endregion SECCIÓN 5: APLICACIÓN DEL MODELO ARIMA



            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;6. PREDICCIÓN DE LAS SERIES", elem_classes="subtitle-text")
           
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;7. TRANSFORMADA DE FOURIER PARA LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")              




            """
            cult.change(
                fn = tab_ST_on_cult_change,
                inputs = [cult],
                outputs = [dataset_state,
                            prov1, dep1, var1, tend1_area,
                            prov2, dep2, var2, tend2_area,
                            prov3, dep3, var3, tend3_area,
                            inicio1, final1, regs1, regsno1,
                            inicio2, final2, regs2, regsno2,
                            inicio3, final3, regs3, regsno3,
                            dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            dataset_diff_state_1, dataset_diff_state_2, dataset_diff_state_3,
                            STL_desc1, STL_desc2, STL_desc3,
                            STL_graph1, STL_graph2, STL_graph3,
                            STL_info1, STL_info2, STL_info3,
                            ADF_desc1, ADF_desc2, ADF_desc3,
                            leveldiff1, leveldiff2, leveldiff3,
                            level_diff_state_1, level_diff_state_2, level_diff_state_3,
                            diff_graph1, diff_graph2, diff_graph3,
                            ADF_info1, ADF_info2, ADF_info3,
                            ACF_desc1, ACF_desc2, ACF_desc3,
                            ACF_graph1, ACF_graph2, ACF_graph3,
                            ACF_info1, ACF_info2, ACF_info3,
                            PACF_graph1, PACF_graph2, PACF_graph3,
                            PACF_info1, PACF_info2, PACF_info3,
                            ARIMA_p_1, ARIMA_p_2, ARIMA_p_3,
                            ARIMA_q_1, ARIMA_q_2, ARIMA_q_3,
                            ARIMA_desc1, ARIMA_desc2, ARIMA_desc3,
                            ARIMA_info1, ARIMA_info2, ARIMA_info3,
                            ARIMA_graph1, ARIMA_graph2, ARIMA_graph3,
                            ARIMA_preds1, ARIMA_preds2, ARIMA_preds3,
                            ARIMA_graph1_resids, ARIMA_graph2_resids, ARIMA_graph3_resids,
                            ARIMA_resids1, ARIMA_resids2, ARIMA_resids3]
            )
            """
            cult1.change(
                fn = tab_ST_on_cult_change,
                inputs = [cult1],
                outputs = [dataset_state_1,
                            cult1, prov1, dep1, var1, tend1_area,
                            inicio1, final1, regs1, regsno1,
                            dataset_filter_state_1, dataset_diff_state_1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, level_diff_state_1, diff_graph1, ADF_info1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1,
                            ARIMA_preds1, ARIMA_graph1_resids, ARIMA_resids1]
            )

            cult2.change(
                fn = tab_ST_on_cult_change,
                inputs = [cult2],
                outputs = [dataset_state_2,
                            cult2, prov2, dep2, var2, tend2_area,
                            inicio2, final2, regs2, regsno2,
                            dataset_filter_state_2, dataset_diff_state_2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, level_diff_state_2, diff_graph2, ADF_info2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2,
                            ARIMA_preds2, ARIMA_graph2_resids, ARIMA_resids2]
            )

            cult3.change(
                fn = tab_ST_on_cult_change,
                inputs = [cult3],
                outputs = [dataset_state_3,
                            cult3, prov3, dep3, var3, tend3_area,
                            inicio3, final3, regs3, regsno3,
                            dataset_filter_state_3, dataset_diff_state_3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, level_diff_state_3, diff_graph3, ADF_info3,
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3,
                            ARIMA_preds3, ARIMA_graph3_resids, ARIMA_resids3]
            )

            prov1.change(
                fn = tab_ST_on_prov_change,
                inputs = [dataset_state_1, cult1, prov1, var1],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            dep1, inicio1, final1, regs1, regsno1, tend1_area,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1]
            )

            prov2.change(
                fn = tab_ST_on_prov_change,
                inputs = [dataset_state_2, cult2, prov2, var2],
                outputs = [dataset_filter_state_2, dataset_diff_state_2,
                            dep2, inicio2, final2, regs2, regsno2, tend2_area, 
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2]
            )
            
            prov3.change(
                fn = tab_ST_on_prov_change,
                inputs = [dataset_state_3, cult3, prov3, var3],
                outputs = [dataset_filter_state_3, dataset_diff_state_3,
                            dep3, inicio3, final3, regs3, regsno3, tend3_area, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3,
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )
            
            dep1.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_1, cult1, prov1, dep1, var1],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            inicio1, final1, regs1, regsno1, tend1_area, 
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1]
            )

            dep2.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_2, cult2, prov2, dep2, var2],
                outputs = [dataset_filter_state_2, dataset_diff_state_2, 
                            inicio2, final2, regs2, regsno2, tend2_area,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2]
            )

            dep3.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_3, cult3, prov3, dep3, var3],
                outputs = [dataset_filter_state_3, dataset_diff_state_3, 
                            inicio3, final3, regs3, regsno3, tend3_area, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )

            var1.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_1, cult1, prov1, dep1, var1],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            inicio1, final1, regs1, regsno1, tend1_area, 
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1]
            )
            
            var2.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_2, cult2, prov2, dep2, var2],
                outputs = [dataset_filter_state_2, dataset_diff_state_2,
                            inicio2, final2, regs2, regsno2, tend2_area,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2]
            )

            var3.change(
                fn = tab_ST_on_option_change,
                inputs = [dataset_state_3, cult3, prov3, dep3, var3],
                outputs = [dataset_filter_state_3, dataset_diff_state_3,
                            inicio3, final3, regs3, regsno3, tend3_area, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )

            graph_serie.change(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )

            graph_mg.change(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )
            
            graph_tend.change(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )

            graph_mm.change(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )

            graph_sd.change(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )

            tab_ST.select(
                fn = tab_ST_on_cult_change_all,
                inputs = [cult1, cult2, cult3],
                outputs = [dataset_state_1,
                            cult1, prov1, dep1, var1, tend1_area,
                            inicio1, final1, regs1, regsno1,
                            dataset_filter_state_1, dataset_diff_state_1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, level_diff_state_1, 
                            diff_graph1, ADF_info1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1, ARIMA_desc1,
                            ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            dataset_state_2,
                            cult2, prov2, dep2, var2, tend2_area,
                            inicio2, final2, regs2, regsno2,
                            dataset_filter_state_2, dataset_diff_state_2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, level_diff_state_2, 
                            diff_graph2, ADF_info2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2, ARIMA_desc2,
                            ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            dataset_state_3,
                            cult3, prov3, dep3, var3, tend3_area,
                            inicio3, final3, regs3, regsno3,
                            dataset_filter_state_3, dataset_diff_state_3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, level_diff_state_3, 
                            diff_graph3, ADF_info3,
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3, ARIMA_desc3,
                            ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
                )

            leveldiff1.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph1, ADF_info1, level_diff_state_1,
                            ACF_graph1, ACF_info1, PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1]
            )

            leveldiff2.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph2, ADF_info2, level_diff_state_2,
                            ACF_graph2, ACF_info2, PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2]
            )

            leveldiff3.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph3, ADF_info3, level_diff_state_3,
                            ACF_graph3, ACF_info3, PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )

            tend_button.click(
                fn = tab_ST_on_graph_change,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [tend1_area, tend2_area, tend3_area, tend1, tend2, tend3]
            )

            imp1_button.click(
                fn = tab_ST_on_imputacion_change,
                inputs = [dataset_filter_state_1, imp1_option],
                outputs = [dataset_filter_state_1, regs1, regsno1, imp1_button, noimp1_button]
            )

            STL_button.click(
                fn = tab_ST_stl_decomp_all,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3],
                outputs = [STL_graph1, STL_info1, STL_graph2, STL_info2, STL_graph3, STL_info3]
            )

            ADF_button.click(
                fn = tab_ST_diff_ADF_all,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3, 
                            leveldiff1, leveldiff2, leveldiff3],
                outputs = [dataset_diff_state_1, diff_graph1, ADF_info1, level_diff_state_1,
                            dataset_diff_state_2, diff_graph2, ADF_info2, level_diff_state_2,
                            dataset_diff_state_3, diff_graph3, ADF_info3, level_diff_state_3]
            )

            ACF_PACF_button.click(
                fn = tab_ST_ACF_PACF_all,
                inputs = [dataset_diff_state_1, dataset_diff_state_2, dataset_diff_state_3,
                            var1, var2, var3, level_diff_state_1, level_diff_state_2, level_diff_state_3],
                outputs = [ACF_graph1, ACF_info1, ARIMA_q_1,
                            ACF_graph2, ACF_info2, ARIMA_q_2,
                            ACF_graph3, ACF_info3, ARIMA_q_3,
                            PACF_graph1, PACF_info1, ARIMA_p_1,
                            PACF_graph2, PACF_info2, ARIMA_p_2,
                            PACF_graph3, PACF_info3, ARIMA_p_3]
            )

            ARIMA_button.click(
                fn = tab_ST_ARIMA_all,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3,
                            ARIMA_p_1, ARIMA_p_2, ARIMA_p_3,
                            level_diff_state_1, level_diff_state_2, level_diff_state_3,
                            ARIMA_q_1, ARIMA_q_2, ARIMA_q_3],
                outputs = [ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )



        ###### PESTAÑA BOSQUES ALEATORIOS
        with gr.Tab("Bosques Aleatorios"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE BOSQUES ALEATORIOS", elem_classes="title-text")
                
        
        ###### PESTAÑA PROBABILIDAD BAYESIANA
        with gr.Tab("Probabilidad Bayesiana"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE SERIES TEMPORALES MEDIANTE PROBABILIDAD BAYESIANA", elem_classes="title-text")
        
        
        ###### PESTAÑA REDES NEURONALES
        with gr.Tab("Redes Neuronales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE REDES NEURONALES", elem_classes="title-text")
        
        
        ###### PESTAÑA KNN & SVM
        with gr.Tab("KNN & SVM"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS CON K-NN Y SVM", elem_classes="title-text")
            
        
        ###### PESTAÑA CONCLUSIONES
        with gr.Tab("Conclusiones"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;CONCLUSIONES", elem_classes="title-text")


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
