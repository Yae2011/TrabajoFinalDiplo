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
                          annotation_text="Media")

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
                
                # 1. Definir los límites superior e inferior
                upper_bound = y_med_mov + y_sd_mov
                lower_bound = y_med_mov - y_sd_mov
                
                # 2. Filtrar los valores NaN en ambos vectores simultáneamente
                # Esto asegura que solo se grafique donde existan datos válidos para la ventana k
                mask = upper_bound.notna() & lower_bound.notna()
                
                x_filtered = x_data[mask]
                upper_filtered = upper_bound[mask]
                lower_filtered = lower_bound[mask]
                
                # 3. Generar el trazado con los datos limpios
                fig.add_trace(go.Scatter(
                    x=pd.concat([x_filtered, x_filtered[::-1]]),
                    y=pd.concat([upper_filtered, lower_filtered[::-1]]),
                    fill='toself',
                    fillcolor='rgba(148, 103, 189, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Banda SD ({lab_m})',
                    hoverinfo='skip'
                ))

        # Ajustes estéticos de Plotly
        fig.update_layout(
            title={
                'text': f"<b>{titulo}</b>",
                'y': 0.95,        # Posición vertical (0 a 1)
                'x': 0.5,         # Posición horizontal (0.5 es centrado)
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'family': "Arial",
                    'size': 16,   # Nota: Plotly usa píxeles; 16px aprox. 12pt
                    'color': "black"
                }
            },
            # xaxis_title="Año",
            # yaxis_title="Hectáreas",
            hovermode="x unified",
                        # Dimensiones del gráfico tratando de que se aproximen al figsize=(10, 4)
            # width=1000, 
            height=h_interactivo,
            xaxis=dict(
                tickmode='auto', #linear',
                # tick0=2011, dtick=1, # range=[2010.5, 2024.5],
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
            margin=dict(l=20, r=20, t=60, b=20), 
            # Expansión y dimensiones
            autosize=True,
            template="plotly_white",
            # Ajuste de Leyenda para que no reste espacio al gráfico
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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


# region FUNCIONES PARA LA PESTAÑA "SERIES TEMPORALES"

INTERACTIVO = True
H_INTERACTIVO = 320
H_INTERACTIVO1 = 500

def tab_ST_on_cult_select(dataset_type, serie, mg, tend, mm, sd):
    df, provincias = load_data(dataset_type)

    if df.empty:
        msg = "Sin datos"
        return (pd.DataFrame(), # Dataset vacío
                pd.DataFrame(), # Dataset filtrado vacío
                pd.DataFrame(), # Dataset diferenciado vacío
                ## Campos vacíos: Cultivo, Prov., Depto., Indicador
                gr.update(choices=[], value=None), 
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), 
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"),
                gr.update(value="-"), gr.update(value="-"),
                # Botones de imputación y corte
                gr.Button(interactive=False), gr.Button(interactive=False),
                gr.update(value="Interpolación lineal"),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=False),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
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
    years = sorted(pd.to_numeric(filtered['periodo']).unique())
    years = [int(y) for y in years]
    years_inv = years[::-1]

    # Se realiza el gráfico de tendencia
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador_first), indicador_first)
    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, 
                                    tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    
    # Al actualizar el dataset, se muestra la primera provincia, el primer departamento
    # y el primer indicador.
    msg = ("<b>"
           f"CULTIVO: {dataset_type.upper()}<br>"
           f"PROVINCIA: {prov_first.upper()}<br>"
           f"DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"INDICADOR: {dict_nlargos[indicadores_originales[0]].upper()}"
           "</b>")

    return (df, # Dataset cargado del cultivo
                filtered, # Dataset filtrado del cultivo
                pd.DataFrame(), # Dataset diferenciado vacío
                ## Campos por defecto de la serie: Cultivo, Prov., Depto., Indicador
                gr.update(choices=["ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO", "SOJA",
                                    "SORGO", "TRIGO", "YERBA MATE"], value=dataset_type),
                gr.update(choices=provincias_sorted, value=prov_first),
                gr.update(choices=dptos_sorted, value=dpto_first),
                gr.update(choices=indicadores, value=indicador_first),
                # Info de la serie
                inicio, final, regs, regsno,
                # Botones de imputación y corte
                gr.Button(interactive=True), gr.Button(interactive=True),
                gr.update(value="Interpolación lineal"),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=years, value=years[0]),
                gr.update(choices=years_inv, value=years_inv[0]), 
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=True, value=graph),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                msg,
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )

def tab_ST_on_prov_select(df, cultivo, provincia, indicador,
                                            serie, mg, tend, mm, sd):
    
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
        filtered = get_filtered_subset(df, cultivo, provincia, dpto_first, KEY_COLUMNS)
    

    if df.empty or filtered.empty:
        return (pd.DataFrame(), pd.DataFrame(),
                # Lista de departamentos
                gr.update(choices=[], value=None),
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"), gr.update(value="-"), gr.update(value="-"),
                # Botones de imputación y corte
                gr.Button(interactive=False), gr.Button(interactive=False),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=False),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )
    
    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs
    years = sorted(pd.to_numeric(filtered['periodo']).unique())
    years = [int(y) for y in years]
    years_inv = years[::-1]

    # Se realiza el gráfico de tendencia
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, 
                                    tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    

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
                # Botones de imputación y corte
                gr.Button(interactive=True), gr.Button(interactive=True),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=years, value=years[0]),
                gr.update(choices=years_inv, value=years_inv[0]), 
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=True, value=graph),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                msg,
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )

def tab_ST_on_option_select(df, cultivo, provincia, departamento, indicador,
                                                            serie, mg, tend, mm, sd):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de cultivo
    if not df.empty:
        filtered = get_filtered_subset(df, cultivo, provincia, departamento, KEY_COLUMNS)
    
    if df.empty or filtered.empty:
        return (pd.DataFrame(), pd.DataFrame(),
                # Info de la serie
                gr.update(value="-"), gr.update(value="-"),
                gr.update(value="-"), gr.update(value="-"),
                # Botones de imputación y corte
                gr.Button(interactive=False), gr.Button(interactive=False),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=False),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(visible=False), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(visible=False),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )
    
    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs
    years = sorted(pd.to_numeric(filtered['periodo']).unique())
    years = [int(y) for y in years]
    years_inv = years[::-1]

    # Se realiza el gráfico de tendencia
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, 
                            tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
        
    msg = ("<b>"
           f"CULTIVO: {cultivo.upper()}<br>"
           f"PROVINCIA: {provincia.upper()}<br>"
           f"DEPARTAMENTO: {departamento.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return (filtered, pd.DataFrame(),
                # Info de la serie
                inicio, final, regs, regsno,
                # Botones de imputación y corte
                gr.Button(interactive=True), gr.Button(interactive=True),
                # Botón de restauración
                gr.Button(interactive=False),
                # Info de corte
                gr.update(choices=years, value=years[0]),
                gr.update(choices=years_inv, value=years_inv[0]), 
                None,
                # Área del gráfico de tendencia
                gr.update(),
                # Gráfico de tendencia
                gr.update(visible=True, value=graph),
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
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # msg, gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                msg,
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
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
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    graph2 = tab_EDA_create_evolution_graph(filtered2, ind_orig2, serie, mg, tend, mmov, sd, 
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    graph3 = tab_EDA_create_evolution_graph(filtered3, ind_orig3, serie, mg, tend, mmov, sd, 
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)

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
    
    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se seleccionó una serie temporal.
            </div>
            """
        return None, reporte

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

    return (gr.update(value = fig1, visible = fig1 is not None),
            gr.update(value = desc1, visible = True),
            gr.update(value = fig2, visible = fig2 is not None),
            gr.update(value = desc2, visible = True),
            gr.update(value = fig3, visible = fig3 is not None),
            gr.update(value = desc3, visible = True)
            )

def tab_ST_ACF_old(df, indicador, grado_dif):
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
    lags = int((n_obs / 2)) - 1 # Los lags son la mitad del tamaño de la muestra
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

def tab_ST_PACF_old(df, indicador, grado_dif):
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

def tab_ST_ACF(df, indicador, grado_dif):
    """
    Calcula la ACF optimizada para series anuales de cultivos,
    genera un gráfico interactivo y analiza la ciclicidad.
    - df: dataset ya diferenciado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    - grado_dif: grado de diferenciación (0 = no diferenciada; 1, 2, 3, 4 = grado de dif.; NO_EXISTE = sin prueba ADF)
    IMPORTANTE: la serie debe ser estacionaria!!!
    Nota: el valor sugerido de 'q' no puede ser mayor a 5 para no quitar muchos GL
          al modelo ARIMA. El límite 'max_horizonte' igual a 5 sirve para aplicar
          el Principio de Parsimonia para datos anuales.
    """

    # Se verifica que el dataframe con la serie diferenciada ya exista
    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                La serie aún no fue diferenciada. Debe aplicarse la diferenciación (si corresponde) y 
                la prueba ADF para verificar estacionariedad, para luego graficar
                la FUNCIÓN DE AUTOCORRELACIÓN.</b></div>
            """
        return None, reporte, NO_EXISTE
    
    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if grado_dif == NO_EXISTE:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                La serie debe ser estacionaria para que la función de AUTOCORRELACIÓN tenga relevancia estadística.
                Debe aplicarse previamente la prueba ADF para verificar estacionariedad 
                y diferenciar la serie si corresponde.</div>
            """
        return None, reporte, NO_EXISTE

    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    serie = df.sort_values('periodo')[ind_orig].dropna()
    n_obs = len(serie)
    
    # Lags a calcular: máximo 50% de la muestra, pero sugerencia limitada
    lags_a_calcular = int(n_obs / 2) - 1
    valores_acf = acf(serie, nlags=lags_a_calcular)
    conf_interval = 1.96 / np.sqrt(n_obs)
    
    # Se verifica que los retardos sean proporcionales al número de observaciones
    # La restricción es nlags < n_obs / 2
    if lags_a_calcular >= int(n_obs / 2) or lags_a_calcular < 1:
        reporte = (f"<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    f"El tamaño de la serie ({n_obs} observaciones) es demasiado pequeño "
                    f"para calcular la FUNCIÓN DE AUTOCORRELACIÓN.</div>")
        return None, reporte, NO_EXISTE

    # En agricultura anual, un proceso MA superior a 5 años es improbable.
    max_horizonte = 5 
    lags_significativos = np.where(np.abs(valores_acf[1:]) > conf_interval)[0] + 1
    
    # Se filtran solo lags dentro de un horizonte razonable (parsimonia)
    lags_validos = [l for l in lags_significativos if l <= max_horizonte]
    q_sugerido = int(lags_validos[-1]) if len(lags_validos) > 0 else 0

    # Gráfico Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(lags_a_calcular + 1)), y=valores_acf, name='ACF'))
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="orange")
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="orange")
    
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

    reporte = (
        f"<b>Valor sugerido para q (ARIMA): {q_sugerido}</b><br>"
        f"<i>Nota: Aunque se calcularon {lags_a_calcular} lags, se limita la sugerencia a los primeros {max_horizonte} "
        f"años para evitar sobreajuste en series anuales de {n_obs} datos.</i>"
    )
    
    return fig, reporte, q_sugerido

def tab_ST_PACF(df, indicador, grado_dif):
    """
    Calcula la función de autocorrelación parcial (PACF) optimizada
    para series anuales de cultivos. Utiliza el método 'ywm' para 
    estabilidad en muestras pequeñas.
    - df: dataset ya diferenciado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    - grado_dif: grado de diferenciación (0 = no diferenciada; 1, 2, 3, 4 = grado de dif.; NO_EXISTE = sin prueba ADF)
    IMPORTANTE: la serie debe ser estacionaria!!!    
    """

    # Se verifica que el dataframe con la serie diferenciada ya exista
    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                "La serie aún no fue diferenciada. Debe aplicarse la diferenciación (si corresponde) y "
                "la prueba ADF para verificar estacionariedad, para luego graficar "
                "la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL.</b></div>"
            """
        return None, reporte, NO_EXISTE
    
    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if grado_dif == NO_EXISTE:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                La serie debe ser estacionaria para que la función de AUTOCORRELACIÓN PARCIAL tenga relevancia estadística.
                Debe aplicarse previamente la prueba ADF para verificar estacionariedad 
                y diferenciar la serie si corresponde.</div>
            """
        return None, reporte, NO_EXISTE
    
    # Modificaciones similares a las aplicadas para ACF
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    serie = df.sort_values('periodo')[ind_orig].dropna()
    n_obs = len(serie)
    
    # Lags para PACF deben ser menores a n_obs/2
    lags_a_calcular = max(1, (n_obs // 2) - 1)

    # Se verifica que los retardos sean proporcionales al número de observaciones
    # Statsmodels requiere que nlags < n_obs // 2
    if lags_a_calcular >= (n_obs / 2) or n_obs < 4:
        reporte = (f"<div style='font-size: 16px !important; color: #FF0000; font-weight: bold;'>"
                    f"Muestra insuficiente: {n_obs} observaciones. "
                    f"No se puede calcular la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL.</div>")
        return None, reporte, NO_EXISTE
    
    # ywm = Yule-Walker modificado, ideal para series cortas
    valores_pacf = pacf(serie, nlags=lags_a_calcular, method='ywm')
    conf_interval = 1.96 / np.sqrt(n_obs)
    
    # En agricultura anual, un proceso AR superior a 5 años es improbable.
    max_horizonte = 5
    lags_significativos = np.where(np.abs(valores_pacf[1:]) > conf_interval)[0] + 1
    
    # Seleccionamos el último lag significativo dentro del rango lógico
    lags_validos = [l for l in lags_significativos if l <= max_horizonte]
    p_sugerido = int(lags_validos[-1]) if len(lags_validos) > 0 else 0

    # Gráfico Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(lags_a_calcular + 1)), y=valores_pacf, marker_color='orange'))
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

    reporte = (
        f"<b>Valor sugerido para p (ARIMA): {p_sugerido}</b><br>"
        f"<i>El coeficiente p indica la influencia directa de años previos. En superficies sembradas, "
        f"valores de p > 3 suelen indicar una tendencia mal corregida más que un patrón AR real.</i>"
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
    
    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se seleccionó una serie temporal.
            </div>
            """
        return (pd.DataFrame(), None, reporte, NO_EXISTE)


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
                    <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
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
    return df_result, fig, html_output, level

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

    return (dfdiff1, gr.update(value = fig1, visible = fig1 is not None),
            gr.update(value=desc1, visible=True), difflevel1,
            dfdiff2, gr.update(value = fig2, visible = fig2 is not None),
            gr.update(value=desc2, visible=True), difflevel2,
            dfdiff3, gr.update(value = fig3, visible = fig3 is not None),
            gr.update(value=desc3, visible=True), difflevel3
            )

def tab_ST_on_level_change():
    
    return (## Sección ADF
            gr.Plot(visible=False), # Gráfico de series diferenciadas
            gr.update(visible=False), # Estadísticos e informe de Prueba ADF
            gr.update(value=NO_EXISTE), # Variable para grado de dif. (NO_EXISTE = no estacionaria)
            ## Sección ACF y PACF
            gr.Plot(visible=False), # Gráfico de ACF
            gr.update(visible=False), # Informe de ACF
            gr.Plot(visible=False), # Gráfico de PACF
            gr.update(visible=False), # Informe de PACF
            gr.update(value=NO_EXISTE), # Variable 'p' de ARIMA
            gr.update(value=NO_EXISTE), # Variable 'q' de ARIMA
            ## Sección ARIMA
            gr.update(visible=False), # Informe de ARIMA
            gr.Plot(visible=False), # Gráfico de ARIMA predicciones
            gr.update(visible=False), # Tabla de ARIMA predicciones
            gr.Plot(visible=False), # Gráfico de ARIMA residuos
            gr.update(visible=False), # Informe de ARIMA residuos
            ## Sección FOURIER
            gr.Plot(visible=False), # Gráfico de series originales
            gr.Plot(visible=False), # Gráfico de series diferenciadas
            gr.Plot(visible=False), # Gráficos de Transformada de Fourier y otras
            gr.update(visible=False) # Informe de Transformada de Fourier y otras
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

    df_final1, desc1, fig1a, predic1, fig1b, resid1 = tab_ST_ARIMA(df1, var1, p1, d1, q1, n = 5)
    df_final2, desc2, fig2a, predic2, fig2b, resid2 = tab_ST_ARIMA(df2, var2, p2, d2, q2, n = 5)
    df_final3, desc3, fig3a, predic3, fig3b, resid3 = tab_ST_ARIMA(df3, var3, p3, d3, q3, n = 5)

    return (# ARIMA de la Serie 1
            df_final1,
            gr.update(value = desc1, visible = True),
            gr.update(value = fig1a, visible = fig1a is not None),
            gr.update(value = predic1, visible = True),
            gr.update(value = fig1b, visible = fig1b is not None),
            gr.update(value = resid1, visible = True),
            # ARIMA de la Serie 2
            df_final2,
            gr.update(value = desc2, visible = True),
            gr.update(value = fig2a, visible = fig2a is not None),
            gr.update(value = predic2, visible = True),
            gr.update(value = fig2b, visible = fig2b is not None),
            gr.update(value = resid2, visible = True),
            # ARIMA de la Serie 3
            df_final3,
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

    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se seleccionó una serie temporal.
            </div>
            """
        return pd.DataFrame(), reporte, None, None, None, None

    # Se verifica que la serie sea estacionaria (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if d == NO_EXISTE:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                La serie debe ser estacionaria para aplicar el modelo ARIMA.
                Debe aplicarse la diferenciación (si corresponde) y 
                la prueba ADF para verificar ESTACIONARIEDAD.</div>
            """
        return pd.DataFrame(), reporte, None, None, None, None
    
    # Se verifica que el dataframe con la serie diferenciada ya exista
    if p == NO_EXISTE or q == NO_EXISTE:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                Faltan parámetros para el modelo ARIMA. El coeficiente
                <i style='color: #FF0000;'>p</i> se obtiene con la FUNCIÓN DE AUTOCORRELACIÓN (ACF) y
                el coeficiente <i style='color: #FF0000;'>q</i> con la FUNCIÓN DE AUTOCORRELACIÓN PARCIAL (PACF).</div>
            """
        return pd.DataFrame(), reporte, None, None, None, None
    
    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    # Se convierte 'periodo' a formato datetime y establecerlo como índice con frecuencia,
    # porque así lo requiere el modelo ARIMA
    df = df.copy()
    
    # Si 'periodo' son años (int o str), forzamos la conversión a datetime especificando el formato %Y
    # Esto evita que se interpreten como milisegundos/nanosegundos (época 1970)
    df['periodo'] = pd.to_datetime(df['periodo'].astype(str), format='%Y', errors='coerce')
    
    df = df.sort_values('periodo').set_index('periodo')
    
    # Convertimos a PeriodIndex con frecuencia ANUAL ('Y') explícita
    # Esto es lo que statsmodels necesita para entender que cada salto es 1 año
    df.index = df.index.to_period('Y') 
    
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
            <h4 style='color: #000000;'>Coeficientes ARIMA: <i>p</i> = {p}, <i>d</i> = {d}, <i>q</i> = {q}</h4>
            
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
    # Los residuos se obtienen del modelo ajustado
    residuos = resultado.resid

    # Definición de rezagos a evaluar: 
    # Generalmente se evalúan varios o un valor basado en la longitud de la serie (n_obs)
    lag_val = min(10, len(residuos) // 5) 

    # Es vital pasar model_df = p + q para ajustar los grados de libertad.
    # Esto resta los parámetros del modelo ARIMA de la distribución Chi-cuadrado.
    lb_test = acorr_ljungbox(
        residuos, 
        lags=[lag_val], 
        return_df=True, 
        model_df=(p + q)
    )

    p_value_lb = lb_test['lb_pvalue'].iloc[0]


    # Determinación del estado de ruido blanco
    # H0: Los residuos están distribuidos independientemente (Ruido Blanco)
    # Ha: Los residuos presentan autocorrelación (Modelo insuficiente)
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
            lo cual es un requisito para un modelo ARIMA robusto.</i><br><br>
        </p>
    </div>
    """

    # Determinación del estado de ruido blanco
    # H0: Los residuos están distribuidos independientemente (Ruido Blanco)
    # Ha: Los residuos presentan autocorrelación (Modelo insuficiente)
    es_ruido_blanco = p_value_lb > 0.05
    color_status = "#28B463" if es_ruido_blanco else "#CB4335"
    mensaje_lb = "Residuos independientes (Ruido Blanco)" if es_ruido_blanco else "Residuos Autocorrelacionados"

    # Interpretación técnica para el reporte
    nota_tecnica = (
        "Los residuos no muestran patrones sistemáticos." 
        if es_ruido_blanco else 
        "El modelo no capturó toda la estructura; deben ajustarse p o q."
    )

    mensaje_lb = mensaje_lb + nota_tecnica


    # Predicción de 'n' pasos futuros
    forecast_obj = resultado.get_forecast(steps=n)
    pronostico = forecast_obj.predicted_mean
    intervalos = forecast_obj.conf_int()

    # extraemos los años directamente del índice de la serie filtrada y limpia.
    anios_hist = serie.index.astype(str).tolist()

    # Para los años futuros (predicción), partimos del último periodo de la serie
    ultimo_periodo = serie.index[-1]
    anios_fut = [str(ultimo_periodo + i) for i in range(1, n + 1)]

    # Ahora ambas estructuras tendrán garantizada la misma longitud
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

    # Se guarda un dataframe que tiene concatenados el histórico con el predicho
    df_final = pd.concat([df_plot_hist, df_plot_pred[['Año', 'Valor']]], axis=0).reset_index(drop=True)

    df_plot_hist = pd.concat([
        df_plot_hist, 
        primer_dato_pred[['Año', 'Valor']]
    ], ignore_index=True)


    fig_prediccion = go.Figure()

    # --- Trazado de datos históricos ---
    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_hist['Año'], y=df_plot_hist['Valor'],
        mode='lines+markers', name='Histórico',
        line=dict(color='#1f77b4', width=3), showlegend=False
    ))

    # --- Trazado de predicción ---
    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_pred['Año'], y=df_plot_pred['Valor'],
        mode='lines+markers', name='Predicción',
        line=dict(color='#FF7F0E', width=3), showlegend=False
    ))

    # --- Intervalo de confianza ---
    fig_prediccion.add_trace(go.Scatter(
        x=df_plot_pred['Año'].tolist() + df_plot_pred['Año'].tolist()[::-1],
        y=df_plot_pred['Sup'].tolist() + df_plot_pred['Inf'].tolist()[::-1],
        fill='toself', fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
    ))

    # --- Configuración de Ejes ---
    fig_prediccion.update_xaxes(
        tickmode='auto', #'linear',
        nticks=10, # Número máximo de etiquetas en el eje x
        tickangle=0,
        tickfont=dict(size=10, color='black', family='Arial Black'),
        tickformat='d',
    )

    fig_prediccion.update_yaxes(
        tickfont=dict(size=12, color='black', family='Arial Black'),
        automargin=True,
        separatethousands=False  # Elimina la coma de miles en el eje Y si se desea consistencia
    )

    # --- Configuración de Layout ---
    titulo = f"{dict_nlargos[ind_orig].upper()}"
    fig_prediccion.update_layout(
        title={'text': f"<b>{titulo}</b>", 'font': {'size': 14, 'color': 'black'}, 'x': 0.01},
        height=400,
        autosize=True,
        margin=dict(l=5, r=5, t=40, b=5), 
        template="plotly_white",
        hovermode="x unified"
    )

    # 1. Aseguramos que la columna 'Año' contenga solo el string de 4 dígitos
    # (Esto ya debería venir así del paso anterior, pero lo garantizamos)
    df_plot_pred['Año'] = df_plot_pred['Año'].astype(str).str[:4]

    # 2. Generamos el HTML con los formateadores específicos
    # Nota: Se usa "{:.2f}" para evitar la coma de miles que produce "{:,.2f}"
    tabla_pred_html = df_plot_pred.to_html(
        classes='table_arima', 
        border=0, 
        index=False, 
        justify='center',
        formatters={
            'Año': lambda x: f"{x}",
            'Valor': lambda x: f"{x:.2f}", 
            'Inf': lambda x: f"{x:.2f}", 
            'Sup': lambda x: f"{x:.2f}"
        }
    )

    # 3. Estructura final de la tabla con el estilo CSS
    tabla_pred = f"""
        <div style='font-family: Arial; font-size: 14px; overflow-x: auto; margin-top: 20px;'>
            <h4 style='color: #000000;'>Valores Pronosticados ({df_plot_pred['Año'].iloc[0]} en adelante)</h4>
            <style>
                .table_arima {{ width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #FFFFFF !important; }}
                .table_arima th {{ background-color: #F2F4F4 !important; color: #000000 !important; padding: 12px; text-align: center; border: 1px solid black; font-size: 15px; font-weight: bold; }}
                .table_arima td {{ background-color: #FFFFFF !important; color: #000000 !important; padding: 10px; border: 1px solid black; text-align: right; font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }}
                .table_arima tr:hover {{ background-color: #F2F4F4 !important; }}
            </style>
            {tabla_pred_html}<br><br>
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

    # 1. Ejecución de la comparativa
    df_comparativa = comparar_modelos_ARIMA(serie, p, d, q)

    # 2. Identificación del mejor modelo
    mejor_orden_aic = df_comparativa.iloc[0]['Orden (p,d,q)']
    es_optimo = mejor_orden_aic == str((p, d, q))

    # 3. Construcción de las filas HTML (Debe ir ANTES de armar el reporte final)
    filas_comp_html = ""
    for _, fila in df_comparativa.iterrows():
        # Resaltar en verde la fila que tenga el mejor AIC
        estilo_fila = "background-color: #D4EFDF;" if fila['Orden (p,d,q)'] == mejor_orden_aic else ""
        filas_comp_html += f"""
            <tr style='{estilo_fila}'>
                <td style='padding: 8px; border: 1px solid black; font-weight: bold;'>{fila['Orden (p,d,q)']}</td>
                <td style='padding: 8px; border: 1px solid black; text-align: right;'>{fila['AIC']:.2f}</td>
                <td style='padding: 8px; border: 1px solid black; text-align: right;'>{fila['BIC']:.2f}</td>
            </tr>
        """

    # 4. Definición del mensaje de eficiencia basado en la bandera 'es_optimo'
    if es_optimo:
        mensaje_eficiencia = (
            f"<div style='color: #000000; font-weight: bold; margin-bottom: 10px;'>"
            f"✓ El modelo actual {mejor_orden_aic} es el más eficiente según el criterio AIC.</div>"
        )
    else:
        mensaje_eficiencia = (
            f"<div style='color: #000000; font-weight: bold; margin-bottom: 10px;'>"
            f"El modelo {mejor_orden_aic} presenta un menor AIC. Deben ajustarse los parámetros.</div>"
        )

    # 5. Construcción del Reporte Final (Versión consolidada y única)
    reporte_comparativo = f"""
    <div style='margin-top: 20px; font-family: Arial; padding: 15px; border: 1px solid #D5DBDB; border-radius: 8px; background-color: #FBFCFC;'>
        <b style='color: #000000; font-size: 16px;'>Comparativa de Eficiencia (AIC y BIC)</b>
        <hr style='border: 0; border-top: 1px solid #D5DBDB; margin: 10px 0;'>
        {mensaje_eficiencia}
        <table style='width: 100%; border-collapse: collapse; border: 1px solid black; font-size: 14px;'>
            <thead style='background-color: #EBEDEF;'>
                <tr>
                    <th style='border: 1px solid black; padding: 10px;'>Modelo (p, d, q)</th>
                    <th style='border: 1px solid black; padding: 10px;'>AIC</th>
                    <th style='border: 1px solid black; padding: 10px;'>BIC</th>
                </tr>
            </thead>
            <tbody>
                {filas_comp_html}
            </tbody>
        </table>
        <p style='font-size: 12px; color: #566573; margin-top: 10px;'>
            <i>* El AIC (Akaike Information Criterion) estima la calidad relativa de los modelos; 
            el valor más bajo indica la mejor combinación de ajuste y simplicidad.</i>
        </p>
    </div>
    """

    reporte_detallado = reporte_lb + reporte_comparativo

    return df_final, reporte, fig_prediccion, tabla_pred, fig_residuos, reporte_detallado

def comparar_modelos_ARIMA(serie, p_actual, d, q_actual):
    """
    Compara el modelo actual contra modelos candidatos comunes.
    """
    candidatos = [
        (p_actual, d, q_actual), # Modelo actual
        (0, 1, 0),               # Caminata aleatoria
        (1, d, 0),               # AR(1)
        (0, d, 1),               # MA(1)
        (1, d, 1)                # Mixto simple
    ]
    
    # Eliminar duplicados si el actual coincide con un candidato
    candidatos = list(set(candidatos))
    resultados_comp = []

    for order in candidatos:
        try:
            # Ajuste rápido para comparación
            mod = ARIMA(serie, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit()
            resultados_comp.append({
                'Orden (p,d,q)': str(order),
                'AIC': res.aic,
                'BIC': res.bic,
                'Log-Likelihood': res.llf
            })
        except:
            continue

    df_comp = pd.DataFrame(resultados_comp).sort_values(by='AIC')
    return df_comp

def tab_ST_AUTO_ARIMA_all(df1, df2, df3, var1, var2, var3):

    '''
    IMPORTANTE: los datasetes df1, df2 y df3 deben ser los ORIGINALES ya filtrados por
                provincia y departamento, pero SIN DIFERENCIACIÓN.
    '''

    df_final1, desc1, fig1, predic1, info1 = tab_ST_AUTO_ARIMA(df1, var1, n = 5)
    df_final2, desc2, fig2, predic2, info2 = tab_ST_AUTO_ARIMA(df2, var2, n = 5)
    df_final3, desc3, fig3, predic3, info3 = tab_ST_AUTO_ARIMA(df3, var3, n = 5)

    return (# AUTO-ARIMA de la Serie 1
            df_final1,
            gr.update(value = desc1, visible = True),
            gr.update(value = fig1, visible = fig1 is not None),
            gr.update(value = predic1, visible = True),
            gr.update(value = info1, visible = True),
            # AUTO-ARIMA de la Serie 2
            df_final2,
            gr.update(value = desc2, visible = True),
            gr.update(value = fig2, visible = fig2 is not None),
            gr.update(value = predic2, visible = True),
            gr.update(value = info2, visible = True),
            # AUTO-ARIMA de la Serie 3
            df_final3,
            gr.update(value = desc3, visible = True),
            gr.update(value = fig3, visible = fig3 is not None),
            gr.update(value = predic3, visible = True),
            gr.update(value = info3, visible = True),
            )

def tab_ST_AUTO_ARIMA(df, indicador, n):
    """
    Identifica automáticamente el mejor modelo ARIMA, genera pronósticos 
    y reporta métricas de ajuste y pruebas de diagnóstico.
    
    df: Dataset original con columna 'periodo'.
    indicador: Nombre de la variable objetivo.
    n: Número de periodos a proyectar.
    """
    
    if df.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se seleccionó una serie temporal.
            </div>
            """
        return pd.DataFrame(), reporte, None, None, None

    # --- PREPARACIÓN DE DATOS ---
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    df_prep = df.copy()
    df_prep['periodo'] = pd.to_datetime(df_prep['periodo'].astype(str), format='%Y', errors='coerce')
    df_prep = df_prep.sort_values('periodo').set_index('periodo')
    df_prep.index = df_prep.index.to_period('Y') 
    serie = df_prep[ind_orig].dropna()

    # --- EJECUCIÓN AUTO-ARIMA ---
    # m=1 para datos anuales (no estacional), m=12 para mensuales, etc.
    # d=None permite que el test ADF/KPSS determine el orden de integración automáticamente.
    modelo_auto = pm.auto_arima(
        serie, 
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        d=None,           
        seasonal=False,   
        stepwise=True,
        suppress_warnings=True, 
        error_action='ignore'
    )
    
    p, d, q = modelo_auto.order
    residuos = modelo_auto.resid()
    
    # --- PRUEBA ESTADÍSTICA: SHAPIRO-WILK (Normalidad de Residuos) ---
    # H0: Los residuos siguen una distribución normal.
    sw_stat, sw_p = stats.shapiro(residuos)
    sw_status = "Normal" if sw_p > 0.05 else "No Normal"
    sw_color = "#28B463" if sw_p > 0.05 else "#CB4335"

    # --- TABLA DE COEFICIENTES (HTML) ---
    # Extraemos la tabla 1 del summary (coeficientes, errores, p-values)
    html_coefs = modelo_auto.summary().tables[1].as_html()
    
    # --- REPORTE DE MÉTRICAS ---
    reporte_html = f"""
    <div style='font-family: Arial; font-size: 14px; border: 1px solid #2E86C1; border-radius: 8px; padding: 15px;'>
        <h4 style='color: #2E86C1; margin-top: 0;'>Optimización Auto-ARIMA: ({p}, {d}, {q})</h4>
        <table style='width: 100%; border-collapse: collapse; margin-bottom: 15px;'>
            <tr style='background-color: #F2F4F4;'>
                <th style='border: 1px solid black; padding: 8px;'>Métrica de Selección</th>
                <th style='border: 1px solid black; padding: 8px;'>Valor</th>
            </tr>
            <tr><td style='border: 1px solid black; padding: 8px;'>AIC (Akaike)</td><td style='border: 1px solid black; padding: 8px; text-align: right;'>{modelo_auto.aic():.3f}</td></tr>
            <tr><td style='border: 1px solid black; padding: 8px;'>BIC (Bayesiano)</td><td style='border: 1px solid black; padding: 8px; text-align: right;'>{modelo_auto.bic():.3f}</td></tr>
            <tr><td style='border: 1px solid black; padding: 8px;'>Normalidad Residuos (Shapiro)</td><td style='border: 1px solid black; padding: 8px; text-align: right; color: {sw_color}; font-weight: bold;'>{sw_status} (p={sw_p:.4f})</td></tr>
        </table>
        <style>
            .table_auto_coef {{ width: 100%; border-collapse: collapse; font-family: Arial; }}
            .table_auto_coef td, .table_auto_coef th {{ border: 1px solid black; padding: 8px; text-align: center; }}
            .table_auto_coef th {{ background-color: #F2F4F4; }}
        </style>
        {html_coefs}
    </div>
    """

    # --- PRONÓSTICO ---
    pronostico, intervalos = modelo_auto.predict(n_periods=n, return_conf_int=True)
    
    anios_hist = serie.index.astype(str).tolist()
    ultimo_periodo = serie.index[-1]
    anios_fut = [str(ultimo_periodo + i) for i in range(1, n + 1)]

    df_plot_hist = pd.DataFrame({'Año': anios_hist, 'Valor': serie.values})
    df_plot_pred = pd.DataFrame({
        'Año': anios_fut,
        'Valor': pronostico,
        'Inf': intervalos[:, 0],
        'Sup': intervalos[:, 1]
    })

    # --- GRÁFICO ---
    fig_pred = go.Figure()
    # Histórico
    fig_pred.add_trace(go.Scatter(x=df_plot_hist['Año'], 
                            y=df_plot_hist['Valor'], 
                            mode='lines+markers', 
                            name='Histórico', 
                            line=dict(color='#1f77b4', width=2)))
    # Predicción
    fig_pred.add_trace(go.Scatter(x=df_plot_pred['Año'], 
                            y=df_plot_pred['Valor'], 
                            mode='lines+markers', 
                            name='Auto-ARIMA', 
                            line=dict(color='#D35400', width=3)))
    # IC
    fig_pred.add_trace(go.Scatter(
        x=df_plot_pred['Año'].tolist() + df_plot_pred['Año'].tolist()[::-1],
        y=df_plot_pred['Sup'].tolist() + df_plot_pred['Inf'].tolist()[::-1],
        fill='toself', fillcolor='rgba(211, 84, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='IC 95%'
    ))
    fig_pred.update_layout(template="plotly_white", height=450, title=f"<b>PROYECCIÓN AUTO-ARIMA: {indicador.upper()}</b>", margin=dict(l=10, r=10, t=50, b=10))

    # --- TABLA DE VALORES PREDICHOS ---
    # Formateadores para las columnas numéricas (con 2 decimales)
    dict_formatters = {'Valor': lambda x: f"{x:.2f}", 'Inf': lambda x: f"{x:.2f}", 'Sup': lambda x: f"{x:.2f}"}
    tabla_pred_html = f"""
    <div style='margin-top: 20px;'>
        <h4 style='color: #000;'>Valores Pronosticados</h4>
        <style>
            .table_res {{ width: 100%; border-collapse: collapse; border: 1px solid black; }}
            .table_res th {{ background-color: #F2F4F4; border: 1px solid black; padding: 10px; }}
            .table_res td {{ border: 1px solid black; padding: 8px; text-align: right; font-weight: bold; }}
        </style>
        {df_plot_pred.to_html(classes='table_res', index=False, formatters=dict_formatters)}
    </div>
    """

    # --- VALIDACIÓN DE RESIDUOS (LJUNG-BOX) ---
    lb_test = acorr_ljungbox(residuos, lags=[min(10, len(residuos)//5)], model_df=(p + q))
    p_lb = lb_test['lb_pvalue'].iloc[0]
    status_lb = "Independientes (Ruido Blanco)" if p_lb > 0.05 else "Autocorrelacionados"
    color_lb = "#28B463" if p_lb > 0.05 else "#CB4335"

    reporte_detallado = f"""
    <div style='margin-top: 15px; padding: 10px; border: 1px solid #ddd; background-color: #FDFEFE;'>
        <b>Prueba de Ljung-Box (Residuos):</b><br>
        p-value: <span style='color: {color_lb}; font-weight: bold;'>{p_lb:.4f}</span><br>
        Estado: <b style='color: {color_lb};'>{status_lb}</b>
        <p style='font-size: 11px;'><i>* Un modelo válido no debe tener autocorrelación en los residuos.</i></p>
    </div>
    """

    df_final = pd.concat([df_plot_hist, df_plot_pred[['Año', 'Valor']]]).reset_index(drop=True)

    return df_final, reporte_html, fig_pred, tabla_pred_html, reporte_detallado

def tab_ST_serie_complete(df, metodo):
    """
    Completa años faltantes en una serie temporal para una provincia y departamento específicos.
    df           -- DataFrame filtrado por provincia y departamento
    metodo       -- Entero (1 a 5) para la técnica de imputación
    """
    
    # 1. Filtrado y preparación de tipos
    # df_filtered = df[(df['provincia'] == provincia) & 
    #                  (df['departamento'] == departamento)].copy()
    df_filtered = df.copy()

    if df_filtered.empty:
        return pd.DataFrame(), 0, 0

    cols_fijas = ['cultivo', 'periodo', 'provincia', 'departamento']
    cols_num = [c for c in df_filtered.columns if c not in cols_fijas]
    
    df_filtered['periodo'] = df_filtered['periodo'].astype(int)
    
    anio_min = df_filtered['periodo'].min()
    anio_max = df_filtered['periodo'].max()
    rango_completo = np.arange(anio_min, anio_max + 1)

    # 2. Reindexación (Expansión de la serie)
    df_res = (
        df_filtered.set_index('periodo')
        .reindex(rango_completo)
        .reset_index()
        .rename(columns={'index': 'periodo'})
    )

    # 3. Propagación de etiquetas cualitativas
    df_res[['cultivo', 'provincia', 'departamento']] = (
        df_res[['cultivo', 'provincia', 'departamento']].ffill().bfill()
    )

    # 4. Lógica de Imputación
    if metodo == 1:
        df_res[cols_num] = df_res[cols_num].interpolate(method='linear')
    elif metodo == 2:
        if df_filtered.shape[0] > 2:
            df_res[cols_num] = df_res[cols_num].interpolate(method='polynomial', order=2)
        else:
            df_res[cols_num] = df_res[cols_num].interpolate(method='linear')
    elif metodo in [3, 4, 5]:
        window_size = 3 if metodo in [3, 5] else 4
        is_centered = (metodo == 5)
        ma_values = df_res[cols_num].rolling(window=window_size, center=is_centered, min_periods=1).mean()
        df_res[cols_num] = df_res[cols_num].fillna(ma_values)

    # 5. Extrapolación de bordes
    df_res[cols_num] = df_res[cols_num].ffill().bfill()
    
    # 6. Cálculo de registros que aún faltan (Remanentes)
    # Se considera un registro "faltante" si tiene valores NaN en sus columnas numéricas
    # después de haber intentado todas las técnicas de imputación.
    regs_faltantes_final = df_res[cols_num].isnull().any(axis=1).sum()
    
    # Reordenamiento final
    orden_final = ['cultivo', 'periodo', 'provincia', 'departamento'] + cols_num
    df_res = df_res[orden_final]

    return df_res, anio_min, anio_max, len(df_res), regs_faltantes_final

def tab_ST_imputar_df(df, imp_option, indicador, serie, mg, tend, mm, sd):

    df_imput, inicio, fin, regs, regsno = tab_ST_serie_complete(df, imp_option)

    # Se arman las listas desplegables de los años con los años imputados
    start_year = int(inicio)
    end_year = int(fin)
    years = list(range(start_year, end_year + 1))
    years_inv = years[::-1]

    # "indicador" se convierte de nombre descriptivo corto a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(df_imput, ind_orig, serie, mg, tend, mmov, sd, 
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    
    return (df_imput, 
                # Cantidad de registros
                regs, regsno,
                # Campos de listado de años y mensaje de error
                gr.update(choices=years, value=start_year),
                gr.update(choices=years_inv, value=end_year),
                None,
                # Botones imputar, cortar, restaurar
                gr.update(interactive=False), gr.update(interactive=True),
                gr.update(visible=True), 
                # Gráfico de tendencia
                graph,
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(),
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
                gr.update(),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
            )

def tab_ST_restaurar_df(df, cultivo, provincia, departamento,
                                  indicador, serie, mg, tend, mm, sd):

    # Se filtra el dataset de cultivo
    if not df.empty:
        filtered = get_filtered_subset(df, cultivo, provincia, departamento, KEY_COLUMNS)

    # Se obtiene información de la serie
    inicio = pd.to_numeric(filtered['periodo']).min()
    final = pd.to_numeric(filtered['periodo']).max()
    regs = filtered['periodo'].nunique()
    regsno = (final - inicio + 1) - regs
    inicio = int(inicio)
    final = int(final)

    # "indicador" se convierte de nombre descriptivo corto a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, 
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)
    
    return (filtered, # Dataset filtrado original
                # Campos de registros
                inicio, final, regs, regsno,
                # Listado de años y mensaje de error
                inicio, final, None,
                # Botones imputar, cortar, restaurar
                gr.Button(interactive=True), gr.Button(interactive=True),
                gr.Button(interactive=False),
                # Área y gráfico de tendencia
                gr.update(visible=True), graph,
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(),
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
                gr.update(),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )

def tab_ST_cortar_df(df, indicador, yinicial, yfinal, serie, mg, tend, mm, sd):

    df_cut, inicio, final, regs, regsno, error = tab_ST_serie_cut(df, yinicial, yfinal)

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    
    if error:
        return (df_cut, # Dataframe sin cortar
                # Campos de registros
                inicio, final, regs, regsno,
                # Campos de listado de años y mensaje de error
                yinicial, yfinal, error,
                # Botones cortar, restaurar
                gr.Button(interactive=True), gr.update(),
                # Area y gráfico de tendencia
                gr.update(), gr.update(),
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(),
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
                gr.update(),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )


    mmov =  0 < mm < 4
    tipo = mm + 1
    graph = tab_EDA_create_evolution_graph(df_cut, ind_orig, serie, mg, tend, mmov, sd, 
                                        tipo, interactivo=INTERACTIVO, h_interactivo=H_INTERACTIVO)

    return (df_cut, # Dataframe cortado
                # Campos de registros
                inicio, final, regs, regsno,
                # Campos de listado de años y mensaje de error
                yinicial, yfinal, error,
                # Botones cortar, restaurar
                gr.Button(interactive=False), gr.Button(interactive=True),
                # Area y gráfico de tendencia
                gr.update(visible=True), graph,
                ## Sección de Descomposición de la Serie
                # Información de filtros
                gr.update(),
                # Gráfico de descomposición e interpretación
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección de Diferenciación y Prueba ADF
                # Información de filtros
                gr.update(),
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
                gr.update(),
                # Gráfico e interpretación de ACF
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráfico e interpretación de PACF
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección ARIMA
                # Variables de estado 'p' y 'q'
                gr.update(value=NO_EXISTE), gr.update(value=NO_EXISTE),
                # Información de filtros y parámetros de ARIMA
                gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                gr.Plot(visible=False), gr.update(visible=False),
                # Gráficos de residuos e info de residuos
                gr.Plot(visible=False), gr.update(visible=False),
                ## Sección AUTO-ARIMA
                # Información de filtros y parámetros de AUTO-ARIMA
                # gr.update(), gr.update(visible=False),
                # Gráfico de predicciones y tabla de predicciones
                # gr.Plot(visible=False), gr.update(visible=False),
                # Info de residuos
                # gr.update(visible=False),
                ## Sección FOURIER
                # Información de filtros
                gr.update(),
                # Gráfico de series originales y diferenciadas
                gr.Plot(visible=False), gr.Plot(visible=False),
                # Gráficos de Transformada de Fourier y otras
                gr.Plot(visible=False),
                # Informe de Transformada de Fourier y otras
                gr.update(visible=False)
                )

def tab_ST_serie_cut(df, yinicial, yfinal):

    if yinicial >= yfinal:
        msg = "RANGO INCORRECTO"
        return df, gr.update(), gr.update(), gr.update(), gr.update(), msg
    
    if (yfinal - yinicial + 1) < 30:
        msg = "RANGO < 30 AÑOS"
        return df, gr.update(), gr.update(), gr.update(), gr.update(), msg

    # Se recorta el df
    df['periodo'] = pd.to_numeric(df['periodo'], errors='coerce')
    df_cut = df[(df['periodo'] >= yinicial) & (df['periodo'] <= yfinal)].copy()
    df_cut = df_cut.sort_values('periodo')

    # Sobre el df cortado se calculan los nuevos datos
    inicio = pd.to_numeric(df_cut['periodo']).min()
    final = pd.to_numeric(df_cut['periodo']).max()
    regs = df_cut['periodo'].nunique()
    regsno = (final - inicio + 1) - regs

    return df_cut, inicio, final, regs, regsno, None

def tab_ST_FOURIER(df1, df2, df3, diff1, diff2, diff3, d1, d2, d3,
                                    cult1, cult2, cult3, prov1, prov2, prov3, 
                                    dep1, dep2, dep3, ind1, ind2, ind3):
    
    if df1.empty:
        text1, ind1 = None, None
    else:
        ind1 = next((k for k, v in dict_ncortos.items() if v == ind1), ind1) # De nombre descriptivo corto a nombre original
        text1 = f"{cult1} - {prov1} - {dep1} - {dict_nlargos[ind1].upper()}"
    
    if df2.empty:
        text2, ind2 = None, None
    else:
        ind2 = next((k for k, v in dict_ncortos.items() if v == ind2), ind2)
        text2 = f"{cult2} - {prov2} - {dep2} - {dict_nlargos[ind2].upper()}"

    if df3.empty:
        text3, ind3 = None, None
    else:
        ind3 = next((k for k, v in dict_ncortos.items() if v == ind3), ind3)
        text3 = f"{cult3} - {prov3} - {dep3} - {dict_nlargos[ind3].upper()}"
    
    # Gráfico de las series originales
    titulo = "SERIES ORIGINALES"
    graph_orig = tab_ST_create_evolution_graph_triple(df1, df2, df3, 
                            ind1, ind2, ind3, text1, text2, text3, titulo, serie=True,
                            med_glob=False, tend=False, med_mov=False, sd_mov=False, tipo_mov=4, 
                            interactivo=True, h_estatico=6, h_interactivo=H_INTERACTIVO1)
        
    # Gráfico de las series diferenciadas
    titulo = "SERIES DIFERENCIADAS"
    graph_diff = tab_ST_create_evolution_graph_triple(diff1, diff2, diff3, 
                            ind1, ind2, ind3, text1, text2, text3, titulo, serie=True,
                            med_glob=False, tend=False, med_mov=False, sd_mov=False, tipo_mov=4, 
                            interactivo=True, h_estatico=6, h_interactivo=H_INTERACTIVO1)

    # Cálculo completo de FFT, SFTF y wavelets para series originales y diferenciadas
    graph_full_1, reporte1 = tab_ST_Fourier_Wavelets_Full(df1, diff1, ind1, d1, fs=1.0)
    graph_full_2, reporte2 = tab_ST_Fourier_Wavelets_Full(df2, diff2, ind2, d2, fs=1.0)
    graph_full_3, reporte3 = tab_ST_Fourier_Wavelets_Full(df3, diff3, ind3, d3, fs=1.0)

    return (# Gráfico series originales
            gr.update(visible=graph_orig is not None, value=graph_orig),
            # Gráfico de series diferenciadas
            gr.update(visible=graph_diff is not None, value=graph_diff),
            # gráficos e informe de serie 1
            gr.update(visible=graph_full_1 is not None, value=graph_full_1),
            gr.update(visible=True, value=reporte1),
            # Gráficos e informe de serie 2
            gr.update(visible=graph_full_2 is not None, value=graph_full_2), 
            gr.update(visible=True, value=reporte2),
            # Gráficos e informe de serie 3
            gr.update(visible=graph_full_3 is not None, value=graph_full_3), 
            gr.update(visible=True, value=reporte3)
            )

def tab_ST_create_evolution_graph_triple(df1, df2, df3, ind1, ind2, ind3, 
                                            text1, text2, text3, title_graph,
                                            serie=True, med_glob=True, tend=True, 
                                            med_mov=False, sd_mov=False, tipo_mov=4, 
                                            interactivo=False, h_estatico=6, h_interactivo=450):
    """
    - df1, df2, df3: datasets con columnas ['periodo', indicador]
    - ind1, ind2, ind3: nombres originales de columnas para cada df
    - text1, text2, text3: leyendas personalizadas para cada serie
    - title_graph: título del gráfico
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

    # 1. Validación y Preparación de Datos en una estructura iterable
    # Se agrupan los datos con sus respectivos textos y colores asignados
    input_data = [
        (df1, ind1, text1, 'red'),
        (df2, ind2, text2, 'blue'),
        (df3, ind3, text3, 'orange')
    ]
    
    datasets = []
    for d, ind, txt, color in input_data:
        if d is not None and not d.empty:
            df_s = d.sort_values('periodo').reset_index(drop=True)
            datasets.append({'df': df_s, 'ind': ind, 'label': txt, 'color': color})

    if not datasets:
        return None

    # Configuración de la ventana para cálculos móviles (k y centrado)
    if tipo_mov == 3:
        k, centro, lab_m = 3, False, "3 atrás"
    elif tipo_mov == 4:
        k, centro, lab_m = 3, True, "3 centrado"
    else:
        k, centro, lab_m = 2, False, "2"

    # --- OPCIÓN 1: GRÁFICO INTERACTIVO (PLOTLY) ---
    if interactivo:
        fig = go.Figure()

        for ds in datasets:
            x_data = ds['df']['periodo']
            y_data = ds['df'][ds['ind']]
            label = ds['label']
            color = ds['color']

            # Serie principal: usa el parámetro text correspondiente
            if serie:
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode='lines+markers', name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=7)
                ))

            # Media Global (Línea horizontal)
            if med_glob:
                media_v = y_data.mean()
                fig.add_hline(y=media_v, line_dash="dash", line_color=color, 
                              opacity=0.4, annotation_text=f"Media {label}")

            # Línea de Tendencia (Regresión Lineal)
            if tend:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=x_data, y=p(x_data), mode='lines', name=f'Tend. {label}',
                    line=dict(color=color, dash='dash', width=1.5),
                    opacity=0.7
                ))

            # Cálculos móviles
            if med_mov or sd_mov:
                y_med_mov = y_data.rolling(window=k, center=centro).mean()
                if med_mov:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_med_mov, mode='lines', 
                        name=f'MM {label} ({lab_m})', line=dict(dash='dot', width=2)
                    ))
                if sd_mov:
                    y_sd_mov = y_data.rolling(window=k, center=centro).std()
                    upper_b, lower_b = y_med_mov + y_sd_mov, y_med_mov - y_sd_mov
                    mask = upper_b.notna() & lower_b.notna()
                    
                    fig.add_trace(go.Scatter(
                        x=pd.concat([x_data[mask], x_data[mask][::-1]]),
                        y=pd.concat([upper_b[mask], lower_b[mask][::-1]]),
                        fill='toself', fillcolor=f'rgba(0,0,0,0.05)', # Sombra muy tenue para no solapar
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'Banda SD {label}', hoverinfo='skip'
                    ))

        # Ajustes estéticos de Plotly
        fig.update_layout(
            title={'text': f"<b>{title_graph}</b>",
                'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20}
            },
            height=h_interactivo,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=80, b=40),
            
            legend=dict(
                yanchor="top",    # El punto de referencia vertical es la parte superior de la leyenda
                y=0.98,           # Ubicación vertical (ligeramente debajo del borde superior del área de trazado)
                xanchor="right",  # El punto de referencia horizontal es la parte derecha de la leyenda
                x=0.98,           # Ubicación horizontal (ligeramente a la izquierda del borde derecho)
                bgcolor="rgba(255, 255, 255, 0)",  # Fondo semitransparente para no tapar datos
            ),
            
            xaxis=dict(
                # title=dict(text="<b>Etiqueta Eje X</b>", font=dict(size=10)),
                tickfont=dict(family='Arial, sans-serif', size=16, color='black'),
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                tickmode='auto',
                nticks=20
            ),
            
            yaxis=dict(
                # title=dict(text="<b>Etiqueta Eje Y</b>", font=dict(size=10)),
                tickfont=dict(family='Arial, sans-serif', size=16, color='black'),
                showgrid=True,
                gridcolor='LightGray',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            )
        )
        return fig

    # --- OPCIÓN 2: GRÁFICO ESTÁTICO (MATPLOTLIB) ---
    else:
        fig, ax = plt.subplots(figsize=(12, h_estatico))
        try:
            for ds in datasets:
                x_data = ds['df']['periodo']
                y_data = ds['df'][ds['ind']]
                label = ds['label']
                color = ds['color']

                if serie:
                    ax.plot(x_data, y_data, label=label, marker='o', linewidth=2, color=color)

                if med_glob:
                    ax.axhline(y=y_data.mean(), color=color, linestyle='--', alpha=0.3)

                if tend:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), color=color, linestyle=':', alpha=0.5)

                if med_mov or sd_mov:
                    y_med_mov = y_data.rolling(window=k, center=centro).mean()
                    if med_mov:
                        ax.plot(x_data, y_med_mov, linestyle='--', alpha=0.7, label=f'MM {label}')
                    if sd_mov:
                        y_sd_mov = y_data.rolling(window=k, center=centro).std()
                        ax.fill_between(x_data, y_med_mov - y_sd_mov, y_med_mov + y_sd_mov, color=color, alpha=0.08)

            ax.grid(True, linestyle='--', alpha=0.6)
            # Colocamos la leyenda fuera del gráfico para evitar solapamiento con 3 series
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            plt.tight_layout()
            return fig
        finally:
            plt.close(fig)

def tab_ST_Fourier_Wavelets_Full(df_orig, df_diff, var, d, fs=1.0):
    """
    Análisis Espectral con visualización unificada.
    Optimizado para visualizar ciclos en el dominio del tiempo.
    """
    
    # Validaciones de entrada
    if df_orig.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se seleccionó una serie temporal.
            </div>
            """
        return None, reporte
    
    if df_diff.empty:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                No se verificó la estacionariedad de la serie temporal.
            </div>
            """
        return None, reporte

    # Se verifica que la serie sea estacionaria
    # (NO_EXISTE = sin prueba ADF, no se verificó estacionariedad)
    if d == NO_EXISTE:
        reporte = """
            <div style="padding:15px; border:2px solid #ffa000; background-color:#fff9c4; border-radius:8px;">
                <strong style="color:#f57c00;"> ERROR EN EL ANÁLISIS:</strong><br>
                La serie debe ser estacionaria para aplicar la Transformada Rápida de Fourier.
                Debe aplicarse la diferenciación (si corresponde) y
                la prueba ADF para verificar ESTACIONARIEDAD.</div>
            """
        return None, reporte

   
    # 1. Preparación de datos
    y_orig = df_orig[var].values
    anios_orig = df_orig['periodo'].values
    
    # 1. Preparación de la señal
    y_diff = df_diff[var].dropna().values
    anios_diff = df_diff['periodo'].dropna().values
    n = len(y_diff)

    # --- 2. CÁLCULOS TÉCNICOS ---
    
    """
    # Cálculo de la FFT Real
    fft_coeffs = np.fft.rfft(y_diff)
    frecuencias = np.fft.rfftfreq(n, d=1/fs)
    # Magnitud normalizada (como en su ejemplo: * 2 / n)
    magnitud = np.abs(fft_coeffs) * 2 / n

    # Conversión de Frecuencias a Períodos (Eje X para el segundo gráfico)
    # Evitamos división por cero en la frecuencia fundamental (0)
    con_periodo = np.zeros_like(frecuencias)
    con_periodo[1:] = 1 / frecuencias[1:] 

    # FILTRADO E IFFT PARA EL PRIMER GRÁFICO (Dominio del Tiempo)
    # Mantener los 5 picos principales para reconstruir la intensidad en el tiempo
    umbral = np.sort(magnitud)[-5]
    fft_filtrada = np.where(magnitud >= umbral, fft_coeffs, 0)
    y_ciclica_reconstruida = np.fft.irfft(fft_filtrada, n=n)
    """

    # FFT para señales reales
    fft_coeffs = np.fft.rfft(y_diff)
    frecuencias = np.fft.rfftfreq(n, d=1/fs)
    # Magnitud normalizada
    magnitud = np.abs(fft_coeffs) * 2 / n

    # Cálculo del umbral de significancia (95% confianza: media + 2 sigma)
    media_mag = np.mean(magnitud)
    desv_mag = np.std(magnitud)
    umbral_95 = media_mag + (2 * desv_mag)

    # Conversión de Frecuencia a Período (Eje X del gráfico 2)
    con_periodo = np.zeros_like(frecuencias)
    con_periodo[1:] = 1 / frecuencias[1:] 

    # TRANSFORMADA INVERSA (IFFT): Solo componentes que superan el ruido (Significativos)
    fft_filtrada = np.where(magnitud >= umbral_95, fft_coeffs, 0)
    y_ciclica_reconstruida = np.fft.irfft(fft_filtrada, n=n)


    # STFT: Evolución temporal de las frecuencias
    nperseg = min(20, len(y_orig) // 3)
    f_stft, t_stft, Zxx = stft(y_orig, fs=fs, nperseg=nperseg, noverlap=nperseg-1)
    t_stft_años = anios_orig[0] + t_stft

    # Wavelets (CWT)
    scales = np.arange(1, 31)
    coef, freqs_wav = pywt.cwt(y_orig, scales, 'cmor1.5-1.0', sampling_period=1/fs)
    power_wav = (np.abs(coef)) ** 2

    # --- 3. GENERACIÓN DE GRÁFICOS ---
    fig_espectral = make_subplots(
        rows=4, cols=1, 
        subplot_titles=(
            'Intensidad de Ciclos Dominantes<br>en el Tiempo (Inversa de la FFT)',
            'Espectro de Magnitud vs. Período<br>(Ciclos Detectados)', 
            'Evolución Temporal de Frecuencias<br>(STFT)', 
            'Análisis Tiempo-Frecuencia<br>(Wavelets)'
        ),
        vertical_spacing=0.07
    )

    """
    # A. GRÁFICO 1: IFFT (Dominio del Tiempo)
    fig_espectral.add_trace(go.Scatter(
        x=anios_diff, y=y_ciclica_reconstruida,
        mode='lines', line=dict(color='#2ECC71', width=2.5),
        fill='tozeroy', name='Intensidad Temporal'
    ), row=1, col=1)

    # B. GRÁFICO 2: Magnitud vs Período (Como su código de ventas)
    # Filtramos periodos muy largos para que el gráfico sea legible (ej. < 20 años)
    mask_visible = (con_periodo > 0) & (con_periodo < 20)
    fig_espectral.add_trace(go.Scatter(
        x=con_periodo[mask_visible], 
        y=magnitud[mask_visible],
        mode='lines+markers',
        line=dict(color='#1F77B4', width=2),
        marker=dict(size=4),
        name='Espectro de Período'
    ), row=2, col=1)
    """

    # A. Reconstrucción Temporal (IFFT)
    fig_espectral.add_trace(go.Scatter(
        x=anios_diff, y=y_ciclica_reconstruida,
        mode='lines', line=dict(color='#2ECC71', width=2.5),
        fill='tozeroy', name='Ciclos Significativos'
    ), row=1, col=1)

    # B. Magnitud vs Período con Validación Visual
    mask_vis = (con_periodo > 0) & (con_periodo < 25) # Filtro de visualización
    fig_espectral.add_trace(go.Scatter(
        x=con_periodo[mask_vis], y=magnitud[mask_vis],
        mode='lines+markers', line=dict(color='#1F77B4', width=2),
        marker=dict(size=5)
    ), row=2, col=1)

    # LÍNEA DE VALIDACIÓN VISUAL (Umbral Estadístico)
    fig_espectral.add_shape(
        type="line", x0=0, x1=25, y0=umbral_95, y1=umbral_95,
        line=dict(color="Red", width=2, dash="dash"), row=2, col=1
    )

    fig_espectral.add_annotation(
        x=22, y=umbral_95, text="Confianza 95%", showarrow=False,
        font=dict(color="red", size=11), bgcolor="white", row=2, col=1
    )

    # C. Espectrograma (STFT)
    fig_espectral.add_trace(go.Heatmap(
        z=np.abs(Zxx), x=t_stft_años, y=f_stft, 
        colorscale='Viridis', showscale=False
    ), row=3, col=1)

    # D. Wavelets (CWT)
    fig_espectral.add_trace(go.Contour(
        z=power_wav, x=anios_orig, y=freqs_wav, 
        colorscale='Magma', showscale=False
    ), row=4, col=1)

    # --- Configuración Estética ---
    fig_espectral.update_layout(
        height=1200, 
        template="plotly_white",
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False # Desactivación global de leyendas
    )
    
    fig_espectral.update_annotations(font=dict(family="Arial Black", size=14, color="black"))
    fig_espectral.update_xaxes(showgrid=True, gridcolor='lightgray', nticks=10)
    fig_espectral.update_yaxes(showgrid=True, gridcolor='lightgray')

    # --- 4. REPORTE ANALÍTICO ---

    # Cálculo del umbral de significancia (95% de confianza aproximado)
    media_mag = np.mean(magnitud)
    desv_mag = np.std(magnitud)
    umbral_95 = media_mag + (2 * desv_mag)

    # Obtenemos los índices de los 4 picos con mayor magnitud
    picos_idx = np.argsort(magnitud)[-4:][::-1]
    filas_html = ""
    for idx in picos_idx:
        p_val, f_val, mag_val = con_periodo[idx], frecuencias[idx], magnitud[idx]
        es_sig = mag_val > umbral_95
        color = "#D4EFDF" if es_sig else "#FFFFFF" # Verde si es significativo
        
        filas_html += f"""
            <tr style='background-color: {color};'>
                <td style='padding:8px; border:1px solid black;'>{p_val:.2f} años{' (Sig.)' if es_sig else ''}</td>
                <td style='padding:8px; border:1px solid black; text-align:right;'>{f_val:.4f}</td>
                <td style='padding:8px; border:1px solid black; text-align:right; font-weight:bold;'>{mag_val:.2f}</td>
            </tr>"""

    reporte_html = f"""
    <div style='font-family: Arial; font-size: 14px;'>
        <h4 style='color: #000000;'>Análisis de Ciclicidad con Umbral Estadístico</h4>
        <table style='width: 100%; border-collapse: collapse; border: 1px solid black;'>
            <tr style='background-color: #F2F4F4;'>
                <th style='padding:10px; border:1px solid black;'>Período (Años)</th>
                <th style='padding:10px; border:1px solid black;'>Frecuencia (1/T)</th>
                <th style='padding:10px; border:1px solid black;'>Potencia</th>
            </tr>
            {filas_html}
        </table>
        <div style='margin-top:10px; padding:10px; border: 1px solid #27AE60; background-color: #EBF5FB;'>
            Las filas con fondo verde y la línea roja en el gráfico
            indican componentes que superan el umbral de confianza del 95% (media + 2 SD). 
            Solo estos componentes han sido utilizados para la reconstrucción temporal (IFFT).
        </div>
    </div>"""

    return fig_espectral, reporte_html

def tab_ST_Fourier_Wavelets_Triple(df_o1, df_o2, df_o3, df_d1, df_d2, df_d3, var1, var2, var3, fs=1.0):
    """
    Compara simultáneamente tres series temporales en los dominios de 
    tiempo, frecuencia y tiempo-frecuencia (STFT y Wavelets).
    """
    
    # Listas para iteración
    dfs_orig = [df_o1, df_o2, df_o3]
    dfs_diff = [df_d1, df_d2, df_d3]
    vars_list = [var1, var2, var3]
    colores = ['#1f77b4', '#FF7F0E', '#2ca02c'] # Azul, Naranja, Verde
    
    # Crear Subplots: 4 filas (Tiempo, FFT, STFT, CWT) x 3 columnas (Serie 1, 2, 3)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            f"Tiempo: {var1}", f"Tiempo: {var2}", f"Tiempo: {var3}",
            f"FFT: {var1}", f"FFT: {var2}", f"FFT: {var3}",
            f"STFT: {var1}", f"STFT: {var2}", f"STFT: {var3}",
            f"Wavelets: {var1}", f"Wavelets: {var2}", f"Wavelets: {var3}"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    reporte_filas = ""

    for i in range(3):
        col = i + 1
        v = vars_list[i]
        
        # 1. Preparación de datos
        y_o = dfs_orig[i][v].values
        a_o = dfs_orig[i]['periodo'].values
        df_d_c = dfs_diff[i][['periodo', v]].dropna()
        y_d = df_d_c[v].values
        a_d = df_d_c['periodo'].values

        # 2. Cálculos
        # FFT Welch
        f_fft, mag_fft = welch(y_d, fs=fs, nperseg=len(y_d))
        # STFT
        nperseg = min(16, len(y_o) // 3)
        f_s, t_s, Zxx = stft(y_o, fs=fs, nperseg=nperseg, noverlap=nperseg-1)
        t_s_años = a_o[0] + t_s
        # CWT
        coef, freqs_w = pywt.cwt(y_o, np.arange(1, 31), 'cmor1.5-1.0', sampling_period=1/fs)
        p_w = (np.abs(coef)) ** 2

        # 3. Gráficos
        # Fila 1: Tiempo (Serie Diferenciada)
        fig.add_trace(go.Scatter(x=a_d, y=y_d, name=v, line=dict(color=colores[i])), row=1, col=col)
        
        # Fila 2: FFT
        fig.add_trace(go.Bar(x=f_fft, y=mag_fft, marker_color=colores[i], showlegend=False), row=2, col=col)
        
        # Fila 3: STFT
        fig.add_trace(go.Heatmap(z=np.abs(Zxx), x=t_s_años, y=f_s, colorscale='Viridis', showscale=False), row=3, col=col)
        
        # Fila 4: CWT
        fig.add_trace(go.Contour(z=p_w, x=a_o, y=freqs_w, colorscale='Magma', showscale=False), row=4, col=col)

        # 4. Construcción de Reporte Analítico
        picos = np.argsort(mag_fft)[-2:][::-1]
        for p_idx in picos:
            per = round(1/f_fft[p_idx], 2) if f_fft[p_idx] != 0 else "∞"
            reporte_filas += f"""
                <tr>
                    <td style='border:1px solid black; padding:5px;'><b>{v}</b></td>
                    <td style='border:1px solid black; padding:5px; text-align:center;'>{per} años</td>
                    <td style='border:1px solid black; padding:5px; text-align:right;'>{mag_fft[p_idx]:.2f}</td>
                </tr>"""

    # Diseño final
    fig.update_layout(height=1400, template="plotly_white", title_text="<b>COMPARATIVA ESPECTRAL TRIPLE</b>", title_font_size=20)
    fig.update_annotations(font=dict(family="Arial Black", size=12))
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))

    reporte_html = f"""
    <div style='font-family: Arial; font-size: 14px;'>
        <h4 style='color: #000000;'>Resumen de Periodicidades Dominantes</h4>
        <table style='width: 100%; border-collapse: collapse; border: 1px solid black;'>
            <thead style='background-color: #F2F4F4;'>
                <tr>
                    <th style='border: 1px solid black; padding: 8px;'>Variable</th>
                    <th style='border: 1px solid black; padding: 8px;'>Ciclo Dominante</th>
                    <th style='border: 1px solid black; padding: 8px;'>Potencia</th>
                </tr>
            </thead>
            <tbody>{reporte_filas}</tbody>
        </table>
    </div>"""

    return fig, reporte_html

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
    # Almacenamiento de los tres datasets con valores históricos y predichos con ARIMA
    # CUIDADO: no es un dataset completo, solo tiene PERIODO y la variable (VALOR) que se esté analizando
    # No es un dataset filtrado que tenga CULTIVO, PROVINCIA y DEPARTAMENTO
    dataset_ARIMA_state_1 = gr.State(pd.DataFrame())
    dataset_ARIMA_state_2 = gr.State(pd.DataFrame())
    dataset_ARIMA_state_3 = gr.State(pd.DataFrame())
    # Almacenamiento de los tres datasets con valores históricos y predichos con AUTO-ARIMA
    # CUIDADO: no es un dataset completo, solo tiene PERIODO y la variable (VALOR) que se esté analizando
    # No es un dataset filtrado que tenga CULTIVO, PROVINCIA y DEPARTAMENTO
    dataset_AUTO_ARIMA_state_1 = gr.State(pd.DataFrame())
    dataset_AUTO_ARIMA_state_2 = gr.State(pd.DataFrame())
    dataset_AUTO_ARIMA_state_3 = gr.State(pd.DataFrame())


    gr.Row(elem_classes="header-tab")
    

    ###### ESTRUCTURA DE PESTAÑAS
    with gr.Tabs():


        ###### PESTAÑA INICIO
        with gr.Tab("Inicio"):
            with gr.Row():
                with gr.Column(scale=8, elem_classes="portrait-bg-video"): # elem_classes="portrait-bg-1"):
                    gr.HTML(portada_video)
                    gr.HTML("ANÁLISIS <br> DE SERIES<br>TEMPORALES<br>"
                            "ANUALES<br>DE LOS<br>PRINCIPALES<br>CULTIVOS<br>"
                            "DE LA<br>REPÚBLICA ARGENTINA", elem_classes="portrait-title")
                with gr.Column(scale=2, elem_classes="portrait-bg-2"):
                    gr.HTML("Aplicación de algoritmos de Machine Learning "
                            "a las Bases de Datos Abiertas "
                            "de la Dirección Nacional de Agricultura "
                            "del Ministerio de Agricultura, Ganadería y Pesca "
                            "de la República Argentina, "
                            "para el análisis de series temporales anuales "
                            "de los cultivos más importantes "
                            "en las principales regiones productoras del país.",
                            elem_classes="portrait-subtitle")
        

        ###### PESTAÑA DASHBOARD
        with gr.Tab("Dashboard"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;INDICADORES DE CULTIVOS", elem_classes="title-text")
        

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

        ###### PESTAÑA COMPARACIÓN DE SERIES TEMPORALES
        with gr.Tab("Series Temporales") as tab_ST:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;COMPARACIÓN DE SERIES TEMPORALES", elem_classes="title-text")
            
            # region SECCIÓN 1: SELECCIÓN DE LAS TRES SERIES TEMPORALES
            with gr.Tab("Selección de las Series") as subtab_sel_series:
               
                with gr.Row():
                    with gr.Column(elem_classes="custom-tab-2"):    
                        gr.HTML("&nbsp;&nbsp;1. SELECCIÓN DE LAS SERIES TEMPORALES A COMPARAR", 
                                elem_classes="subtitle-text")
                
                with gr.Row():
                    with gr.Column():
                        ### CULTIVO 1
                        with gr.Column(elem_classes="custom-tab"):
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    with gr.Row():
                                        gr.HTML("CULTIVO", elem_classes="info-display-4")
                                        cult1 = gr.Dropdown(label="", 
                                                            choices=["Elegir cultivo...", "ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                            "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                            value="Elegir cultivo...", elem_classes="custom-dropdown-small")
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

                                with gr.Column(scale=20):
                                    with gr.Row(visible=True, min_height=H_INTERACTIVO, 
                                                elem_classes="custom-graf-panel") as tend1_area:
                                        tend1 = gr.Plot(show_label=False, visible=False,
                                                        elem_classes="custom-graf-panel")

                            with gr.Row():
                                with gr.Column(min_width=80):
                                    imp1_button = gr.Button("Imputar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", min_width=50)
                                with gr.Column(min_width=80):
                                    cut1_button = gr.Button("Cortar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", min_width=50)
                                with gr.Column(scale=8):
                                    imp1_option = gr.Radio(label="Imputación",
                                                choices=["Interpolación lineal", 
                                                "Interpolación polinómica",
                                                "Media móvil (k=2)",
                                                "Media móvil (k=3 hacia atrás)",
                                                "Media móvil (k=3 centrada)"],
                                                value="Interpolación lineal", type="index", 
                                                show_label=False,
                                                elem_classes=["custom-radio", "radio-horizontal", "special-margin1"])
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    rest1_button = gr.Button("Restaurar Serie", variant="primary", visible=True, 
                                            elem_classes="custom-button5", min_width=50, 
                                            interactive=False)
                                with gr.Column(scale=8):
                                    with gr.Row():
                                        gr.HTML("Año inicial de la serie", elem_classes="info-display-4")
                                        yinic1 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        gr.HTML("Año final de la serie", elem_classes="info-display-4")
                                        yfin1 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        err1 = gr.HTML("<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                                                    elem_classes="info-display-6")
                        
                        ### CULTIVO 2                
                        with gr.Column(elem_classes="custom-tab"):
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    with gr.Row():
                                        gr.HTML("CULTIVO", elem_classes="info-display-4")
                                        cult2 = gr.Dropdown(label="", 
                                                            choices=["Elegir cultivo...", "ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                            "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                            value="Elegir cultivo...", elem_classes="custom-dropdown-small")
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
                                    with gr.Row(visible=True, min_height=H_INTERACTIVO, 
                                                elem_classes="custom-graf-panel") as tend2_area:
                                        tend2 = gr.Plot(show_label=False, visible=False,
                                                        elem_classes="custom-graf-panel")

                            with gr.Row():
                                with gr.Column(min_width=80):
                                    imp2_button = gr.Button("Imputar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", min_width=50)
                                with gr.Column(min_width=80):
                                    cut2_button = gr.Button("Cortar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", interactive=True, min_width=50)
                                with gr.Column(scale=8):
                                    imp2_option = gr.Radio(label="Imputación",
                                                choices=["Interpolación lineal", 
                                                "Interpolación polinómica",
                                                "Media móvil (k=2)",
                                                "Media móvil (k=3 hacia atrás)",
                                                "Media móvil (k=3 centrada)"],
                                                value="Interpolación lineal", type="index", 
                                                show_label=False,
                                                elem_classes=["custom-radio", "radio-horizontal", "special-margin1"])
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    rest2_button = gr.Button("Restaurar Serie", variant="primary", visible=True, 
                                            elem_classes="custom-button5", min_width=50, 
                                            interactive=False)
                                with gr.Column(scale=8):
                                    with gr.Row():
                                        gr.HTML("Año inicial de la serie", elem_classes="info-display-4")
                                        yinic2 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        gr.HTML("Año final de la serie", elem_classes="info-display-4")
                                        yfin2 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        err2 = gr.HTML("<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                                                    elem_classes="info-display-6")

                        ### CULTIVO 3
                        with gr.Column(elem_classes="custom-tab"):
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    with gr.Row():
                                        gr.HTML("CULTIVO", elem_classes="info-display-4")
                                        cult3 = gr.Dropdown(label="", 
                                                            choices=["Elegir cultivo...", "ARROZ", "AVENA", "GIRASOL", "MAÍZ", "POROTO",
                                                            "SOJA", "SORGO", "TRIGO", "YERBA MATE"],
                                                            value="Elegir cultivo...", elem_classes="custom-dropdown-small")
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
                                    with gr.Row(visible=True, min_height=H_INTERACTIVO, 
                                                elem_classes="custom-graf-panel") as tend3_area:
                                        tend3 = gr.Plot(show_label=False, visible=False,
                                                        elem_classes="custom-graf-panel")

                            with gr.Row():
                                with gr.Column(min_width=80):
                                    imp3_button = gr.Button("Imputar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", min_width=50)
                                with gr.Column(min_width=80):
                                    cut3_button = gr.Button("Cortar", variant="primary", visible=True, 
                                            elem_classes="custom-button4", interactive=True, min_width=50)
                                with gr.Column(scale=8):
                                    imp3_option = gr.Radio(label="Imputación",
                                                choices=["Interpolación lineal", 
                                                "Interpolación polinómica",
                                                "Media móvil (k=2)",
                                                "Media móvil (k=3 hacia atrás)",
                                                "Media móvil (k=3 centrada)"],
                                                value="Interpolación lineal", type="index", 
                                                show_label=False,
                                                elem_classes=["custom-radio", "radio-horizontal", "special-margin1"])
                            with gr.Row():
                                with gr.Column(min_width=250):
                                    rest3_button = gr.Button("Restaurar Serie", variant="primary", visible=True, 
                                            elem_classes="custom-button5", min_width=50, 
                                            interactive=False)
                                with gr.Column(scale=8):
                                    with gr.Row():
                                        gr.HTML("Año inicial de la serie", elem_classes="info-display-4")
                                        yinic3 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        gr.HTML("Año final de la serie", elem_classes="info-display-4")
                                        yfin3 = gr.Dropdown(show_label=False, choices=[], 
                                                            elem_classes="custom-dropdown-small",
                                                            interactive=True)
                                        err3 = gr.HTML("<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                                                    elem_classes="info-display-6")

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

            # region SECCION 2: DESCOMPOSICIÓN DE LAS TRES SERIES TEMPORALES (MÉTODO STL)
            with gr.Tab("Descomposición de las Series") as subtab_desc_series:
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
                            STL_desc1 = gr.HTML("Debe seleccionarse la Serie 1", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():                        
                                STL_graph1 = gr.Plot(show_label=False, visible=False)
                                STL_info1 = gr.HTML("Interpretación", visible=False)

                    with gr.Column():
                        with gr.Row():
                            STL_desc2 = gr.HTML("Debe seleccionarse la Serie 2", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                STL_graph2 = gr.Plot(show_label=False, visible=False)
                                STL_info2 = gr.HTML("Interpretación", visible=False)

                    with gr.Column():
                        with gr.Row():
                            STL_desc3 = gr.HTML("Debe seleccionarse la Serie 3", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                STL_graph3 = gr.Plot(show_label=False, visible=False)
                                STL_info3 = gr.HTML("Interpretación", visible=False)
            # endregion SECCION 2: DESCOMPOSICIÓN DE LAS TRES SERIES TEMPORALES

            # region SECCIÓN 3: DIFERENCIACIÓN DE LAS SERIES Y PRUEBA ADF
            with gr.Tab("Diferenciación de las Series") as subtab_dif_series:
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
                            ADF_desc1 = gr.HTML("Debe seleccionarse la Serie 1", elem_classes="info-display-3")
                        with gr.Row():
                            leveldiff1 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                        with gr.Row():
                            diff_graph1 = gr.Plot(show_label=False, visible=False)
                        with gr.Row():
                            with gr.Column():                        
                                ADF_info1 = gr.HTML("Interpretación", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ADF_desc2 = gr.HTML("Debe seleccionarse la Serie 2", elem_classes="info-display-3")
                        with gr.Row():
                            leveldiff2 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                        with gr.Row():
                            diff_graph2 = gr.Plot(show_label=False, visible=False)                    
                        with gr.Row():
                            with gr.Column():
                                ADF_info2 = gr.HTML("Interpretación", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ADF_desc3 = gr.HTML("Debe seleccionarse la Serie 3", elem_classes="info-display-3")
                        with gr.Row():
                            leveldiff3 = gr.Radio(label="Grado de Diferenciación", choices=[0, 1, 2, 3, 4], value=0, visible=False)
                        with gr.Row():
                            diff_graph3 = gr.Plot(show_label=False, visible=False)
                        with gr.Row():
                            with gr.Column():
                                ADF_info3 = gr.HTML("Interpretación", visible=False)
            # endregion SECCIÓN 3: DIFERENCIACIÓN DE LAS SERIES Y PRUEBA ADF

            # region SECCIÓN 4: CÁLCULO DE AUTOCORRELACIONES DE LAS SERIES
            with gr.Tab("ACF y PACF de las Series") as subtab_ACF_PACF:
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
                            ACF_desc1 = gr.HTML("Debe seleccionarse la Serie 1", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():                        
                                ACF_graph1 = gr.Plot(show_label=False, visible=False)
                                ACF_info1 = gr.HTML("Interpretación ACF", visible=False)
                                PACF_graph1 = gr.Plot(show_label=False, visible=False)
                                PACF_info1 = gr.HTML("Interpretación PACF", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ACF_desc2 = gr.HTML("Debe seleccionarse la Serie 2", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                ACF_graph2 = gr.Plot(show_label=False, visible=False)
                                ACF_info2 = gr.HTML("Interpretación ACF", visible=False)
                                PACF_graph2 = gr.Plot(show_label=False, visible=False)
                                PACF_info2 = gr.HTML("Interpretación PACF", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ACF_desc3 = gr.HTML("Debe seleccionarse la Serie 3", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                ACF_graph3 = gr.Plot(show_label=False, visible=False)
                                ACF_info3 = gr.HTML("Interpretación ACF", visible=False)
                                PACF_graph3 = gr.Plot(show_label=False, visible=False)
                                PACF_info3 = gr.HTML("Interpretación PACF", visible=False)
            # endregion SECCIÓN 4: CÁLCULO DE AUTOCORRELACIONES DE LAS SERIES

            # region SECCIÓN 5: APLICACIÓN DEL MODELO ARIMA
            with gr.Tab("Modelo ARIMA") as subtab_ARIMA:
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
                            ARIMA_desc1 = gr.HTML("Debe seleccionarse la Serie 1", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():                        
                                ARIMA_info1 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                                ARIMA_graph1 = gr.Plot(show_label=False, visible=False)
                                ARIMA_preds1 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                ARIMA_graph1_resids = gr.Plot(show_label=False, visible=False)
                                ARIMA_resids1 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ARIMA_desc2 = gr.HTML("Debe seleccionarse la Serie 2", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                ARIMA_info2 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                                ARIMA_graph2 = gr.Plot(show_label=False, visible=False)
                                ARIMA_preds2 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                ARIMA_graph2_resids = gr.Plot(show_label=False, visible=False)
                                ARIMA_resids2 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                    with gr.Column():
                        with gr.Row():
                            ARIMA_desc3 = gr.HTML("Debe seleccionarse la Serie 3", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                ARIMA_info3 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                                ARIMA_graph3 = gr.Plot(show_label=False, visible=False)
                                ARIMA_preds3 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                ARIMA_graph3_resids = gr.Plot(show_label=False, visible=False)
                                ARIMA_resids3 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)
            # endregion SECCIÓN 5: APLICACIÓN DEL MODELO ARIMA

            """
            # region SECCÓN 6: APLICACIÓN DEL MODELO AUTO-ARIMA
            with gr.Tab("Modelo AUTO-ARIMA") as subtab_AUTO_ARIMA:
                with gr.Row():
                    with gr.Column(elem_classes="custom-tab-2", scale=20): 
                        gr.HTML("&nbsp;&nbsp;5. APLICACIÓN DEL MODELO PREDICTIVO AUTO-ARIMA",
                                elem_classes="subtitle-text")
                    with gr.Column(min_width=150):
                        AUTO_ARIMA_button = gr.Button("Calcular", variant="primary", visible=True, 
                                                elem_classes="custom-button3")
                        
                with gr.Row(elem_classes="custom-tab"):
                    gr.HTML("El modelo <b>Auto-ARIMA</b> automatiza la selección óptima de los hiperparámetros (p, d, q) "
                                "mediante algoritmos de búsqueda <i>stepwise</i> que minimizan el Criterio de Información de "
                                "Akaike (AIC). Es ideal cuando la estructura de la serie es compleja o cuando se busca eliminar "
                                "la subjetividad en la interpretación de las gráficas ACF y PACF. Este enfoque garantiza un "
                                "equilibrio estadístico entre el ajuste del modelo y su parsimonia, evitando el sobreajuste "
                                "y optimizando la capacidad predictiva de forma computacional.",
                                elem_classes="info-display-2a")
            
                with gr.Row(elem_classes="custom-tab"):
                    with gr.Column():
                        with gr.Row():
                            AUTO_ARIMA_desc1 = gr.HTML("AUTO-ARIMA de la Serie 1", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():                        
                                AUTO_ARIMA_info1 = gr.HTML("Parámetros y Estadísticos de AUTO-ARIMA", visible=False)
                                AUTO_ARIMA_graph1 = gr.Plot(show_label=False, visible=False)
                                AUTO_ARIMA_preds1 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                AUTO_ARIMA_resids1 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                    with gr.Column():
                        with gr.Row():
                            AUTO_ARIMA_desc2 = gr.HTML("AUTO-ARIMA de la Serie 2", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                AUTO_ARIMA_info2 = gr.HTML("Parámetros y Estadísticos de ARIMA", visible=False)
                                AUTO_ARIMA_graph2 = gr.Plot(show_label=False, visible=False)
                                AUTO_ARIMA_preds2 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                AUTO_ARIMA_resids2 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)

                    with gr.Column():
                        with gr.Row():
                            AUTO_ARIMA_desc3 = gr.HTML("AUTO-ARIMA de la Serie 3", elem_classes="info-display-3")
                        with gr.Row():
                            with gr.Column():
                                AUTO_ARIMA_info3 = gr.HTML("Parámetros y Estadísticos de AUTO-ARIMA", visible=False)
                                AUTO_ARIMA_graph3 = gr.Plot(show_label=False, visible=False)
                                AUTO_ARIMA_preds3 = gr.HTML("Tabla de Valores Predichos", visible=False)
                                AUTO_ARIMA_resids3 = gr.HTML("Prueba Ljung-Box de Residuos", visible=False)
            # endregion SECCIÓN 6: APLICACIÓN DEL MODELO AUTO-ARIMA
            """

            # region SECCIÓN 6: TRANSFORMADAS DE FOURIER
            with gr.Tab("Transformada de Fourier") as subtab_Fourier:
                with gr.Row():
                    with gr.Column(elem_classes="custom-tab-2", scale=20): 
                        gr.HTML("&nbsp;&nbsp;6. APLICACIÓN DE LA TRANSFORMADA RÁPIDA DE FOURIER",
                                elem_classes="subtitle-text")
                    with gr.Column(min_width=150):
                        Fourier_button = gr.Button("Calcular", variant="primary", visible=True, 
                                                elem_classes="custom-button3")
                        
                with gr.Row(elem_classes="custom-tab"):
                    gr.HTML("Se aplica la Transformada Rápida de Fourier a cada una de las Series Temporales "
                            "para poder comparar las curvas en el dominio del período, de modo de detectar "
                            "comportamientos similares en la evolución de los cultivos.",
                            elem_classes="info-display-2a")

                with gr.Column(elem_classes="custom-tab"):
                    with gr.Row():
                        Fourier_desc1 = gr.HTML("Debe seleccionarse la Serie 1", elem_classes="info-display-3")
                        Fourier_desc2 = gr.HTML("Debe seleccionarse la Serie 2", elem_classes="info-display-3")
                        Fourier_desc3 = gr.HTML("Debe seleccionarse la Serie 3", elem_classes="info-display-3")
                    
                    with gr.Row(visible=True):
                        Fourier_series_orig = gr.Plot(show_label=False, visible=False, elem_classes="custom-graf-panel")

                    with gr.Row(visible=True):
                        Fourier_series_dif = gr.Plot(show_label=False, visible=False, elem_classes="custom-graf-panel")

                    with gr.Row():
                        with gr.Column():                        
                            Fourier_graph1 = gr.Plot(show_label=False, visible=False)
                            Fourier_info1 = gr.HTML("Estadísticos Serie 1", visible=False)
                        with gr.Column():                        
                            Fourier_graph2 = gr.Plot(show_label=False, visible=False)
                            Fourier_info2 = gr.HTML("Estadísticos Serie 2", visible=False)
                        with gr.Column():                        
                            Fourier_graph3 = gr.Plot(show_label=False, visible=False)
                            Fourier_info3 = gr.HTML("Estadísticos Serie 3", visible=False)


                # enderegion SECCIÓN 6: TRANSFORMADAS DE FOURIER


            cult1.select(
                fn = tab_ST_on_cult_select,
                inputs = [cult1, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_state_1, dataset_filter_state_1, dataset_diff_state_1,
                            cult1, prov1, dep1, var1,
                            inicio1, final1, regs1, regsno1,
                            imp1_button, cut1_button, imp1_option, 
                            rest1_button, yinic1, yfin1, err1,
                            tend1_area, tend1,
                            dataset_filter_state_1, dataset_diff_state_1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, level_diff_state_1, diff_graph1, ADF_info1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1,
                            ARIMA_preds1, ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            cult2.select(
                fn = tab_ST_on_cult_select,
                inputs = [cult2, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_state_2, dataset_filter_state_2, dataset_diff_state_2,
                            cult2, prov2, dep2, var2,
                            inicio2, final2, regs2, regsno2,
                            imp2_button, cut2_button, imp2_option, 
                            rest2_button, yinic2, yfin2, err2, 
                            tend2_area, tend2, 
                            dataset_filter_state_2, dataset_diff_state_2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, level_diff_state_2, diff_graph2, ADF_info2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2,
                            ARIMA_preds2, ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            cult3.select(
                fn = tab_ST_on_cult_select,
                inputs = [cult3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_state_3, dataset_filter_state_3, dataset_diff_state_3,
                            cult3, prov3, dep3, var3,
                            inicio3, final3, regs3, regsno3,
                            imp3_button, cut3_button, imp3_option, 
                            rest3_button, yinic3, yfin3, err3,
                            tend3_area, tend3,
                            dataset_filter_state_3, dataset_diff_state_3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, level_diff_state_3, diff_graph3, ADF_info3,
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3,
                            ARIMA_preds3, ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )

            prov1.select(
                fn = tab_ST_on_prov_select,
                inputs = [dataset_state_1, cult1, prov1, var1, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            dep1, inicio1, final1, regs1, regsno1,
                            imp1_button, cut1_button, rest1_button, 
                            yinic1, yfin1, err1,
                            tend1_area, tend1, 
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            prov2.select(
                fn = tab_ST_on_prov_select,
                inputs = [dataset_state_2, cult2, prov2, var2, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2, dataset_diff_state_2,
                            dep2, inicio2, final2, regs2, regsno2,
                            imp2_button, cut2_button, rest2_button, 
                            yinic2, yfin2, err2,
                            tend2_area, tend2, 
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )
            
            prov3.select(
                fn = tab_ST_on_prov_select,
                inputs = [dataset_state_3, cult3, prov3, var3,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3, dataset_diff_state_3,
                            dep3, inicio3, final3, regs3, regsno3, 
                            imp3_button, cut3_button, rest3_button, 
                            yinic3, yfin3, err3,
                            tend3_area, tend3, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3,
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )
            
            dep1.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_1, cult1, prov1, dep1, var1,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            inicio1, final1, regs1, regsno1, 
                            imp1_button, cut1_button, rest1_button, 
                            yinic1, yfin1, err1,
                            tend1_area, tend1, 
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            dep2.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_2, cult2, prov2, dep2, var2, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2, dataset_diff_state_2, 
                            inicio2, final2, regs2, regsno2, 
                            imp2_button, cut2_button, rest2_button, 
                            yinic2, yfin2, err2,
                            tend2_area, tend2, 
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            dep3.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_3, cult3, prov3, dep3, var3,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3, dataset_diff_state_3, 
                            inicio3, final3, regs3, regsno3, 
                            imp3_button, cut3_button, rest3_button, 
                            yinic3, yfin3, err3,
                            tend3_area, tend3, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )

            var1.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_1, cult1, prov1, dep1, var1,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1, dataset_diff_state_1,
                            inicio1, final1, regs1, regsno1, 
                            imp1_button, cut1_button, rest1_button, 
                            yinic1, yfin1, err1,
                            tend1_area, tend1, 
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )
            
            var2.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_2, cult2, prov2, dep2, var2, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2, dataset_diff_state_2,
                            inicio2, final2, regs2, regsno2, 
                            imp2_button, cut2_button, rest2_button, 
                            yinic2, yfin2, err2,
                            tend2_area, tend2, 
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            var3.select(
                fn = tab_ST_on_option_select,
                inputs = [dataset_state_3, cult3, prov3, dep3, var3, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3, dataset_diff_state_3,
                            inicio3, final3, regs3, regsno3, 
                            imp3_button, cut3_button, rest3_button, 
                            yinic3, yfin3, err3,
                            tend3_area, tend3, 
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
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

            leveldiff1.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph1, ADF_info1, level_diff_state_1,
                            ACF_graph1, ACF_info1, PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            leveldiff2.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph2, ADF_info2, level_diff_state_2,
                            ACF_graph2, ACF_info2, PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            leveldiff3.change(
                fn = tab_ST_on_level_change,
                outputs = [diff_graph3, ADF_info3, level_diff_state_3,
                            ACF_graph3, ACF_info3, PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )

            imp1_button.click(
                fn = tab_ST_imputar_df,
                inputs = [dataset_filter_state_1, imp1_option, var1, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1, regs1, regsno1, 
                            yinic1, yfin1, err1,
                            imp1_button, rest1_button,
                            tend1_area, tend1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            imp2_button.click(
                fn = tab_ST_imputar_df,
                inputs = [dataset_filter_state_2, imp2_option, var2, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2, regs2, regsno2, 
                            yinic2, yfin2, err2,
                            imp2_button, rest2_button,
                            tend2_area, tend2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            imp3_button.click(
                fn = tab_ST_imputar_df,
                inputs = [dataset_filter_state_3, imp3_option, var3, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3, regs3, regsno3, 
                            yinic3, yfin3, err3,
                            imp3_button, rest3_button,
                            tend3_area, tend3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )

            rest1_button.click(
                fn = tab_ST_restaurar_df,
                inputs = [dataset_state_1, cult1, prov1, dep1, var1, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1,
                            inicio1, final1, regs1, regsno1, yinic1, yfin1, err1,
                            imp1_button, cut1_button, rest1_button, 
                            tend1_area, tend1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )

            rest2_button.click(
                fn = tab_ST_restaurar_df,
                inputs = [dataset_state_2, cult2, prov2, dep2, var2, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2,
                            inicio2, final2, regs2, regsno2, yinic2, yfin2, err2,
                            imp2_button, cut2_button, rest2_button,
                            tend2_area, tend2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            rest3_button.click(
                fn = tab_ST_restaurar_df,
                inputs = [dataset_state_3, cult3, prov3, dep3, var3, 
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3,
                            inicio3, final3, regs3, regsno3, yinic3, yfin3, err3,
                            imp3_button, cut3_button, rest3_button,
                            tend3_area, tend3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
            )

            cut1_button.click(
                fn = tab_ST_cortar_df,
                inputs = [dataset_filter_state_1, var1, yinic1, yfin1,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_1, inicio1, final1, regs1, regsno1,
                            yinic1, yfin1, err1,
                            cut1_button, rest1_button,
                            tend1_area, tend1,
                            STL_desc1, STL_graph1, STL_info1,
                            ADF_desc1, leveldiff1, diff_graph1, ADF_info1,
                            level_diff_state_1,
                            ACF_desc1, ACF_graph1, ACF_info1,
                            PACF_graph1, PACF_info1,
                            ARIMA_p_1, ARIMA_q_1,
                            ARIMA_desc1, ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            Fourier_desc1, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1]
            )
            
            cut2_button.click(
                fn = tab_ST_cortar_df,
                inputs = [dataset_filter_state_2, var2, yinic2, yfin2,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_2, inicio2, final2, regs2, regsno2,
                            yinic2, yfin2, err2,
                            cut2_button, rest2_button,
                            tend2_area, tend2,
                            STL_desc2, STL_graph2, STL_info2,
                            ADF_desc2, leveldiff2, diff_graph2, ADF_info2,
                            level_diff_state_2,
                            ACF_desc2, ACF_graph2, ACF_info2,
                            PACF_graph2, PACF_info2,
                            ARIMA_p_2, ARIMA_q_2,
                            ARIMA_desc2, ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            Fourier_desc2, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph2, Fourier_info2]
            )

            cut3_button.click(
                fn = tab_ST_cortar_df,
                inputs = [dataset_filter_state_3, var3, yinic3, yfin3,
                            graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs = [dataset_filter_state_3, inicio3, final3, regs3, regsno3,
                            yinic3, yfin3, err3,
                            cut3_button, rest3_button,
                            tend3_area, tend3,
                            STL_desc3, STL_graph3, STL_info3,
                            ADF_desc3, leveldiff3, diff_graph3, ADF_info3,
                            level_diff_state_3, 
                            ACF_desc3, ACF_graph3, ACF_info3,
                            PACF_graph3, PACF_info3,
                            ARIMA_p_3, ARIMA_q_3,
                            ARIMA_desc3, ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3,
                            Fourier_desc3, Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph3, Fourier_info3]
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
                outputs = [dataset_ARIMA_state_1,
                            ARIMA_info1, ARIMA_graph1, ARIMA_preds1,
                            ARIMA_graph1_resids, ARIMA_resids1,
                            dataset_ARIMA_state_2,
                            ARIMA_info2, ARIMA_graph2, ARIMA_preds2,
                            ARIMA_graph2_resids, ARIMA_resids2,
                            dataset_ARIMA_state_3,
                            ARIMA_info3, ARIMA_graph3, ARIMA_preds3,
                            ARIMA_graph3_resids, ARIMA_resids3]
            )

            Fourier_button.click(
                fn = tab_ST_FOURIER,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            dataset_diff_state_1, dataset_diff_state_2, dataset_diff_state_3,
                            level_diff_state_1, level_diff_state_2, level_diff_state_3,
                            cult1, cult2, cult3, prov1, prov2, prov3,
                            dep1, dep2, dep3, var1, var2, var3],
                outputs = [Fourier_series_orig, Fourier_series_dif,
                            Fourier_graph1, Fourier_info1,
                            Fourier_graph2, Fourier_info2,
                            Fourier_graph3, Fourier_info3]
            )

            """
            AUTO_ARIMA_button.click(
                fn = tab_ST_AUTO_ARIMA_all,
                inputs = [dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                            var1, var2, var3],
                outputs = [dataset_AUTO_ARIMA_state_1,
                            AUTO_ARIMA_info1, AUTO_ARIMA_graph1, AUTO_ARIMA_preds1,
                            AUTO_ARIMA_resids1,
                            dataset_AUTO_ARIMA_state_2,
                            AUTO_ARIMA_info2, AUTO_ARIMA_graph2, AUTO_ARIMA_preds2,
                            AUTO_ARIMA_resids2,
                            dataset_AUTO_ARIMA_state_3,
                            AUTO_ARIMA_info3, AUTO_ARIMA_graph3, AUTO_ARIMA_preds3,
                            AUTO_ARIMA_resids3]
            )
            """
   
   
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
