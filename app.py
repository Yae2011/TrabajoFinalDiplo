import gradio as gr
import pandas as pd
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
import seaborn as sns


# region CONSTANTES Y DICCIONARIOS
# --- Constantes ---
DATA_PATH = "./Datasets"
FILE_MAP = {
    "Por Curso": "Matricula 2011-2024.csv",
    # "Por Edad": "Matricula por Edad 2011-2024.csv",
    "Por Población": "Poblacion 2011-2024.csv",
    "Por Trayectoria": "Trayectoria 2011-2024.csv"
}
KEY_COLUMNS = ['periodo', 'provincia', 'departamento', 'sector', 'ambito']
MIN_REG = 14 # Cantidad mínima de registros para cada serie temporal (serie 2011-2024 = 14 registros anuales)

# Se cargan las descripciones de las variables de los datasets en un diccionario
# para títulos de gráficos de evolución de matrícula
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
        try:
             df = pd.read_csv(path, encoding='latin1', sep=';')
             return df, list(df['provincia'].unique())
        except:
             return pd.DataFrame(), [f"Error cargando: {e}"]
# endregion


# region FUNCIONES PARA LA PESTAÑA "EDA"
def tab_EDA_on_load(dataset_type, automatico, interactivo):
    df, provincias = load_data(dataset_type)
    
    if df.empty:
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
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
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
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
    - indicador: nombre original de columna de matrícula
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
    titulo = f"CANTIDAD DE ALUMNOS MATRICULADOS: {dict_nlargos[indicador].upper()}"
    
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
                x=x_data, y=y_data, mode='lines+markers', name='Matrícula',
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
                tick0=2011, dtick=1, range=[2010.5, 2024.5],
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
                ax.plot(x_data, y_data, label="Matrícula", marker='o', linewidth=3, 
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
            ax.set_xticks(range(2011, 2025))
            ax.set_xlim(2010.5, 2024.5)
            # ax.set_xlabel("Año")
            #   Qax.set_ylabel("Estudiantes")
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
        ax.set_xlabel("MATRÍCULA")
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
        ax.set_xlabel("MATRÍCULA")
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

def get_filtered_subset_2(df, provincia, departamento, sector, ambito, key_columns, agrupar_detalles=True):

    if df is None or df.empty:
        return pd.DataFrame() # Retorna DF vacío si no hay datos
    
    if not provincia or not departamento:
        return pd.DataFrame()

    # Filtrado inicial
    res_df = df[(df['provincia'] == provincia) & (df['departamento'] == departamento)].copy()
    
    # Aplicación de filtros condicionales
    if sector != 'Ambos':
        res_df = res_df[res_df['sector'] == sector]
        
    if ambito != 'Ambos':
        res_df = res_df[res_df['ambito'] == ambito]

    if res_df.empty:
        return pd.DataFrame(columns=df.columns) # Retorna estructura original vacía

    # Si agrupar_detalles es True, forzamos la etiqueta 'Ambos' en las filas
    if agrupar_detalles:
        if sector == 'Ambos' and 'sector' in res_df.columns:
            res_df['sector'] = 'Ambos'
        if ambito == 'Ambos' and 'ambito' in res_df.columns:
            res_df['ambito'] = 'Ambos'

    # Definición de dimensiones de agrupación
    group_cols = list(key_columns)
    
    # Añadimos dimensiones opcionales si existen en el DataFrame
    for col in ['grado', 'sector', 'ambito']:
        if col in res_df.columns and col not in group_cols:
            group_cols.append(col)
    
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

def get_filtered_subset(df, provincia, departamento, sector, ambito, key_columns, agrupar_detalles=True, min_reg=0):

    if df is None or df.empty:
        return pd.DataFrame() # Retorna DF vacío si no hay datos
    
    if not provincia or not departamento:
        return pd.DataFrame()

    # Filtrado inicial
    res_df = df[(df['provincia'] == provincia) & (df['departamento'] == departamento)].copy()
    
    # Aplicación de filtros condicionales
    if sector != 'Ambos':
        res_df = res_df[res_df['sector'] == sector]
        
    if ambito != 'Ambos':
        res_df = res_df[res_df['ambito'] == ambito]

    if res_df.empty:
        # return pd.DataFrame(columns=df.columns) # Retorna estructura original vacía
        return pd.DataFrame()

    # Si agrupar_detalles es True, forzamos la etiqueta 'Ambos' en las filas
    if agrupar_detalles:
        if sector == 'Ambos' and 'sector' in res_df.columns:
            res_df['sector'] = 'Ambos'
        if ambito == 'Ambos' and 'ambito' in res_df.columns:
            res_df['ambito'] = 'Ambos'

    # Definición de dimensiones de agrupación
    group_cols = list(key_columns)
    
    # Añadimos dimensiones opcionales si existen en el DataFrame
    for col in ['grado', 'sector', 'ambito']:
        if col in res_df.columns and col not in group_cols:
            group_cols.append(col)
    
    # Identificación de métricas (solo numéricas)
    numeric_cols = [col for col in res_df.columns 
                    if col not in group_cols and pd.api.types.is_numeric_dtype(res_df[col])]
    
    # Agregación
    final_df = res_df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    
    # Limpieza final para Gradio
    # Gradio a veces tiene problemas con tipos de datos complejos o NaNs en la visualización
    final_df = final_df.fillna(0)
    
    # Verificación del umbral mínimo de registros solicitado
    if len(final_df) < min_reg: # Comprueba si el tamaño del DF es menor al umbral
        # return pd.DataFrame(columns=final_df.columns)
        return pd.DataFrame()

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

    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        info_text = f" MATRÍCULA {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): SIN REGISTROS"
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
    info_text = f" MATRÍCULA {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): {len(filtered)} REGISTROS  -  {len(cols_to_show)} CAMPOS"
   
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
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
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

def tab_ST_on_mat_change(dataset_type):
    df, provincias = load_data(dataset_type)

    if df.empty:
        msg = "Sin datos"
        return df, gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), gr.Plot(visible=False), \
                gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), gr.Plot(visible=False), \
                gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), gr.Plot(visible=False), \
                msg, msg, msg, \
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                msg, msg, msg, \
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)                
    
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

    # Se filtra el dataset de matrícula
    sector = "Ambos"
    ambito = "Ambos"
    filtered = get_filtered_subset(df, prov_first, dpto_first, sector, ambito, 
                                   KEY_COLUMNS, True, min_reg=MIN_REG)

    # Se genera el gráfico para el primer indicador
    # mmov =  0 < mm < 4
    # tipo = mm + 1
    # graph = tab_EDA_create_evolution_graph(filtered, indicadores_originales[0], serie, mg,
    #                                       tend, mmov, sd, tipo)

    # Al actualizar el dataset, se muestra la primera provincia, el primer departamento,
    # sector = "Ambos", ambiente="Ambos", el primer indicador y el gráfico correspodiente.
    # Se hace para las tres series temporales
    msg = ("<b>"
           f"PROVINCIA: {prov_first.upper()}<br>"
           f"DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"SECTOR: {sector.upper()} - ÁMBITO: {ambito.upper()}<br>"
           f"INDICADOR: {dict_nlargos[indicadores_originales[0]].upper()}"
           "</b>")
    return df, gr.update(choices=provincias_sorted, value=prov_first), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
                gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
                gr.update(choices=indicadores, value=indicador_first), \
                gr.Plot(visible=False), \
                gr.update(choices=provincias_sorted, value=prov_first), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
                gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
                gr.update(choices=indicadores, value=indicador_first), \
                gr.Plot(visible=False), \
                gr.update(choices=provincias_sorted, value=prov_first), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
                gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
                gr.update(choices=indicadores, value=indicador_first), \
                gr.Plot(visible=False), \
                filtered, filtered, filtered, \
                msg, msg, msg, \
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                msg, msg, msg, \
                gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
def tab_ST_on_prov_change(df, provincia, sector, ambito, indicador):
    
    # Se arma el listado ordenado de departamentos de la provincia
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    
    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de matrícula
    filtered = get_filtered_subset(df, provincia, dpto_first, sector, ambito, 
                                   KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None, None, gr.Plot(visible=False), None, gr.Plot(visible=False), gr.update(visible=False)
    
    # Se genera el gráfico para el primer indicador
    # mmov =  0 < mm < 4
    # tipo = mm + 1
    # graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, tipo)

    msg = ("<b>"
           f"PROVINCIA: {provincia.upper()}<br>DEPARTAMENTO: {dpto_first.upper()}<br>"
           f"SECTOR: {sector.upper()} - ÁMBITO: {ambito.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return gr.update(choices=dptos_sorted, value=dpto_first), filtered, gr.Plot(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False)

def tab_ST_on_dep_change(df, provincia, departamento, sector, ambito, indicador):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de matrícula
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None, gr.Plot(visible=False), None, gr.Plot(visible=False), gr.update(visible=False)
    
    # Se genera el gráfico para el primer indicador
    # mmov =  0 < mm < 4
    # tipo = mm + 1
    # graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, tipo)

    msg = ("<b>"
           f"PROVINCIA: {provincia.upper()}<br>DEPARTAMENTO: {departamento.upper()}<br>"
           f"SECTOR: {sector.upper()} - ÁMBITO: {ambito.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return filtered, gr.Plot(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False)

def tab_ST_on_option_change(df, provincia, departamento, sector, ambito, indicador):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de matrícula
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None, gr.Plot(visible=False), None, gr.Plot(visible=False), gr.update(visible=False)
    
    # Se genera el gráfico para el primer indicador
    # mmov =  0 < mm < 4
    # tipo = mm + 1
    # graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie, mg, tend, mmov, sd, tipo)

    msg = ("<b>"
           f"PROVINCIA: {provincia.upper()}<br>DEPARTAMENTO: {departamento.upper()}<br>"
           f"SECTOR: {sector.upper()} - ÁMBITO: {ambito.upper()}<br>"
           f"INDICADOR: {dict_nlargos[ind_orig].upper()}"
           "</b>")
    return filtered, gr.Plot(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False), \
            msg, gr.Plot(visible=False), gr.update(visible=False)

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
                                            tipo, interactivo=False, h_estatico=4.13)
    graph2 = tab_EDA_create_evolution_graph(filtered2, ind_orig2, serie, mg, tend, mmov, sd, 
                                            tipo, interactivo=False, h_estatico=4.13)
    graph3 = tab_EDA_create_evolution_graph(filtered3, ind_orig3, serie, mg, tend, mmov, sd, 
                                            tipo, interactivo=False, h_estatico=4.13)

    return gr.Plot(value=graph1, visible=True), gr.Plot(value=graph2, visible=True), \
            gr.Plot(value=graph3, visible=True)

def tab_ST_stl_decomp(df, indicador):
    """
    Se aplica la descomposición STL (Seasonal-Trend decomposition using LOESS) 
    debido a que se trabaja con una serie corta de datos (n = 14), donde no resulta 
    conveniente utilizar modelos clásicos de medias móviles ni modelos predictivos complejos 
    como ARIMA/SARIMA. Este enfoque se destaca por tres ventajas clave:
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
    
    titulo = f"DESCOMPOSICIÓN STL DE LA MATRÍCULA: {dict_nlargos[ind_orig].upper()}"
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
        tickmode='linear', 
        tick0=2011, 
        dtick=1, 
        range=[2010.5, 2024.5],
        tickfont=dict(size=8, color='black', family='Arial Black'),
        tickformat='d',
        showticklabels=True # Fuerza la visibilidad en todos los subgráficos
    )
    fig.update_yaxes(
        tickfont=dict(size=8, color='black', family='Arial Black'),
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
        desc_ruido = "moderada. Existen fluctuaciones aleatorias o eventos puntuales que afectan la matrícula."
    else:
        desc_ruido = "alta. La serie es altamente impredecible; hay una fuerte influencia de variables no capturadas por la tendencia."

    reporte = (f"<b>Variabilidad No Explicada (Ruido): </b>"
                f"{variabilidad_residuo:.2f}% respecto a la media. La variabilidad es {desc_ruido}<br>"
                f"<b>Tendencia: </b>"
                f"La serie presenta {interpretacion_tendencia} de la matrícula.<br>"
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

def tab_ST_autocorr(df, indicador):
    """
    Calcula función de autocorrelación, genera un gráfico interactivo y analiza la ciclicidad.
    - df: dataset filtrado con columnas con nombres originales ['periodo', indicadores]
    - indicador: nombre corto del indicador
    """
    # Se convierte el nombre corto del  "indicador" a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)
    df = df.sort_values('periodo').reset_index(drop=True)
    serie = df[ind_orig]

    n_obs = len(df)
    lags = 7 # Desfases (lags); para 14 datos, lags=7 para mantener potencia estadística
    valores_acf = acf(serie, nlags=lags)
    
    # Identificación del lag más significativo; se excluye lag=0 que siempre es 1
    lag_max = np.argmax(valores_acf[1:]) + 1
    val_max = valores_acf[lag_max]
    
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
        tickmode='linear', 
        tick0=0, 
        dtick=1, 
        # range=[0, 7],
        tickfont=dict(size=12, color='black', family='Arial Black'),
        tickformat='d',
    )
    fig.update_yaxes(
        tickfont=dict(size=12, color='black', family='Arial Black'),
    )
    fig.update_layout(
        # title=f"{dict_nlargos[ind_orig].upper()}",
        # height=400,
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
        f"Este valor {es_significativo} bajo un intervalo de confianza del 95%."
        )
    
    return fig, reporte

def tab_ST_autocorr_all(df1, df2, df3, var1, var2, var3):

    fig1, desc1 = tab_ST_autocorr(df1, var1)
    fig2, desc2 = tab_ST_autocorr(df2, var2)
    fig3, desc3 = tab_ST_autocorr(df3, var3)

    return gr.update(value = fig1, visible = True), \
            gr.update(value = desc1, visible = True), \
            gr.update(value = fig2, visible = True), \
            gr.update(value = desc2, visible = True), \
            gr.update(value = fig3, visible = True), \
            gr.update(value = desc3, visible = True)

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

with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Almacenamiento para el dataset elegido, en pestaña EDA
    dataset_state = gr.State(pd.DataFrame())
    # Almacenamiento para el dataset elegido y filtrado por campos clave, en pestaña EDA
    dataset_filter_state = gr.State(pd.DataFrame())
    # Almacenamiento para los tres datasets filtrados, en pestaña ST
    dataset_filter_state_1 = gr.State(pd.DataFrame())
    dataset_filter_state_2 = gr.State(pd.DataFrame())
    dataset_filter_state_3 = gr.State(pd.DataFrame())
    

    gr.Row(elem_classes="header-tab")
    

    ###### ESTRUCTURA DE PESTAÑAS
    with gr.Tabs():


        ###### PESTAÑA INICIO
        with gr.Tab("Inicio"):
            with gr.Row():
                with gr.Column(scale=8, elem_classes="portrait-bg-video"): # elem_classes="portrait-bg-1"):
                    gr.HTML(portada_video)
                    gr.HTML("ANÁLISIS DE LA MATRÍCULA ESCOLAR<br>"
                            "DE LA REPÚBLICA ARGENTINA,<br>"
                            "EN EL PERÍODO 2011-2024,<br>"
                            "PARA TODAS LAS JURISDICCIONES<br>"
                            "EDUCATIVAS DEL PAÍS", elem_classes="portrait-title")
                with gr.Column(scale=2, elem_classes="portrait-bg-2"):
                    gr.HTML("Aplicación de algoritmos de Machine Learning "
                            "a las Bases de Datos Abiertas de la Secretaría de Educación "
                            "del Ministerio de Capital Humano de la República Argentina, "
                            "para el análisis de los indicadores de matrícula escolar, "
                            "para todas las provincias y CABA y sus respectivos "
                            "departamentos, partidos o comunas.",
                            elem_classes="portrait-subtitle")
        

        ###### PESTAÑA PROCESO
        with gr.Tab("Proceso"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FLUJOGRAMA DEL PROCESO", elem_classes="title-text")
            
        
        ###### PESTAÑA DASHBOARD
        with gr.Tab("Dashboard"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;INDICADORES DE MATRÍCULA", elem_classes="title-text")
        
        
        ###### PESTAÑA EDA
        with gr.Tab("EDA") as tab_EDA:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;VISUALIZACIÓN DE DATOS DE LAS JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
            with gr.Row():
                with gr.Column(min_width=180, # scale=1, 
                               elem_classes="custom-tab"):
                    tipo_matricula = gr.Radio(
                        label="Tipo de Matrícula", 
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
                    
                    # Componente que contiene toda la información a mostrar y que alterna visble = True/False
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


            tipo_matricula.change(
                fn=tab_EDA_on_dataset_change,
                inputs=[tipo_matricula, chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            provincia.change(
                fn=tab_EDA_on_provincia_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia, 
                        chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            departamento.change(
                fn=tab_EDA_on_departamento_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia, departamento, 
                        chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            sector.change(
                fn=tab_EDA_on_opcion_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
            
            ambito.change(
                fn=tab_EDA_on_opcion_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            btn_mostrar.click(
                fn=tab_EDA_show_data,
                inputs=[dataset_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            indicador.change(
                fn=tab_EDA_create_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution, output_plot_histogram, output_plot_normal_dist]
            )

            btn_anterior.click(
                fn=tab_EDA_create_prev_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )
            
            btn_siguiente.click(
                fn=tab_EDA_create_next_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )

            chk_serie.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution]
            )

            chk_mg.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution]
            )

            chk_tend.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution]
            )

            chk_mm.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution]
            )

            chk_sd.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_evolution]
            )

            chk_mostrar.select(
                fn=tab_EDA_on_checkbox,
                inputs=[tipo_matricula, provincia, departamento, sector, ambito, 
                        chk_mostrar, chk_interactivo],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )

            chk_interactivo.select(
                fn=tab_EDA_graph_interactive,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd, chk_interactivo],
                outputs=[output_plot_box, output_plot_evolution]
            )

            tab_EDA.select(
                fn=tab_EDA_on_load, 
                inputs=[tipo_matricula, chk_mostrar, chk_interactivo], 
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar,
                            chk_serie, chk_mg, chk_tend, chk_mm, chk_sd]
            )
        

        ###### PESTAÑA SERIES TEMPORALES
        with gr.Tab("Series Temporales") as tab_ST:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;COMPARACIÓN DE SERIES TEMPORALES", elem_classes="title-text")
            
            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;1. SELECCIÓN DE LAS SERIES TEMPORALES A COMPARAR", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    graph_button = gr.Button("Graficar", variant="primary", visible=True, 
                                               elem_classes="custom-button3")
            
            with gr.Row():
                with gr.Column(min_width=180):
                    with gr.Row(elem_classes="custom-tab"):
                        mat = gr.Radio(label="Tipo de Matrícula", 
                            choices=["Por Curso", "Por Población", "Por Trayectoria"],
                            value="Por Curso", elem_classes="custom-radio")
                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column():
                            gr.HTML("Opciones de Gráficos", elem_classes="title-group")
                            graph_serie = gr.Checkbox(label="Serie", value=True, elem_classes="custom-checkbox")
                            graph_mg = gr.Checkbox(label="Media Global", value=True, elem_classes="custom-checkbox")
                            graph_tend = gr.Checkbox(label="Tendencia", value=True, elem_classes="custom-checkbox")
                            graph_mm = gr.Radio(label="Media Móvil", choices=["No", "k = 2 atrás",
                                                            "k = 3 atrás", "k = 3 centrado"],
                                                            value="k = 3 centrado", type="index", elem_classes="custom-radio")
                            graph_sd = gr.Checkbox(label="SD Móvil", value=True, elem_classes="custom-checkbox")

                with gr.Column(scale=20):
                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("Provincia")
                                prov1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Departamento")
                                dep1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Sector")
                                sec1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Ámbito")
                                amb1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Indicador")
                                var1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                        with gr.Column(scale=20):
                            tend1 = gr.Plot(show_label=False, visible=False)

                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("Provincia")
                                prov2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Departamento")
                                dep2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Sector")
                                sec2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Ámbito")
                                amb2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Indicador")
                                var2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                        with gr.Column(scale=20):
                            tend2 = gr.Plot(show_label=False, visible=False)

                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column(min_width=250):
                            with gr.Row():
                                gr.HTML("Provincia")
                                prov3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Departamento")
                                dep3= gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Sector")
                                sec3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Ámbito")
                                amb3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                            with gr.Row():
                                gr.HTML("Indicador")
                                var3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown-small")
                        with gr.Column(scale=20):
                            tend3 = gr.Plot(show_label=False, visible=False)

            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;2. DESCOMPOSICIÓN DE LAS SERIES - MÉTODO STL (SEASONAL-TREND DECOMPOSITION USING LOESS) PARA SERIES CORTAS", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    stats_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")
            
            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        desc1 = gr.HTML("Descomposición de la Serie 1")
                    with gr.Row():
                        with gr.Column():                        
                            STL_graph1 = gr.Plot(show_label=False, visible=False)
                            STL_info1 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        desc2 = gr.HTML("Descomposición de la Serie 2")
                    with gr.Row():
                        with gr.Column():
                            STL_graph2 = gr.Plot(show_label=False, visible=False)
                            STL_info2 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        desc3 = gr.HTML("Descomposición de la Serie 3")
                    with gr.Row():
                        with gr.Column():
                            STL_graph3 = gr.Plot(show_label=False, visible=False)
                            STL_info3 = gr.HTML("Interpretación", visible=False)


            with gr.Row():
                with gr.Column(elem_classes="custom-tab-2", scale=20):    
                    gr.HTML("&nbsp;&nbsp;3. AUTOCORRELACIÓN DE LAS SERIES", 
                            elem_classes="subtitle-text")
                with gr.Column(min_width=150):
                    autocor_button = gr.Button("Calcular", variant="primary", visible=True, 
                                               elem_classes="custom-button3")

            with gr.Row(elem_classes="custom-tab"):
                with gr.Column():
                    with gr.Row():
                        desc1a = gr.HTML("Autocorrelación de la Serie 1")
                    with gr.Row():
                        with gr.Column():                        
                            auto_graph1 = gr.Plot(show_label=False, visible=False)
                            auto_info1 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        desc2a = gr.HTML("Autocorrelación de la Serie 2")
                    with gr.Row():
                        with gr.Column():
                            auto_graph2 = gr.Plot(show_label=False, visible=False)
                            auto_info2 = gr.HTML("Interpretación", visible=False)

                with gr.Column():
                    with gr.Row():
                        desc3a = gr.HTML("Autocornrelació de la Serie 3")
                    with gr.Row():
                        with gr.Column():
                            auto_graph3 = gr.Plot(show_label=False, visible=False)
                            auto_info3 = gr.HTML("Interpretación", visible=False)









            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;4. TEST DE DICKEY-FÜLLER AUMENTADO (ADF)", elem_classes="subtitle-text")
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;5. SELECCIÓN DEL MODELO Y VALORES DE SUS HIPERPARÁMETROS APLICANDO UN ALGORITMO GENÉTICO", elem_classes="subtitle-text")  
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;6. PREDICCIÓN DE LAS SERIES", elem_classes="subtitle-text")
           
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;7. TRANSFORMADA DE FOURIER PARA LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")              


            mat.change(
                fn=tab_ST_on_mat_change,
                inputs=[mat],
                outputs=[dataset_state, prov1, dep1, sec1, amb1, var1, tend1,
                         prov2, dep2, sec2, amb2, var2, tend2,
                         prov3, dep3, sec3, amb3, var3, tend3,
                         dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                         desc1, desc2, desc3,
                         STL_graph1, STL_graph2, STL_graph3,
                         STL_info1, STL_info2, STL_info3,
                         desc1a, desc2a, desc3a,
                         auto_graph1, auto_graph2, auto_graph3,
                         auto_info1, auto_info2, auto_info3]
            )
            
            prov1.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov1, sec1, amb1, var1],
                outputs=[dep1, dataset_filter_state_1, tend1,
                         desc1, STL_graph1, STL_info1,
                         desc1a, auto_graph1, auto_info1]
            )

            prov2.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov2, sec2, amb2, var2],
                outputs=[dep2, dataset_filter_state_2, tend2, 
                         desc2, STL_graph2, STL_info2,
                         desc2a, auto_graph2, auto_info2]
            )
            
            prov3.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov3, sec3, amb3, var3],
                outputs=[dep3, dataset_filter_state_3, tend3, 
                         desc3, STL_graph3, STL_info3,
                         desc3a, auto_graph3, auto_info3]
            )
            
            dep1.change(
                fn=tab_ST_on_dep_change,
                inputs=[dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[dataset_filter_state_1, tend1, 
                         desc1, STL_graph1, STL_info1,
                         desc1a, auto_graph1, auto_info1]
            )

            dep2.change(
                fn=tab_ST_on_dep_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[dataset_filter_state_2, tend2,
                         desc2, STL_graph2, STL_info2,
                         desc2a, auto_graph2, auto_info2]
            )

            dep3.change(
                fn=tab_ST_on_dep_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[dataset_filter_state_3, tend3, 
                         desc3, STL_graph3, STL_info3,
                         desc3a, auto_graph3, auto_info3]
            )

            sec1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[dataset_filter_state_1, tend1, 
                         desc1, STL_graph1, STL_info1,
                         desc1a, auto_graph1, auto_info1]
            )
            
            sec2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[dataset_filter_state_2, tend2, 
                         desc2, STL_graph2, STL_info2,
                         desc2a, auto_graph2, auto_info2]
            )

            sec3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[dataset_filter_state_3, tend3, 
                         desc3, STL_graph3, STL_info3,
                         desc3a, auto_graph3, auto_info3]
            )

            amb1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[dataset_filter_state_1, tend1, 
                         desc1, STL_graph1, STL_info1,
                         desc1a, auto_graph1, auto_info1]
            )
            
            amb2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[dataset_filter_state_2, tend2, 
                         desc2, STL_graph2, STL_info2,
                         desc2a, auto_graph2, auto_info2]
            )

            amb3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[dataset_filter_state_3, tend3, 
                         desc3, STL_graph3, STL_info3,
                         desc3a, auto_graph3, auto_info3]
            )

            var1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[dataset_filter_state_1, tend1, 
                         desc1, STL_graph1, STL_info1,
                         desc1a, auto_graph1, auto_info1]
            )
            
            var2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[dataset_filter_state_2, tend2, 
                         desc2, STL_graph2, STL_info2,
                         desc2a, auto_graph2, auto_info2]
            )

            var3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[dataset_filter_state_3, tend3, 
                         desc3, STL_graph3, STL_info3,
                         desc3a, auto_graph3, auto_info3]
            )

            graph_serie.change(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )

            graph_mg.change(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )
            
            graph_tend.change(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )

            graph_mm.change(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )

            graph_sd.change(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )

            tab_ST.select(
                fn=tab_ST_on_mat_change,
                inputs=[mat],
                outputs=[dataset_state, prov1, dep1, sec1, amb1, var1, tend1,
                         prov2, dep2, sec2, amb2, var2, tend2,
                         prov3, dep3, sec3, amb3, var3, tend3,
                         dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                         desc1, desc2, desc3,
                         STL_graph1, STL_graph2, STL_graph3,
                         STL_info1, STL_info2, STL_info3,
                         desc1a, desc2a, desc3a,
                         auto_graph1, auto_graph2, auto_graph3,
                         auto_info1, auto_info2, auto_info3]
            )

            graph_button.click(
                fn=tab_ST_on_graph_change,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3, graph_serie, graph_mg, graph_tend, graph_mm, graph_sd],
                outputs=[tend1, tend2, tend3]
            )

            stats_button.click(
                fn=tab_ST_stl_decomp_all,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3],
                outputs=[STL_graph1, STL_info1, STL_graph2, STL_info2, STL_graph3, STL_info3]
            )

            autocor_button.click(
                fn=tab_ST_autocorr_all,
                inputs=[dataset_filter_state_1, dataset_filter_state_2, dataset_filter_state_3,
                        var1, var2, var3],
                outputs=[auto_graph1, auto_info1, auto_graph2, auto_info2, auto_graph3, auto_info3]
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
