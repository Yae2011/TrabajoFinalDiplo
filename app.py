import gradio as gr
import pandas as pd
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
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
def tab_EDA_on_load(dataset_type):
    df, provincias = load_data(dataset_type)
    m_inic = "EL DATASET SELECCIONADO DE MATRÍCULA NO ESTÁ DISPONIBLE"

    if df.empty:
        return df, gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value=m_inic)
    
    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"

    return df, gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), gr.update(value=m_inic)

def tab_EDA_on_dataset_change(dataset_type):
    df, provincias = load_data(dataset_type)
    m_inic = "EL DATASET SELECCIONADO DE MATRÍCULA NO ESTÁ DISPONIBLE"

    if df.empty:
        return df, gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False)
    
    # Se arma el listado ordenado de provincias y se guarda la primera provincia
    provincias_sorted = sorted([str(p) for p in provincias])
    prov_first = provincias_sorted[0]

    # Se arma el listado ordenado de departamentos de la primera provincia de la lista
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == prov_first]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"

    # Al actualizar el dataset, el valor de la lista desplegabls "provincia" 
    # muestra la primera provincia del listado y la lista desplegable "departamento"
    # muestra el primer departamento de la provincia. La lista desplegable "indicador"
    # se actualiza recién luego de presionar el botón "Mostrar Datos"
    return df, gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
            gr.update(visible=False), None, None, None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False)

def tab_EDA_on_provincia_change(df, provincia):
    # Se arma el listado ordenado de departamentos de la provincia
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    
    m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
    return  gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
            gr.update(visible=False), None, None, None, None, None, None, \
            gr.Dropdown(choices=[], value="", interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False)

def tab_EDA_on_departamento_change():
    # Al cambiar de departamento se resetea toda la data en pantalla
    m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
    return gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
            gr.update(visible=False), None, None, None, None, None, None, \
            gr.Dropdown(choices=[], value="", interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False)

def tab_EDA_on_opcion_change():
    # Al cambiar la opción de sector o ámbito se resetea toda la data en pantalla
    m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
    return gr.HTML(value=m_inic), gr.update(visible=False), None, None, None, None, None, None, \
            gr.Dropdown(choices=[], value="", interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False)

def tab_EDA_create_boxplot_graph(df):
    # df: dataset filtrado con columnas con "nombres originales"

    if df is None or df.empty:
        return None
    
    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None

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

def tab_EDA_create_evolution_graph(df, indicador):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre original de columna

    # Se crea la figura para el gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    
    try:
        df_sorted = df.sort_values('periodo')
        x_data = df_sorted['periodo']
        y_data = df_sorted[indicador]
        
        ax.plot(x_data, y_data, label=indicador,
                marker='o',                 # Tipo marcador
                linewidth=3.0,              # Espesor línea
                color='green',              # Color línea
                markerfacecolor='red',      # Color marcador
                markeredgecolor='red',      # Color borde marcador
                markeredgewidth=3.0)        # Espesor borde marcador)
        
        # Se calcula la línea de tendencia (Regresión lineal: y = mx + b)
        # Se obtienen la pendiente (z[0]) y la intersección (z[1])
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        
        # Se grafica la línea de tendencia
        ax.plot(x_data, p(x_data), color='orange', linestyle='--', 
                linewidth=2, label='Tendencia')
        
        # Se calcula el promedio de los datos del indicador
        media_valor = np.mean(y_data)
        # Se grafica la línea horizontal representativa de la media
        ax.hlines(y=media_valor, xmin=x_data.min(), xmax=x_data.max(), 
                  color='skyblue', linestyle='--', linewidth=2, label='Media')

        titulo = f"EVOLUCIÓN DE LA MATRÍCULA: {dict_nlargos[indicador].upper()}"
        ax.set_title(titulo)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # se agrega la leyenda para diferenciar la serie de la tendencia
        # ax.legend()

        years = range(2011, 2025)
        ax.set_xticks(years)
        ax.set_xlim(2010.5, 2024.5)
        plt.tight_layout() 
        
        return fig
    
    finally:
        # Se cierra la figura para liberar memoria del backend de Matplotlib.
        # Gradio ya ha convertido la 'fig' en una imagen o formato transferible 
        # antes de que este cierre afecte la visualización en la UI.
        plt.close(fig)

def tab_EDA_create_normal_dist_graph(df, indicador):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre original de columna

    # Se crea la figura para el gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    
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
        ax.axvline(x=mu, color='orange', linestyle='--', linewidth=2, label=f'Media: {mu:.2f}')

        # Configuración de títulos y etiquetas utilizando el diccionario de nombres largos
        titulo = f"DISTRIBUCIÓN NORMAL Y DISPERSIÓN: {dict_nlargos[indicador].upper()}"
        ax.set_title(titulo)
        ax.set_xlabel("Valor de Matrícula")
        ax.set_ylabel("Densidad")
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
    fig, ax = plt.subplots(figsize=(10, 4))
    
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

def tab_EDA_create_all_graphs(df, indicador):
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
    
    fig1 = tab_EDA_create_evolution_graph(df, indicador)
    fig2 = tab_EDA_create_histogram_graph(df, indicador)
    fig3 = tab_EDA_create_normal_dist_graph(df, indicador)
    
    return fig1, fig2, fig3
    
def tab_EDA_create_next_all_graphs(df, indicador):
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
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador)

    # Se genera el gráfico de histograma para el indicador siguiente
    fig2 = tab_EDA_create_histogram_graph(df, nuevo_indicador)

    # Se genera el gráfico de distribución normal para el indicador siguiente
    fig3 = tab_EDA_create_normal_dist_graph(df, nuevo_indicador)

    # Se renombra el nuevo indicador con el nombre corto del diccionario para
    # colocarlo en la lista desplegable
    indicador_ncorto = dict_ncortos.get(nuevo_indicador)

    return gr.update(value=indicador_ncorto), fig1, fig2, fig3

def tab_EDA_create_prev_all_graphs(df, indicador):
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
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador)

    # Se genera el gráfico de histograma para el indicador siguiente
    fig2 = tab_EDA_create_histogram_graph(df, nuevo_indicador)

    # Se genera el gráfico de distribución normal para el indicador siguiente
    fig3 = tab_EDA_create_normal_dist_graph(df, nuevo_indicador)

    # Se renombra el nuevo indicador con el nombre corto del diccionario para
    # colocarlo en la lista desplegable
    indicador_ncorto = dict_ncortos.get(nuevo_indicador)

    return gr.update(value=indicador_ncorto), fig1, fig2, fig3

def get_filtered_subset(df, provincia, departamento, sector, ambito, key_columns, agrupar_detalles=True):

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

def tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito):
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True)
    
    if filtered.empty:
        info_text = f" MATRÍCULA {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): SIN REGISTROS"
        return info_text, gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), None, None, None, None, None, None, None

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
    fig_boxplot = tab_EDA_create_boxplot_graph(final_df)
    
    # Data 5: Generar gráfico de serie temporal con la variable numérica indicada
    fig_evolution = tab_EDA_create_evolution_graph(filtered, indicadores_originales[0])
    
    # Data 6: Generar gráfico de histograma con la variable numérica indicada
    fig_histogram = tab_EDA_create_histogram_graph(filtered, indicadores_originales[0])

    # Data 7: Generar gráfico de distribución normal con la variable numérica indicada
    fig_normal_dist = tab_EDA_create_normal_dist_graph(filtered, indicadores_originales[0])

    return filtered, info_text, gr.update(visible=True), stats, final_df, \
            fig_boxplot, fig_evolution, fig_histogram, fig_normal_dist, \
            gr.Dropdown(choices=indicadores, value=indicador_first, interactive=True), \
            gr.Button(interactive=True), gr.Button(interactive=True)

# endregion


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
# endregion




###### INTERFACE GRADIO ######

with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Almacenamiento para el dataset elegido
    dataset_state = gr.State(pd.DataFrame())
    # Almacenamiento para el dataset elegido y filtrado por campos clave
    dataset_filter_state = gr.State(pd.DataFrame())
    

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
                
                    btn_mostrar = gr.Button("Mostrar Datos", variant="primary", elem_classes="custom-button")
        
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
                            with gr.Column(scale=20):
                                output_plot_evolution = gr.Plot(show_label=False)
                                with gr.Row():
                                    output_plot_histogram = gr.Plot(show_label=False)
                                    output_plot_normal_dist = gr.Plot(show_label=False)


            tipo_matricula.change(
                fn=tab_EDA_on_dataset_change,
                inputs=[tipo_matricula],
                outputs=[dataset_state, provincia, departamento, sector, ambito,
                         info_label, data_dataset, stats_table, output_table, output_plot_box,
                         output_plot_evolution, output_plot_histogram, output_plot_normal_dist,
                         indicador, btn_anterior, btn_siguiente]
            )
            
            provincia.change(
                fn=tab_EDA_on_provincia_change,
                inputs=[dataset_state, provincia],
                outputs=[departamento, sector, ambito,
                         info_label, data_dataset, stats_table, output_table, output_plot_box, 
                         output_plot_evolution, output_plot_histogram, output_plot_normal_dist,
                         indicador, btn_anterior, btn_siguiente]
            )
            
            departamento.change(
                fn=tab_EDA_on_departamento_change,
                outputs=[sector, ambito, info_label, data_dataset, stats_table, output_table, 
                         output_plot_box, output_plot_evolution, 
                         output_plot_histogram, output_plot_normal_dist, indicador, 
                         btn_anterior, btn_siguiente]
            )

            sector.change(
                fn=tab_EDA_on_opcion_change,
                outputs=[info_label, data_dataset, stats_table, output_table, 
                         output_plot_box, output_plot_evolution, 
                         output_plot_histogram, output_plot_normal_dist, indicador,
                         btn_anterior, btn_siguiente]
            )
            
            ambito.change(
                fn=tab_EDA_on_opcion_change,
                outputs=[info_label, data_dataset, stats_table, output_table, 
                         output_plot_box, output_plot_evolution,
                         output_plot_histogram, output_plot_normal_dist, indicador, 
                         btn_anterior, btn_siguiente]
            )

            btn_mostrar.click(
                fn=tab_EDA_show_data,
                inputs=[dataset_state, tipo_matricula, provincia, departamento, sector, ambito],
                outputs=[dataset_filter_state, info_label, data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution, output_plot_histogram,
                            output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente]
            )

            indicador.change(
                fn=tab_EDA_create_all_graphs,
                inputs=[dataset_filter_state, indicador],
                outputs=[output_plot_evolution, output_plot_histogram, output_plot_normal_dist]
            )

            btn_anterior.click(
                fn=tab_EDA_create_prev_all_graphs,
                inputs=[dataset_filter_state, indicador],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )
            
            btn_siguiente.click(
                fn=tab_EDA_create_next_all_graphs,
                inputs=[dataset_filter_state, indicador],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )
            
            '''app.load(
                fn=on_load_page, 
                inputs=[tipo_matricula], 
                outputs=[dataset_state, provincia, departamento, info_label]
            )'''
            tab_EDA.select(
                fn=tab_EDA_on_load, 
                inputs=[tipo_matricula], 
                outputs=[dataset_state, provincia, departamento, info_label]
            )
        

        ###### PESTAÑA SERIES TEMPORALES
        with gr.Tab("Series Temporales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;COMPARACIÓN DE SERIES TEMPORALES APLICANDO LA TRANSFORMADA RÁPIDA DE FOURIER", elem_classes="title-text")
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;1. SELECCIÓN DE LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")
            with gr.Row():
                with gr.Column(min_width=180, elem_classes="custom-tab"):
                    mat = gr.Radio(label="Tipo de Matrícula", 
                            choices=["Por Curso", "Por Población", "Por Trayectoria"],
                            value="Por Curso", elem_classes="custom-radio")
                    var = gr.Dropdown(label="Variable", choices=[], elem_classes="custom-dropdown")
                with gr.Column(scale=20):
                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column(min_width=180):
                            prov1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                            dep1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        with gr.Column(min_width=180):
                            sec1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                            amb1 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        with gr.Column(scale=20):
                            tend1 = gr.Plot(show_label=False)
                    with gr.Row(elem_classes="custom-tab"):
                        prov2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        dep2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        sec2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        amb2 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        tend2 = gr.Plot()
                    with gr.Row(elem_classes="custom-tab"):
                        prov3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        dep3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        sec3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        amb3 = gr.Dropdown(label="", choices=[], elem_classes="custom-dropdown")
                        tend3 = gr.Plot()
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;2. TEST DE DICKEY-FÜLLER AUMENTADO (ADF)", elem_classes="subtitle-text")
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;3. SELECCIÓN DEL MODELO Y VALORES DE SUS HIPERPARÁMETROS APLICANDO UN ALGORITMO GENÉTICO", elem_classes="subtitle-text")  
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;4. PREDICCIÓN DE LAS SERIES", elem_classes="subtitle-text")
           
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;5. TRANSFORMADA DE FOURIER PARA LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")              


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
