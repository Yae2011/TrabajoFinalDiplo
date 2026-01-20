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
def tab_EDA_on_load(dataset_type, automatico):
    df, provincias = load_data(dataset_type)
    
    if df.empty:
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True)
    
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
        return tab_EDA_show_data(df, dataset_type, prov_first, dpto_first, sector, ambito, True)

    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, pd.DataFrame(), gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), \
            gr.update(value=m_inic), gr.update(visible=False), \
            pd.DataFrame(), pd.DataFrame(), \
            None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True)
            
def tab_EDA_on_dataset_change(dataset_type, automatico):
    df, provincias = load_data(dataset_type)
    
    if df.empty:
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True)
    
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
        return tab_EDA_show_data(df, dataset_type, prov_first, dpto_first, sector, ambito, True)
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
                gr.update(visible=True)

def tab_EDA_on_provincia_change(df, df_filtered, dataset_type, provincia, automatico):

    # Se arma el listado ordenado de departamentos de la provincia
    # y se guarda el primer departamento de la lista
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    dpto_first = dptos_sorted[0]
    
    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, provincia, dpto_first, sector, ambito, True)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(choices=dptos_sorted, value=dpto_first), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True)

def tab_EDA_on_departamento_change(df, df_filtered, dataset_type, provincia, departamento, automatico):

    if automatico:
        sector = "Ambos"
        ambito = "Ambos"
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(value=departamento), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True)

def tab_EDA_on_opcion_change(df, df_filtered, dataset_type, provincia, departamento, sector, ambito, automatico):

    if automatico:
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True)
    else:
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, df_filtered, gr.update(value=provincia), \
                gr.update(value=departamento), \
                gr.update(value=sector), gr.update(value=ambito), gr.HTML(value=m_inic), \
                gr.update(visible=False), None, None, None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True)

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

def tab_EDA_create_evolution_graph(df, indicador, serie=True, med_glob=True, tend=True, med_mov=False, sd_mov=False, tipo_mov=4):
    # df: dataset filtrado con columnas con nombres originales
    # indicador: nombre original de columna
    # serie: muestra la serie de datos
    # med_glob: muestra la media global
    # tend: muestra la tendencia
    # med_mov: muestra la media móvil
    # sd_mov: muestra la SD móvil
    # tipo_mov: ventana para media y SD móviles (2: k=2; 3: k=3 hacia atrás; 4: k=3 centrado)

    # Se crea la figura para el gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    
    try:
        df_sorted = df.sort_values('periodo')
        x_data = df_sorted['periodo']
        y_data = df_sorted[indicador]
        
        if serie: # Se grafica la serie temporal de datos
            ax.plot(x_data, y_data, label="Matrícula",
                marker='o',                 # Tipo marcador
                linewidth=3.0,              # Espesor línea
                color='green',              # Color línea
                markerfacecolor='red',      # Color marcador
                markeredgecolor='red',      # Color borde marcador
                markeredgewidth=3.0)        # Espesor borde marcador)

        if med_glob: # Se calcula el promedio de los datos del indicador
            media_valor = np.mean(y_data)
            # Se grafica la línea horizontal representativa de la media
            ax.hlines(y=media_valor, xmin=x_data.min(), xmax=x_data.max(), 
                  color='skyblue', linestyle='--', linewidth=2, label='Media Global')
                    
        if tend: # Se calcula la línea de tendencia (Regresión lineal: y = mx + b)
            # Se obtienen la pendiente (z[0]) y la intersección (z[1])
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            # Se grafica la línea de tendencia
            ax.plot(x_data, p(x_data), color='orange', linestyle='--', 
                linewidth=2, label='Tendencia')
        
        if med_mov or sd_mov:
            # Configuración de la ventana (k) y centrado según tipo_mov
            # 2: k=2; 3: k=3 hacia atrás; 4: k=3 centrado; otro: k=2
            if tipo_mov == 3:
                k, centro = 3, False
                lab_graph = "3 atrás"
            elif tipo_mov == 4:
                k, centro = 3, True
                lab_graph = "3 centrado"
            else:
                k, centro = 2, False
                lab_graph = "2"

            y_med_mov = y_data.rolling(window=k, center=centro).mean()
            if med_mov: # Cálculo de la media móvil
                ax.plot(x_data, y_med_mov, color='purple', linestyle='--', 
                        linewidth=2, label=f'Media Móvil (k={lab_graph})')

            if sd_mov: # Cálculo de la desviación estándar móvil
                y_sd_mov = y_data.rolling(window=k, center=centro).std()
                
                # Se grafica la SD aplicada a la media  (Media +/- SD)
                y_superior = y_med_mov + y_sd_mov
                y_inferior = y_med_mov - y_sd_mov
        
                # Generación del área sombreada
                ax.fill_between(x_data, y_inferior, y_superior, color='purple', alpha=0.2, 
                        label=f'SD Móvil (k={lab_graph})')
        
                # Líneas tenues para marcar los límites de la SD
                ax.plot(x_data, y_superior, color='purple', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.plot(x_data, y_inferior, color='purple', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # Grafica el valor absoluto de la SD, generalmente en la parte inferior
                # del eje Y quitándole escala a la serie principal de datos
                # ax.plot(x_data, y_sd_mov, color='gray', linestyle='--', 
                #        linewidth=1.5, label=f'SD Móvil (k={lab_graph})')
                
                # Banda de confianza (Media Móvil ± SD Móvil)
                # if med_mov:
                    # ax.fill_between(x_data, y_med_mov - y_sd_mov, y_med_mov + y_sd_mov, 
                                    # color='yellow', alpha=0.1, label='Banda Variabilidad')

        titulo = f"EVOLUCIÓN DE LA MATRÍCULA: {dict_nlargos[indicador].upper()}"
        ax.set_title(titulo)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # se agrega la leyenda para diferenciar las diferentes líneas, si las hay
        if serie or med_glob or tend or med_mov or sd_mov:
            ax.legend()

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

def tab_EDA_create_all_graphs(df, indicador, serie, mg, tend, mm, sd):
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
    
    fig1 = tab_EDA_create_evolution_graph(df, indicador, serie, mg, tend, mm, sd, tipo_mov=4)
    fig2 = tab_EDA_create_histogram_graph(df, indicador)
    fig3 = tab_EDA_create_normal_dist_graph(df, indicador)
    
    return fig1, fig2, fig3

def tab_EDA_options_graph(df, indicador, serie, mg, tend, mm, sd):
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
    
    fig = tab_EDA_create_evolution_graph(df, indicador, serie, mg, tend, mm, sd, 4)
   
    return fig
    
def tab_EDA_create_next_all_graphs(df, indicador, serie, mg, tend, mm, sd):
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
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador, serie, mg, tend, mm, sd, tipo_mov=4)

    # Se genera el gráfico de histograma para el indicador siguiente
    fig2 = tab_EDA_create_histogram_graph(df, nuevo_indicador)

    # Se genera el gráfico de distribución normal para el indicador siguiente
    fig3 = tab_EDA_create_normal_dist_graph(df, nuevo_indicador)

    # Se renombra el nuevo indicador con el nombre corto del diccionario para
    # colocarlo en la lista desplegable
    indicador_ncorto = dict_ncortos.get(nuevo_indicador)

    return gr.update(value=indicador_ncorto), fig1, fig2, fig3

def tab_EDA_create_prev_all_graphs(df, indicador, serie, mg, tend, mm, sd):
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
    fig1 = tab_EDA_create_evolution_graph(df, nuevo_indicador, serie, mg, tend, mm, sd, tipo_mov=4)

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

def tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, automatico):

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
            gr.Button(interactive=True), gr.Button(interactive=True), gr.update(visible=True)

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
    fig_evolution = tab_EDA_create_evolution_graph(filtered, indicadores_originales[0], serie=True, med_glob=True,
                                           tend=True, med_mov=False, sd_mov=False, tipo_mov=4)
    
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
            gr.Button(interactive=True), gr.Button(interactive=True), gr.update(visible=mostrar_boton)

def tab_EDA_on_checkbox(dataset_type, provincia, departamento, sector, ambito, automatico):
    df, provincias = load_data(dataset_type)

    if df.empty:
        m_inic = "EL DATASET DE MATRÍCULA SELECCIONADO NO ESTÁ DISPONIBLE"
        return df, pd.DataFrame(), gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), gr.HTML(value=m_inic), \
                gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), \
                None, None, None, None, \
                gr.Dropdown(choices=[], value=None, interactive=False), \
                gr.Button(interactive=False), gr.Button(interactive=False), \
                gr.update(visible=True)

    if automatico: # Se marcó la casilla "Automático", se muestra toda la data y se oculta el botón "Mostrar datos"
        return tab_EDA_show_data(df, dataset_type, provincia, departamento, sector, ambito, True)
    else: # Se desmarcó la casilla "Automático"; se limpia toda la data y se muestra el botón "Mostrar datos"
        m_inic = "DEBE SELECCIONARSE EL BOTÓN \"MOSTRAR DATOS\" PARA VISUALIZAR LOS RESULTADOS"
        return df, pd.DataFrame(), gr.update(value=provincia), \
            gr.update(value=departamento), \
            gr.update(value="Ambos"), gr.update(value="Ambos"), \
            gr.update(value=m_inic), \
            gr.update(visible=False), pd.DataFrame(), pd.DataFrame(), \
            None, None, None, None, \
            gr.Dropdown(choices=[], value=None, interactive=False), \
            gr.Button(interactive=False), gr.Button(interactive=False), gr.update(visible=True)
                          

# endregion FUNCIONES PARA LA PESTAÑA "EDA"


# region FUNCIONES PARA LA PESTAÑA "SERIES TEMPORALES"

def tab_ST_on_mat_change(dataset_type):
    df, provincias = load_data(dataset_type)

    if df.empty:
        return df, gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), None, \
                gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), None, \
                gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
                gr.update(value="Ambos"), gr.update(value="Ambos"), \
                gr.update(choices=[], value=None), None
    
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
    filtered = get_filtered_subset(df, prov_first, dpto_first, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)

    # Se genera el gráfico para el primer indicador
    graph = tab_EDA_create_evolution_graph(filtered, indicadores_originales[0], serie=True, med_glob=True,
                                           tend=True, med_mov=True, sd_mov=True, tipo_mov=4)

    # Al actualizar el dataset, se muestra la primera provincia, el primer departamento,
    # sector = "Ambos", ambiente="Ambos", el primer indicador y el gráfico correspodiente.
    # Se hace para las tres series temporales 
    return df, gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
            gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
            gr.update(choices=indicadores, value=indicador_first), graph, \
            gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
            gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
            gr.update(choices=indicadores, value=indicador_first), graph, \
            gr.update(choices=provincias_sorted, value=prov_first), \
            gr.update(choices=dptos_sorted, value=dpto_first), \
            gr.update(choices=["Estatal", "Privado", "Ambos"], value="Ambos"), \
            gr.update(choices=["Urbano", "Rural", "Ambos"], value="Ambos"), \
            gr.update(choices=indicadores, value=indicador_first), graph
            
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
    filtered = get_filtered_subset(df, provincia, dpto_first, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None, None
    
    # Se genera el gráfico para el primer indicador
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie=True, med_glob=True,
                                           tend=True, med_mov=True, sd_mov=True, tipo_mov=4)

    return  gr.update(choices=dptos_sorted, value=dpto_first), graph

def tab_ST_on_dep_change(df, provincia, departamento, sector, ambito, indicador):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de matrícula
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None
    
    # Se genera el gráfico para el primer indicador
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie=True, med_glob=True,
                                           tend=True, med_mov=True, sd_mov=True, tipo_mov=4)

    return graph

def tab_ST_on_option_change(df, provincia, departamento, sector, ambito, indicador):

    # Como el parámetro "indicador" se recibe con el nombre descriptivo corto
    # se debe convertir a su nombre original
    ind_orig = next((k for k, v in dict_ncortos.items() if v == indicador), indicador)

    # Se filtra el dataset de matrícula
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True, min_reg=MIN_REG)
    
    if filtered.empty:
        return None
    
    # Se genera el gráfico para el primer indicador
    graph = tab_EDA_create_evolution_graph(filtered, ind_orig, serie=True, med_glob=True,
                                           tend=True, med_mov=True, sd_mov=True, tipo_mov=4)

    return graph

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
                    
                    chk_mostrar = gr.Checkbox(label="Automático", value=False, elem_classes="custom-checkbox")
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
                inputs=[tipo_matricula, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )
            
            provincia.change(
                fn=tab_EDA_on_provincia_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )
            
            departamento.change(
                fn=tab_EDA_on_departamento_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia, departamento, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )

            sector.change(
                fn=tab_EDA_on_opcion_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )
            
            ambito.change(
                fn=tab_EDA_on_opcion_change,
                inputs=[dataset_state, dataset_filter_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )

            btn_mostrar.click(
                fn=tab_EDA_show_data,
                inputs=[dataset_state, tipo_matricula, provincia,
                        departamento, sector, ambito, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )

            indicador.change(
                fn=tab_EDA_create_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution, output_plot_histogram, output_plot_normal_dist]
            )

            btn_anterior.click(
                fn=tab_EDA_create_prev_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )
            
            btn_siguiente.click(
                fn=tab_EDA_create_next_all_graphs,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[indicador, output_plot_evolution, output_plot_histogram,
                        output_plot_normal_dist]
            )

            chk_serie.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution]
            )

            chk_mg.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution]
            )

            chk_tend.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution]
            )

            chk_mm.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution]
            )

            chk_sd.change(
                fn=tab_EDA_options_graph,
                inputs=[dataset_filter_state, indicador, chk_serie, chk_mg, chk_tend,
                       chk_mm, chk_sd],
                outputs=[output_plot_evolution]
            )

            chk_mostrar.select(
                fn=tab_EDA_on_checkbox,
                inputs=[tipo_matricula, provincia, departamento, sector, ambito, chk_mostrar],
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )

            tab_EDA.select(
                fn=tab_EDA_on_load, 
                inputs=[tipo_matricula, chk_mostrar], 
                outputs=[dataset_state, dataset_filter_state, provincia, departamento, 
                            sector, ambito, info_label,
                            data_dataset, stats_table, output_table,
                            output_plot_box, output_plot_evolution,
                            output_plot_histogram, output_plot_normal_dist,
                            indicador, btn_anterior, btn_siguiente, btn_mostrar]
            )
        

        ###### PESTAÑA SERIES TEMPORALES
        with gr.Tab("Series Temporales") as tab_ST:
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;COMPARACIÓN DE SERIES TEMPORALES APLICANDO LA TRANSFORMADA RÁPIDA DE FOURIER", elem_classes="title-text")
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;1. SELECCIÓN DE LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")
            
            with gr.Row():
                with gr.Column(min_width=180):
                    with gr.Row(elem_classes="custom-tab"):
                        mat = gr.Radio(label="Tipo de Matrícula", 
                            choices=["Por Curso", "Por Población", "Por Trayectoria"],
                            value="Por Curso", elem_classes="custom-radio")
                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column():
                            gr.HTML("Opciones de Gráficos", elem_classes="title-group")
                            graph_mg = gr.Checkbox(label="Media Global", value=True, elem_classes="custom-checkbox")
                            graph_tend = gr.Checkbox(label="Tendencia", value=True, elem_classes="custom-checkbox")
                            graph_mm = gr.Radio(label="Media Móvil", choices=["No", "k = 2 atrás",
                                                            "k = 3 atrás", "k = 3 centrado"],
                                                            value="k = 3 centrado", elem_classes="custom-radio")
                            graph_sd = gr.Checkbox(label="SD Móvil", value=False, elem_classes="custom-checkbox")

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
                            tend1 = gr.Plot(show_label=False)

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
                            tend2 = gr.Plot(show_label=False)

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
                            tend3 = gr.Plot(show_label=False)


            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;2. TEST DE DICKEY-FÜLLER AUMENTADO (ADF)", elem_classes="subtitle-text")
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;3. SELECCIÓN DEL MODELO Y VALORES DE SUS HIPERPARÁMETROS APLICANDO UN ALGORITMO GENÉTICO", elem_classes="subtitle-text")  
            
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;4. PREDICCIÓN DE LAS SERIES", elem_classes="subtitle-text")
           
            with gr.Row(elem_classes="custom-tab-2"):    
                gr.HTML("&nbsp;&nbsp;5. TRANSFORMADA DE FOURIER PARA LAS SERIES TEMPORALES A COMPARAR", elem_classes="subtitle-text")              


            mat.change(
                fn=tab_ST_on_mat_change,
                inputs=[mat],
                outputs=[dataset_state, prov1, dep1, sec1, amb1, var1, tend1,
                         prov2, dep2, sec2, amb2, var2, tend2,
                         prov3, dep3, sec3, amb3, var3, tend3]
            )
            
            prov1.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov1, sec1, amb1, var1],
                outputs=[dep1, tend1]
            )

            prov2.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov2, sec2, amb2, var2],
                outputs=[dep2, tend2]
            )
            
            prov3.change(
                fn=tab_ST_on_prov_change,
                inputs=[dataset_state, prov3, sec3, amb3, var3],
                outputs=[dep3, tend3]
            )
            
            dep1.change(
                fn=tab_ST_on_dep_change,
                inputs=[dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[tend1]
            )

            dep2.change(
                fn=tab_ST_on_dep_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[tend2]
            )

            dep3.change(
                fn=tab_ST_on_dep_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[tend3]
            )

            sec1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[tend1]
            )
            
            sec2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[tend2]
            )

            sec3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[tend3]
            )

            amb1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[tend1]
            )
            
            amb2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[tend2]
            )

            amb3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[tend3]
            )

            var1.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov1, dep1, sec1, amb1, var1],
                outputs=[tend1]
            )
            
            var2.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov2, dep2, sec2, amb2, var2],
                outputs=[tend2]
            )

            var3.change(
                fn=tab_ST_on_option_change,
                inputs = [dataset_state, prov3, dep3, sec3, amb3, var3],
                outputs=[tend3]
            )

            tab_ST.select(
                fn=tab_ST_on_mat_change,
                inputs=[mat],
                outputs=[dataset_state, prov1, dep1, sec1, amb1, var1, tend1,
                         prov2, dep2, sec2, amb2, var2, tend2,
                         prov3, dep3, sec3, amb3, var3, tend3]
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
