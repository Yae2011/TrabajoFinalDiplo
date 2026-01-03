import gradio as gr
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt

# --- Constantes ---
DATA_PATH = "./Datasets"
FILE_MAP = {
    "Por Curso": "Matricula 2011-2024.csv",
    "Por Edad": "Matricula por Edad 2011-2024.csv",
    "Por Población": "Poblacion 2011-2024.csv",
    "Por Trayectoria": "Trayectoria 2011-2024.csv"
}

KEY_COLUMNS = ['periodo', 'provincia', 'departamento', 'sector', 'ambito']

# --- Dataframe Global ---
# Se almacena el DataFrame actual para evitar recargarlo en cada interacción.
# Para una aplicación local de Gradio, conviene utilizar gr.State() en la interfaz (UI).

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

def on_dataset_change(dataset_type):
    df, provincias = load_data(dataset_type)
    if df.empty:
        return df, gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value="Ambos"), gr.update(value="Ambos"), "", None, None, None, None
    
    lista_provincias = sorted([str(p) for p in provincias])
    return df, gr.update(choices=lista_provincias, value=None), gr.update(choices=[], value=None), gr.update(value="Ambos"), gr.update(value="Ambos"), "", None, None, None, None
                                
def on_provincia_change(df, provincia):
    if df is None or df.empty or not provincia:
        return gr.update(choices=[], value="")
    
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos_sorted = sorted([str(d) for d in dptos if d is not None])
    return gr.update(choices=dptos_sorted, value=None, interactive=True)

def create_boxplot(df):
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
        
    # Se crea el gráfico
    box = ax.boxplot(data_values, patch_artist=True, labels=headers, medianprops=dict(color="white", linewidth=1.5))
    
    # Cajas color celeste
    for patch in box['boxes']:
        patch.set_facecolor('blue')
        
    ax.set_title("Distribución de Estudiantes por Categoría")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    
    return fig

def create_evolution_chart(df):
    if df is None or df.empty:
        return None
    
    # Si no hay columna "período"
    if 'periodo' not in df.columns:
        return None

    # Columnas numéricas para graficar, se excluye la columna "período"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None
        
    # Se crea la figura para el gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Se ordena por "período"
    df_sorted = df.sort_values('periodo')
    x_data = df_sorted['periodo']
    
    # Gráfico de líneas con marcador de puntos
    for col in cols_to_plot:
        ax.plot(x_data, df_sorted[col], label=col, marker='o')
    
    ax.set_title("Cantidad de Estudiantes por Categoría (2011-2024)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Se muestran todos los años del período aunque no tengan datos
    years = range(2011, 2025)
    ax.set_xticks(years)
    ax.set_xlim(2010.5, 2024.5) # Slight padding to show points clearly
    
    # Dimensiones de la leyenda
    # Se calculan las columnas para que entren en 4 filas
    n_vars = len(cols_to_plot)
    n_cols = (n_vars + 3) // 4
    if n_cols < 1: n_cols = 1
    
    # Leyenda debajo del gráfico
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, shadow=False, ncol=n_cols, fontsize=8)
    
    # Se ajusta el gráfico para que la leyenda se acomode correctamente
    plt.tight_layout() 
    
    return fig

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

def filter_data(df, dataset_type, provincia, departamento, sector, ambito):
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito, KEY_COLUMNS, True)
    
    if filtered.empty:
        info_text = f" MATRÍCULA {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): SIN REGISTROS"
        return info_text, pd.DataFrame(), pd.DataFrame(), None, None
        
    # Calcular estadísticas del dataset filtrado
    stats = filtered.drop(columns=['periodo'], errors='ignore').describe().round(2).reset_index().rename(columns={'index': 'Medida'})

    all_cols = list(filtered.columns)
    
    # Columnas para mostrar del dataset
    # cols_to_show = [c for c in all_cols if c not in ['provincia', 'departamento', 'sector', 'ambito']]
    cols_to_show = [c for c in all_cols if c not in ['provincia', 'departamento']]
    final_df = filtered[cols_to_show]

    # Mensaje informativo sobre registros y campos
    info_text = f" MATRÍCULA {dataset_type.upper()} PARA {provincia} - {departamento} (SECTOR {sector.upper()} - ÁMBITO {ambito.upper()}): {len(filtered)} REGISTROS  -  {len(cols_to_show)} CAMPOS"
    
    # Generar gráfico de cajas
    fig_boxplot = create_boxplot(final_df)
    
    # Generar gráfico de serie temporal
    fig_evolution = create_evolution_chart(final_df)
    
    return info_text, stats, final_df, fig_boxplot, fig_evolution

# Funcion para convertir imagen a Base64
def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Leer el contenido del CSS externo para agregar
# las líneas correspondientes a las imágenes de fondo
css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        base_css = f.read()
else:
    base_css = ""
    print("Advertencia: style.css no encontrado.")

# Transformación de imágenes PNG a BASE64
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path_1 = os.path.join(current_dir, "Images", "App_bg.png")
img_path_2 = os.path.join(current_dir, "Images", "Title_bg.png")
img_path_3 = os.path.join(current_dir, "Images", "Container_bg.png")
img_path_4 = os.path.join(current_dir, "Images", "header_bg.png")
img_path_5 = os.path.join(current_dir, "Images", "portrait_bg.png")

fondo_app = image_to_base64(img_path_1)
fondo_titulo = image_to_base64(img_path_2)
fondo_contenedor = image_to_base64(img_path_3)
fondo_encabezado = image_to_base64(img_path_4)
fondo_portada = image_to_base64(img_path_5)

# Se agregan al CSS leído, las líneas que aplican las imágenes de fondo en BASE64,
# que no pueden insertarse direcatmente en "style.css"
extra_css = f"""
.gradio-container {{ background-image: url('data:image/png;base64,{fondo_app}') !important; }}
.title-tab {{ background-image: url('data:image/png;base64,{fondo_titulo}') !important; }}
.custom-tab {{ background-image: url('data:image/png;base64,{fondo_contenedor}') !important; }}
.header-tab {{ background-image: url('data:image/png;base64,{fondo_encabezado}') !important; }}
.portrait {{ background-image: url('data:image/png;base64,{fondo_portada}') !important; }}
"""
custom_css = base_css + extra_css



### INTERFACE GRADIO

with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # State storage for the loaded dataframe
    dataset_state = gr.State(pd.DataFrame())
    
    gr.Row(elem_classes="header-tab")
    
    with gr.Tabs():
        with gr.Tab("Inicio"):
            with gr.Row(elem_classes="portrait-bg"):
                with gr.Column(scale=8, elem_classes="portrait"):
                    gr.HTML("ANÁLISIS DE LA MATRÍCULA ESCOLAR<br>"
                            "DE LA REPÚBLICA ARGENTINA,<br>"
                            "EN EL PERÍODO 2011-2024,<br>"
                            "PARA TODAS LAS JURISDICCIONES<br>"
                            "EDUCATIVAS DEL PAÍS", elem_classes="portrait-title")
                with gr.Column(scale=2, elem_classes="portrait-bg2"):
                    gr.HTML("Aplicación de algoritmos de Machine Learning "
                            "a las Bases de Datos Abiertas de la Secretaría de Educación "
                            "del Ministerio de Capital Humano de la República Argentina "
                            "para el análisis de los indicadores de matrícula escolar "
                            "para todas las provincias y CABA y sus respectivos "
                            "departamentos, partidos o comunas.",
                            elem_classes="portrait-subtitle")

        with gr.Tab("Proceso"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FLUJOGRAMA DEL PROCESO", elem_classes="title-text")
            
        with gr.Tab("Visualización de Datos"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;CONSULTA DE DATOS SOBRE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
            with gr.Row():
                with gr.Column(min_width=180, scale=1, elem_classes="custom-tab"):
                    tipo_matricula = gr.Radio(
                        label="Tipo de Matrícula", 
                        choices=["Por Curso", "Por Edad", "Por Población", "Por Trayectoria"],
                        value="Por Curso",
                        elem_classes="custom-radio"
                    )
        
                    # Dropdowns
                    provincia = gr.Dropdown(label="Provincia", choices=[], elem_classes="custom-dropdown")
                    departamento = gr.Dropdown(label="Departamento", choices=[], elem_classes="custom-dropdown")
            
                    sector = gr.Radio(label="Sector", choices=["Estatal", "Privado", "Ambos"], value="Ambos", elem_classes="custom-radio")
                    ambito = gr.Radio(label="Ámbito", choices=["Urbano", "Rural", "Ambos"], value="Ambos", elem_classes="custom-radio")
                
                    btn_mostrar = gr.Button("Mostrar Datos", variant="primary", elem_classes="custom-button")
        
                with gr.Column(scale=20):
                    with gr.Row(elem_classes="custom-tab"):
                        info_label = gr.HTML(value="", elem_classes="info-display-1")

                    with gr.Row(elem_classes="custom-tab"):
                        with gr.Column():
                            gr.HTML(value="ESTADÍSTICAS DEL DATASET", elem_classes="info-display-2")
                            stats_table = gr.Dataframe(interactive=False, max_height=335)
                        with gr.Column():
                            gr.HTML(value="CONTENIDO DEL DATASET", elem_classes="info-display-2")
                            output_table = gr.Dataframe(interactive=False, max_height=335)
                    
                    with gr.Row(elem_classes="custom-tab"):
                        output_plot_box = gr.Plot()
                    
                    with gr.Row(elem_classes="custom-tab"):
                        output_plot_evolution = gr.Plot()

            
            tipo_matricula.change(
                fn=on_dataset_change,
                inputs=[tipo_matricula],
                outputs=[dataset_state, provincia, departamento, sector, ambito, info_label, stats_table, output_table, output_plot_box, output_plot_evolution]
            )
            
            provincia.change(
                fn=on_provincia_change,
                inputs=[dataset_state, provincia],
                outputs=[departamento]
            )
            
            btn_mostrar.click(
                fn=filter_data,
                inputs=[dataset_state, tipo_matricula, provincia, departamento, sector, ambito],
                outputs=[info_label, stats_table, output_table, output_plot_box, output_plot_evolution]
            )

            # input_components = [departamento, sector, ambito]
            # for comp in input_components:
            #    comp.change(
            #        fn=filter_data, 
            #        inputs=[dataset_state, tipo_matricula, provincia, departamento, sector, ambito],
            #        outputs=[info_label, stats_table, output_table, output_plot_box, output_plot_evolution]
            #    )

            app.load(
                fn=on_dataset_change, 
                inputs=[tipo_matricula], 
                outputs=[dataset_state, provincia, departamento, sector, ambito, info_label, stats_table, output_table, output_plot_box, output_plot_evolution]
            )

        with gr.Tab("Series Temporales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;DEFINICIÓN DE LAS SERIES TEMPORALES A SER COMPARADAS", elem_classes="title-text")
            
        with gr.Tab("Series de Fourier"):
            with gr.Row(elem_classes="titleapp-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE SERIES TEMPORALES MEDIANTE SERIES DE FOURIER", elem_classes="title-text")

        with gr.Tab("Yael - Bosques Aleatorios"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE BOSQUES ALEATORIOS", elem_classes="title-text")
                
        with gr.Tab("Marco - Probabilidad Bayesiana"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE SERIES TEMPORALES MEDIANTE PROBABILIDAD BAYESIANA", elem_classes="title-text")
        
        with gr.Tab("Yael - Redes Neuronales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE REDES NEURONALES", elem_classes="title-text")
        
        with gr.Tab("Marco - KNN & SVM"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS DE INDICADORES EDUCATIVOS CON K-NN Y SVM", elem_classes="title-text")
            
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
