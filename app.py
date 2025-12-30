import gradio as gr
import pandas as pd
import os
import base64
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# --- Constants ---
DATA_PATH = "./Datasets"
FILE_MAP = {
    "Matrícula": "Matricula 2011-2024.csv",
    "Matr. por Edad": "Matricula por Edad 2011-2024.csv",
    "Población": "Poblacion 2011-2024.csv",
    "Trayectorias": "Trayectoria 2011-2024.csv"
}

KEY_COLUMNS = ['periodo', 'provincia', 'departamento', 'sector', 'ambito']

# --- Global State ---
# We store the currently loaded dataframe to avoid reloading on every interaction,
# though for a multi-user web app this might need a different approach (like passing state).
# For a local Gradio app, a global/closure variable or gr.State is fine.
# We will use gr.State explicitly in the UI flow.

def get_file_path(consulta_type):
    filename = FILE_MAP.get(consulta_type)
    if not filename:
        return None
    return os.path.join(DATA_PATH, filename)

def load_data(consulta_type):
    path = get_file_path(consulta_type)
    if not path or not os.path.exists(path):
        return pd.DataFrame(), ["Archivo no encontrado"]
    
    try:
        # Assuming standard CSV format. Adjust sep/encoding if needed after testing.
        df = pd.read_csv(path, encoding='utf-8', sep=',') # Common defaults
        return df, list(df['provincia'].unique())
    except Exception as e:
        # Fallback for common encoding/separator issues
        try:
             df = pd.read_csv(path, encoding='latin1', sep=';')
             return df, list(df['provincia'].unique())
        except:
             return pd.DataFrame(), [f"Error cargando: {e}"]

def update_departamentos(consulta_type, provincia):
    path = get_file_path(consulta_type)
    if not path or not os.path.exists(path):
         return []
    
    # We reload or ideally keep it in memory. 
    # To keep it simple and stateless between calls without gr.State being too complex initially:
    # We'll read the file again or use a cached approach. 
    # Given the requirements, let's try to load efficienty.
    # Note: For better performance, we should pass the dataframe via gr.State.
    
    # However, this function is called by the Dropdown 'change' event.
    # We need the dataframe. We will assume the dataframe is loaded in the 'Mostrar Datos' step?
    # No, we need it for the dropdowns.
    
    # Strategy: Load data when 'Type' changes, output to a State component.
    pass

# Refined Logic with gr.State
def on_dataset_change(consulta_type):
    df, provincias = load_data(consulta_type)
    if df.empty:
        return df, gr.update(choices=[]), gr.update(choices=[])
    
    # Sort for better UX
    provincias = sorted([str(p) for p in provincias])
    return df, gr.update(choices=provincias, value=None), gr.update(choices=[], value=None)

def on_provincia_change(df, provincia):
    if df is None or df.empty or not provincia:
        return gr.update(choices=[])
    
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos = sorted([str(d) for d in dptos])
    return gr.update(choices=dptos, value=None)

def create_boxplot(df):
    if df is None or df.empty:
        return None
    
    # Identify numeric columns for plotting
    # Exclude 'periodo' as it is a time reference
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data values handling potential NaNs
    data_values = []
    headers = []
    for col in cols_to_plot:
        data_values.append(df[col].dropna())
        headers.append(col)
        
    # Create boxplot
    box = ax.boxplot(data_values, patch_artist=True, labels=headers, medianprops=dict(color="white", linewidth=1.5))
    
    # Style: Celeste (lightblue)
    for patch in box['boxes']:
        patch.set_facecolor('blue')
        
    ax.set_title("Cantidad de Estudiantes por Categoría")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=90, fontsize=5)
    plt.tight_layout()
    
    return fig

def get_filtered_subset(df, consulta_type, provincia, departamento, sector, ambito):
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Check if essential inputs are selected
    if not provincia or not departamento:
        return pd.DataFrame()

    # Base filter: provincia and departamento
    base_filter = (df['provincia'] == provincia) & (df['departamento'] == departamento)
    
    # Determine if we need to aggregate
    need_sector_aggregation = (sector == "Ambos")
    need_ambito_aggregation = (ambito == "Ambos")
    
    if not need_sector_aggregation and not need_ambito_aggregation:
        # Simple case: no aggregation needed, just filter
        final_filter = base_filter & (df['sector'] == sector) & (df['ambito'] == ambito)
        return df[final_filter]
    
    # Build the filter for rows to include in aggregation
    rows_filter = base_filter
    
    if not need_sector_aggregation:
        rows_filter = rows_filter & (df['sector'] == sector)
    
    if not need_ambito_aggregation:
        rows_filter = rows_filter & (df['ambito'] == ambito)
    
    # Filter the dataframe
    filtered_df = df[rows_filter].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Determine grouping columns and numeric columns
    group_cols = ['periodo', 'provincia', 'departamento']

    # Special handling for 'grado' column (found in Matrícula por Edad)
    # We want to group by it, not sum/concatenate it.
    if 'grado' in filtered_df.columns:
        group_cols.append('grado')
    
    # Add sector to grouping if not aggregating by sector
    if not need_sector_aggregation:
        group_cols.append('sector')
    
    # Add ambito to grouping if not aggregating by ambito
    if not need_ambito_aggregation:
        group_cols.append('ambito')
    
    # Identify numeric columns (all columns except key columns and 'grado')
    all_cols = filtered_df.columns.tolist()
    # columns to exclude from summation
    exclude_cols = set(KEY_COLUMNS)
    exclude_cols.add('grado')
    
    numeric_cols = [col for col in all_cols if col not in exclude_cols]
    
    # Group and sum
    aggregated = filtered_df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    
    # Add the aggregated columns back with "Ambos" label
    if need_sector_aggregation:
        aggregated['sector'] = 'Ambos'
    
    if need_ambito_aggregation:
        aggregated['ambito'] = 'Ambos'
    
    # Reorder columns to match original structure
    final_cols = list(KEY_COLUMNS)
    if 'grado' in aggregated.columns:
        final_cols.append('grado')
        
    final_cols = final_cols + numeric_cols
    aggregated = aggregated[final_cols]
    
    return aggregated

def format_dataset_name(name):
    if not name: return ""
    return name.upper().replace("MATR.", "MATRÍCULA")

def calculate_info(df, consulta_type, provincia, departamento, sector, ambito):
    if df is None or df.empty or not provincia or not departamento:
        return ""
        
    filtered = get_filtered_subset(df, consulta_type, provincia, departamento, sector, ambito)
    
    # Calculate visible data columns count (excluding key columns)
    # Typically 5 key columns: periodo, provincia, departamento, sector, ambito
    data_cols_count = max(0, len(filtered.columns) - 5)
    
    ds_name = format_dataset_name(consulta_type)
    return f"{ds_name} - {provincia} - {departamento}: {len(filtered)} registros - {data_cols_count} campos"

def filter_data(df, consulta_type, provincia, departamento, sector, ambito):
    filtered = get_filtered_subset(df, consulta_type, provincia, departamento, sector, ambito)
    
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), "", None
        
    # Calcular estadísticas del dataset FILTRADO
    stats = filtered.drop(columns=['periodo'], errors='ignore').describe().round(2).reset_index().rename(columns={'index': 'Medida'})

    all_cols = list(filtered.columns)
    
    # Columns to show: exclude filtering keys to avoid redundancy if desired, 
    # but user request history implied showing everything or specific logic.
    # Previous logic was: data_cols = all_cols[0:1] + all_cols[5:]
    # Now we want better dynamic handling.
    # We will exclude the standard navigation keys from the main view if that's the goal, 
    # OR follow the previous "everything after keys" logic.
    # The keys are: provincia, departamento, sector, ambito.
    # 'periodo' is usually kept. 'grado' should be kept.
    
    cols_to_show = [c for c in all_cols if c not in ['provincia', 'departamento', 'sector', 'ambito']]
    
    final_df = filtered[cols_to_show] 

    # Information string
    ds_name = format_dataset_name(consulta_type)
    info_text = f"{ds_name} - {provincia} - {departamento}: {len(filtered)} registros - {len(cols_to_show)} campos"
    
    # Generate Plot
    fig = create_boxplot(final_df)
    
    return stats, final_df, info_text, fig

# 1. Leer el contenido del CSS manualmente para asegurar compatibilidad
# 1. Funcion para convertir imagen a Base64
def image_to_base64(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 2. Leer el contenido del CSS manualmente
css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        base_css = f.read()
else:
    base_css = ""
    print("Advertencia: style.css no encontrado.")

# 3. Codificar imágenes y agregar al CSS
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path_1 = os.path.join(current_dir, "Images", "App_bg.png")
img_path_2 = os.path.join(current_dir, "Images", "Title_bg.png")
img_path_3 = os.path.join(current_dir, "Images", "Container_bg.png")

fondo_app = image_to_base64(img_path_1)
fondo_titulo = image_to_base64(img_path_2)
fondo_contenedor = image_to_base64(img_path_3)

extra_css = f"""
.gradio-container {{
    background-image: url('data:image/png;base64,{fondo_app}') !important;
}}

.title-tab {{
    background-image: url('data:image/png;base64,{fondo_titulo}') !important;
}}

.custom-tab-bg {{
    background-image: url('data:image/png;base64,{fondo_contenedor}') !important;
}}
"""

custom_css = base_css + extra_css


# --- UI Construction ---
with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # State storage for the loaded dataframe
    dataset_state = gr.State(pd.DataFrame())

    with gr.Tabs():
        with gr.Tab("Inicio"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;ANÁLISIS COMPARATIVO DE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
        with gr.Tab("Proceso"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;FLUJOGRAMA DEL PROCESO", elem_classes="title-text")
            
        with gr.Tab("Visualización de Datos"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;CONSULTA DE DATOS SOBRE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
            with gr.Row():
                with gr.Column(min_width=180, scale=1, elem_classes=["custom-tab-bg", "narrow-column"]):
                    tipo_consulta = gr.Radio(
                        label="Tipo de Consulta", 
                        choices=["Matrícula", "Matr. por Edad", "Población", "Trayectorias"],
                        value="Matrícula",
                        elem_classes=["custom-radio", "vertical-radio"]
                    )
        
                    # Dropdowns
                    jurisdiccion = gr.Dropdown(label="Jurisdicción Educativa", choices=[], elem_classes="custom-input")
                    departamento = gr.Dropdown(label="Departamento", choices=[], elem_classes="custom-input")
            
                    sector = gr.Radio(label="Sector", choices=["Estatal", "Privado", "Ambos"], value="Ambos", elem_classes=["custom-radio", "vertical-radio"])
                    ambito = gr.Radio(label="Ámbito", choices=["Urbano", "Rural", "Ambos"], value="Ambos", elem_classes=["custom-radio", "vertical-radio"])
                
                    
                    btn_mostrar = gr.Button("Mostrar Datos", variant="primary", elem_classes="custom-button")
        
                with gr.Column(scale=20, elem_classes="custom-tab-bg"):
                    # info_label ahora es un HTML con estilo propio
                    info_label = gr.HTML(value=" ", elem_classes="info-display-2")
                    gr.HTML(value="ESTADÍSTICAS DEL DATASET", elem_classes="info-display-1")
                    stats_table = gr.Dataframe(interactive=False)
                    gr.HTML(value="CONTENIDO DEL DATASET", elem_classes="info-display-1")
                    output_table = gr.Dataframe(interactive=False)
                    
                    gr.HTML(value="GRÁFICO DE ESTUDIANTES POR CATEGORÍA", elem_classes="info-display-1")
                    output_plot = gr.Plot(label="Distribución")

            
            # --- Interactions ---
            # 1. Load Data on Type Change
            tipo_consulta.change(
                fn=on_dataset_change,
                inputs=[tipo_consulta],
                outputs=[dataset_state, jurisdiccion, departamento]
            )
            
            # 2. Update Departamentos on Provincia Change
            jurisdiccion.change(
                fn=on_provincia_change,
                inputs=[dataset_state, jurisdiccion],
                outputs=[departamento]
            )
            
            # 3. Filter and Show
            btn_mostrar.click(
                fn=filter_data,
                inputs=[dataset_state, tipo_consulta, jurisdiccion, departamento, sector, ambito],
                outputs=[stats_table, output_table, info_label, output_plot]
            )

            # 4. Clear Outputs on Input Change AND Update Info Label
            def clear_outputs_and_update_info(df, consulta_type, prov, depto, sec, amb):
                # Update info immediately
                new_info = calculate_info(df, consulta_type, prov, depto, sec, amb)
                return pd.DataFrame(), pd.DataFrame(), new_info, None

            input_components = [jurisdiccion, departamento, sector, ambito]
            
            # Bind to inputs
            for comp in input_components:
                comp.change(
                    fn=clear_outputs_and_update_info, 
                    inputs=[dataset_state, tipo_consulta, jurisdiccion, departamento, sector, ambito],
                    outputs=[stats_table, output_table, info_label, output_plot]
                )

            # Also for dataset change (clears everything)
            def clear_all():
                return pd.DataFrame(), pd.DataFrame(), "", None
                
            tipo_consulta.change(fn=clear_all, inputs=None, outputs=[stats_table, output_table, info_label, output_plot])
            
            # Initial Load Trigger (Optional, to load the default selection)
            app.load(fn=on_dataset_change, inputs=[tipo_consulta], outputs=[dataset_state, jurisdiccion, departamento])

        with gr.Tab("Series Temporales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;DEFINICIÓN DE LAS SERIES TEMPORALES A SER COMPARADAS", elem_classes="title-text")
            
        with gr.Tab("Series de Fourier"):
            with gr.Row(elem_classes="title-tab"):
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
    # Obtenemos la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definimos la carpeta de imágenes
    images_folder = os.path.join(current_dir, "Images")
    
    # Verificación de existencia (Debug)
    if not os.path.exists(images_folder):
        print(f"Advertencia: La carpeta {images_folder} no existe.")

    # Lanzamos la aplicación
    # allowed_paths DEBE incluir las rutas absolutas de las carpetas que contienen recursos
    app.launch(
        allowed_paths=[current_dir, images_folder]
    )
