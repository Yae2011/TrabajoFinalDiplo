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
    "Por Curso": "Matricula 2011-2024.csv",
    "Por Edad": "Matricula por Edad 2011-2024.csv",
    "Por Población": "Poblacion 2011-2024.csv",
    "Por Trayectoria": "Trayectoria 2011-2024.csv"
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
        # Explicitly clear values even if empty
        return df, gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
    
    # Sort for better UX
    provincias = sorted([str(p) for p in provincias])
    # Use gr.Dropdown to force full component refresh with new choices and cleared value
    # Clear both provincia and departamento when tipo de matrícula changes
    return df, gr.update(choices=provincias, value=None), gr.update(choices=[], value=None), gr.update(value="Ambos"), gr.update(value="Ambos"), "", None, None, None, None
                    
                    
def on_provincia_change(df, provincia):
    if df is None or df.empty or not provincia:
        # Explicitly clear departamento when no provincia is selected
        return gr.Dropdown(choices=[], value=None)
    
    dptos = df[df['provincia'] == provincia]['departamento'].unique()
    dptos = sorted([str(d) for d in dptos])
    # Clear departamento value and update choices when provincia changes
    return gr.Dropdown(choices=dptos, value=None)

def create_boxplot(df):
    if df is None or df.empty:
        return None
    
    # Identify numeric columns for plotting
    # Exclude 'periodo' as it is a time reference
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    
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
        
    ax.set_title("Distribución de Estudiantes por Categoría")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    
    return fig

def create_evolution_chart(df):
    if df is None or df.empty:
        return None
    
    # Check for 'periodo' column
    if 'periodo' not in df.columns:
        return None

    # Identify numeric columns for plotting (excluding 'periodo')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_plot = [c for c in numeric_cols if c != 'periodo']
    
    if not cols_to_plot:
        return None
        
    # Create figure - Adjusted for better screen fill and height
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Sort by period just in case
    df_sorted = df.sort_values('periodo')
    x_data = df_sorted['periodo']
    
    # Plotting - Always Lines
    for col in cols_to_plot:
        ax.plot(x_data, df_sorted[col], label=col, marker='o')
    
    ax.set_title("Cantidad de Estudiantes por Categoría (2011-2024)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure all years are shown 2011-2024
    years = range(2011, 2025)
    ax.set_xticks(years)
    ax.set_xlim(2010.5, 2024.5) # Slight padding to show points clearly
    
    # Handle Legend: Bottom, ~4 rows
    # Calculate columns needed to fit in ~4 rows
    # Calculate columns needed to fit in ~4 rows
    # Ensure at least 1 column. 
    n_vars = len(cols_to_plot)
    n_cols = (n_vars + 3) // 4
    if n_cols < 1: n_cols = 1
    
    # Position below the chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, shadow=False, ncol=n_cols, fontsize=8)
    
    # Adjust layout to make room for legend at the bottom
    plt.tight_layout() 
    # Extra adjustment might be needed if tight_layout doesn't account for bbox_to_anchor outliers well,
    # but usually it tries. If it cuts off, we might need fig.subplots_adjust(bottom=...)
    
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
    # return f"{ds_name} - {provincia} - {departamento}: {len(filtered)} registros - {data_cols_count} campos"
    return f" MATRÍCULA {consulta_type.upper()} PARA {provincia} - {departamento}:  {len(filtered)} REGISTROS  -  {data_cols_count} CAMPOS"

def show_data(df, consulta_type, provincia, departamento, sector, ambito):
    filtered = get_filtered_subset(df, consulta_type, provincia, departamento, sector, ambito)
    
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), "", None, None
        
    # Calcular estadísticas del dataset FILTRADO
    stats = filtered.drop(columns=['periodo'], errors='ignore').describe().round(2).reset_index().rename(columns={'index': 'Medida'})

    all_cols = list(filtered.columns)
    
    # Columns to show logic...
    cols_to_show = [c for c in all_cols if c not in ['provincia', 'departamento', 'sector', 'ambito']]
    
    final_df = filtered[cols_to_show].tail(8)

    # Information string
    ds_name = format_dataset_name(consulta_type)
    # info_text = f"{ds_name} - {provincia} - {departamento}: {len(filtered)} registros - {len(cols_to_show)} campos"
    info_text = f" MATRÍCULA {consulta_type.upper()} PARA {provincia} - {departamento}: {len(filtered)} REGISTROS  -  {len(cols_to_show)} CAMPOS"
    
    # Generate Boxplot
    fig_boxplot = create_boxplot(final_df)
    
    # Generate Evolution Chart
    fig_evolution = create_evolution_chart(final_df)
    
    return stats, final_df, info_text, fig_boxplot, fig_evolution

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
img_path_4 = os.path.join(current_dir, "Images", "header_bg.png")

fondo_app = image_to_base64(img_path_1)
fondo_titulo = image_to_base64(img_path_2)
fondo_contenedor = image_to_base64(img_path_3)
fondo_encabezado = image_to_base64(img_path_4)

extra_css = f"""
.gradio-container {{
    background-image: url('data:image/png;base64,{fondo_app}') !important;
}}

.title-tab {{
    background-image: url('data:image/png;base64,{fondo_titulo}') !important;
}}

.custom-tab {{
    background-image: url('data:image/png;base64,{fondo_contenedor}') !important;
}}

.header-tab {{
    background-image: url('data:image/png;base64,{fondo_encabezado}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    min-height: 50px;
}}
"""

custom_css = base_css + extra_css


# --- UI Construction ---
with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # State storage for the loaded dataframe
    dataset_state = gr.State(pd.DataFrame())
    
    gr.Row(elem_classes="header-tab")
    
    with gr.Tabs():
        with gr.Tab("Inicio"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ANÁLISIS COMPARATIVO DE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
                
        with gr.Tab("Proceso"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FLUJOGRAMA DEL PROCESO", elem_classes="title-text")
            
        with gr.Tab("Visualización de Datos"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("&nbsp;&nbsp;CONSULTA DE DATOS SOBRE provinciaES EDUCATIVAS", elem_classes="title-text")
            
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
                            stats_table = gr.Dataframe(interactive=False)
                        with gr.Column():
                            gr.HTML(value="CONTENIDO DEL DATASET", elem_classes="info-display-2")
                            output_table = gr.Dataframe(interactive=False)
                    
                    with gr.Row(elem_classes="custom-tab"):
                        output_plot_box = gr.Plot()
                    
                    with gr.Row(elem_classes="custom-tab"):
                        output_plot_evolution = gr.Plot()

            
            # --- Interactions ---
            # 1. Load Data on Type Change
            tipo_matricula.change(
                fn=on_dataset_change,
                inputs=[tipo_matricula],
                outputs=[dataset_state, provincia, departamento, sector, ambito, info_label, stats_table, output_table, output_plot_box, output_plot_evolution]
            )
            
            # 2. Update Departamentos on Provincia Change
            # 2.a JS-side explicit clear to Ensure visual reset (Hybrid approach)
            # provincia.change(fn=None, inputs=None, outputs=departamento, js="(val) => null")
            
            # 2.b Python-side update for choices
            provincia.change(
                fn=on_provincia_change,
                inputs=[dataset_state, provincia],
                outputs=[departamento]
            )
            
            # 3. Filter and Show
            btn_mostrar.click(
                fn=show_data,
                inputs=[dataset_state, tipo_matricula, provincia, departamento, sector, ambito],
                outputs=[stats_table, output_table, info_label, output_plot_box, output_plot_evolution]
            )

            # 4. Auto-Update or Clear Logic
            def auto_update_or_clear(df, consulta_type, prov, depto, sec, amb):
                # Check if all required inputs are present
                if prov and depto and sec and amb:
                    # All present -> Execute filter_data logic completely
                    return filter_data(df, consulta_type, prov, depto, sec, amb)
                else:
                    # Missing inputs -> Clear outputs but update info label if possible
                    new_info = calculate_info(df, consulta_type, prov, depto, sec, amb)
                    return pd.DataFrame(), pd.DataFrame(), new_info, None, None

            # Exclude provincia from triggers to avoid race condition with clearing logic
            input_components = [departamento, sector, ambito]
            
            # Bind to inputs
            for comp in input_components:
                comp.change(
                    fn=auto_update_or_clear, 
                    inputs=[dataset_state, tipo_matricula, provincia, departamento, sector, ambito],
                    outputs=[stats_table, output_table, info_label, output_plot_box, output_plot_evolution]
                )

            # Also for dataset change (clears everything)
            # def clear_all():
            #     return None, None, pd.DataFrame(), pd.DataFrame(), "", None, None
                
            # tipo_matricula.change(fn=clear_all, inputs=None, outputs=[provincia, departamento, stats_table, output_table, info_label, output_plot_box, output_plot_evolution])
            
            # Initial Load Trigger (Optional, to load the default selection)
            app.load(fn=on_dataset_change, inputs=[tipo_matricula], outputs=[dataset_state, provincia, departamento])

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
