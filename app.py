import gradio as gr
import pandas as pd
import os

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

def get_filtered_subset(df, provincia, departamento, sector, ambito):
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
    
    # Add sector to grouping if not aggregating by sector
    if not need_sector_aggregation:
        group_cols.append('sector')
    
    # Add ambito to grouping if not aggregating by ambito
    if not need_ambito_aggregation:
        group_cols.append('ambito')
    
    # Identify numeric columns (all columns except key columns)
    all_cols = filtered_df.columns.tolist()
    numeric_cols = [col for col in all_cols if col not in KEY_COLUMNS]
    
    # Group and sum
    aggregated = filtered_df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    
    # Add the aggregated columns back with "Ambos" label
    if need_sector_aggregation:
        aggregated['sector'] = 'Ambos'
    
    if need_ambito_aggregation:
        aggregated['ambito'] = 'Ambos'
    
    # Reorder columns to match original structure
    final_cols = KEY_COLUMNS + numeric_cols
    aggregated = aggregated[final_cols]
    
    return aggregated

def calculate_info(df, provincia, departamento, sector, ambito):
    if df is None or df.empty or not provincia or not departamento:
        return ""
        
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito)
    
    # Calculate visible data columns count (excluding key columns)
    # Typically 5 key columns: periodo, provincia, departamento, sector, ambito
    data_cols_count = max(0, len(filtered.columns) - 5)
    
    return f"{provincia} - {departamento}: {len(filtered)} registros - {data_cols_count} campos"

def filter_data(df, provincia, departamento, sector, ambito):
    filtered = get_filtered_subset(df, provincia, departamento, sector, ambito)
    
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), ""
        
    # Calcular estadísticas del dataset FILTRADO
    stats = filtered.drop(columns=['periodo'], errors='ignore').describe().round(2).reset_index().rename(columns={'index': 'Medida'})

    all_cols = list(filtered.columns)
    # data_cols = all_cols[5:] # All columns after the keys
    data_cols = all_cols[0:1] + all_cols[5:] # Primera columna del año y luego todas las columnas luego de las de filtrado
    
    # Show columns 1, 6, 7, ... (not keys columns)
    # Mostrar las primeras 14 filas de las columnas seleccionadas
    # final_df = filtered[data_cols].head(14)
    # Mostrar todas las filas de las columnas seleccionadas
    final_df = filtered[data_cols] 
    # Mostrar todas las filas de todas las columnas
    # final_df = filtered # Mostrar todas las filas y todas las columnas

    # Information string
    info_text = f"{provincia} - {departamento}: {len(filtered)} registros {len(data_cols)} campos"
    
    return stats, final_df, info_text


# --- UI Construction ---
# Read CSS file
with open("style.css", "r", encoding="utf-8") as f:
    css_content = f.read()

# Encode background image
import base64
image_path = "Images/Fondo_1.png"
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

css_content += f"""
    .gradio-container {{
        /*background-image: url('data:image/png;base64,{encoded_string}') !important;*/
        background-size: cover !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }}
"""

# --- UI Construction ---
with gr.Blocks(title="Análisis Educativo") as app:
    gr.HTML(f"<style>{css_content}</style>")
    
    # State storage for the loaded dataframe
    dataset_state = gr.State(pd.DataFrame())

    with gr.Tabs():
        with gr.Tab("Inicio"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS COMPARATIVO DE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
        with gr.Tab("Proceso"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("FLUJOGRAMA DEL PROCESO", elem_classes="title-text")
            
        with gr.Tab("Visualización de Datos"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("CONSULTA DE DATOS SOBRE JURISDICCIONES EDUCATIVAS", elem_classes="title-text")
            
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
                    gr.HTML(value="Estadísticas del Dataset", elem_classes="info-display-1")
                    stats_table = gr.Dataframe(interactive=False)
                    gr.HTML(value="Muestra del Dataset", elem_classes="info-display-1")
                    output_table = gr.Dataframe(interactive=False)

            
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
                inputs=[dataset_state, jurisdiccion, departamento, sector, ambito],
                outputs=[stats_table, output_table, info_label]
            )

            # 4. Clear Outputs on Input Change AND Update Info Label
            def clear_outputs_and_update_info(df, prov, depto, sec, amb):
                # Update info immediately
                new_info = calculate_info(df, prov, depto, sec, amb)
                return pd.DataFrame(), pd.DataFrame(), new_info

            input_components = [jurisdiccion, departamento, sector, ambito]
            
            # Bind to inputs
            for comp in input_components:
                comp.change(
                    fn=clear_outputs_and_update_info, 
                    inputs=[dataset_state, jurisdiccion, departamento, sector, ambito],
                    outputs=[stats_table, output_table, info_label]
                )

            # Also for dataset change (clears everything)
            def clear_all():
                return pd.DataFrame(), pd.DataFrame(), ""
                
            tipo_consulta.change(fn=clear_all, inputs=None, outputs=[stats_table, output_table, info_label])
            
            # Initial Load Trigger (Optional, to load the default selection)
            app.load(fn=on_dataset_change, inputs=[tipo_consulta], outputs=[dataset_state, jurisdiccion, departamento])

        with gr.Tab("Series Temporales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("DEFINICIÓN DE LAS SERIES TEMPORALES A SER COMPARADAS", elem_classes="title-text")
            
        with gr.Tab("Series de Fourier"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS DE SERIES TEMPORALES MEDIANTE SERIES DE FOURIER", elem_classes="title-text")

        with gr.Tab("Yael - Bosques Aleatorios"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE BOSQUES ALEATORIOS", elem_classes="title-text")
                
        with gr.Tab("Marco - Probabilidad Bayesiana"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS DE SERIES TEMPORALES MEDIANTE PROBABILIDAD BAYESIANA", elem_classes="title-text")
        
        with gr.Tab("Yael - Redes Neuronales"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS DE INDICADORES EDUCATIVOS MEDIANTE REDES NEURONALES", elem_classes="title-text")
        
        with gr.Tab("Marco - KNN & SVM"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("ANÁLISIS DE INDICADORES EDUCATIVOS CON K-NN Y SVM", elem_classes="title-text")
            
        with gr.Tab("Conclusiones"):
            with gr.Row(elem_classes="title-tab"):
                gr.HTML("CONCLUSIONES", elem_classes="title-text")

if __name__ == "__main__":
    app.launch()
