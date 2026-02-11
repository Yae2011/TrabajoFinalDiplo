==================================================
ANÁLISIS DE CULTIVOS E INDICADORES EDUCATIVOS
==================================================

Esta aplicación es una plataforma interactiva desarrollada con Gradio para el análisis avanzado de datos agrícolas y educativos en Argentina. Permite la visualización espacial, el análisis estadístico y la predicción de series temporales.

ESTRUCTURA DEL PROYECTO
-----------------------

/app.maf.py
    Archivo principal de la aplicación (Versión Productiva). 
    Contiene toda la lógica de Gradio, procesamiento de datos 
    y modelado estadístico (ARIMA, ML).
    
/app.py
    Versión anterior o de desarrollo de la aplicación.
/style.css
    Archivo de estilos personalizados. Define la estética "dark mode", 
    efectos de glassmorphism y el diseño responsivo de la interfaz.

/Datasets/
    Carpeta que contiene los datos fuente en formato CSV:
    - Cultivos (Soja, Trigo, Maíz, etc.) depurados.
    - Archivos de mapeo de nombres largos y cortos.
    - GeoJSONs para la representación cartográfica de Argentina.

/Images/
    Recursos visuales utilizados en la aplicación (fondos, iconos).

/hist/
    Carpeta que contiene versiones históricas de archivos modificados.

/requirements.txt
    Listado de dependencias de Python necesarias para ejecutar el proyecto.

/prompt.md
    Documento técnico maestro para guía y reproducción del proyecto por IA.

/README.txt
    Este archivo explicativo.

FUNCIONAMIENTO DE LA APP
------------------------

La aplicación se divide en varias secciones clave:

1. Dashboard (MAF):
   - Mapa interactivo de Argentina por provincias y departamentos.
   - Resumen de indicadores (Total, Rendimiento, Provincia Líder).
   - Filtros dinámicos por Cultivo, Provincia, Departamento y Rango de Años.
   - Gráficos de distribución (Pie) y Evolución Histórica con línea de trayectoria.
   - NOVEDAD: Carga automática de datos al seleccionar la pestaña para mayor fluidez.

2. EDA (Exploratory Data Analysis):
   - Análisis visual exhaustivo mediante Boxplots, Histogramas y Curvas de Distribución.
   - Comparativas entre departamentos y análisis de normalidad.

3. Series Temporales:
   - Descomposición automática (STL).
   - Pruebas de estacionariedad (Dicky-Fuller).
   - Modelado predictivo ARIMA con ajuste de parámetros p y q.

4. Módulos de Machine Learning (Educación) [EN DESARROLLO]:
   - Estructura preparada para el análisis de indicadores educativos mediante 
     Bosques Aleatorios, Probabilidad Bayesiana, Redes Neuronales y K-NN/SVM.

REQUISITOS Y EJECUCIÓN
----------------------

Para ejecutar la aplicación se recomienda utilizar un entorno virtual (venv):

1. Asegúrate de tener Python instalado (versión 3.12 recomendada).

2. Crear y activar el entorno virtual:
   
   En macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate

   En Windows:
   python -m venv venv
   .\venv\Scripts\activate

3. Actualizar pip e instalar dependencias:
   pip install --upgrade pip
   pip install -r requirements.txt

4. Lanzar la aplicación productiva:
   python app.maf.py

5. Abre la URL local (normalmente http://127.0.0.1:7860) en el navegador.

--------------------------------------------------
Desarrollado para: Trabajo Final - Diplomatura IDSA
--------------------------------------------------
