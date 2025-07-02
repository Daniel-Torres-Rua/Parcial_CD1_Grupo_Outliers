# Importa la librería principal para construir la aplicación web interactiva.
import streamlit as st
# Importa pandas para manipulación y análisis de datos (DataFrames).
import pandas as pd
# Importa numpy para operaciones numéricas y matriciales.
import numpy as np
# Importa el módulo 'os' para interactuar con el sistema operativo (manejo de rutas de archivos, directorios).
import os
# Importa matplotlib.pyplot para crear visualizaciones (gráficos).
import matplotlib.pyplot as plt
# Importa seaborn para crear visualizaciones estadísticas atractivas.
import seaborn as sns
# Importa time para medir el tiempo de ejecución.
import time
# Importa train_test_split para dividir datos y GridSearchCV para tuning de hiperparámetros.
from sklearn.model_selection import train_test_split, GridSearchCV
# Importa métricas y herramientas para evaluar modelos de clasificación.
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
# Importa escaladores y codificadores para preprocesamiento de datos.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# Importa tensorflow, la biblioteca base para aprendizaje profundo.
import tensorflow as tf
# Importa keras, una API de alto nivel para construir y entrenar modelos de redes neuronales.
from tensorflow import keras
# Importa la capa 'layers' de Keras para definir las capas de la red neuronal.
from tensorflow.keras import layers
# Importar PyTorch y skorch
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

# Define el directorio donde se guardarán los archivos de salida.
output_dir = 'existing_analysis_outputs'

@st.cache_resource # Almacena en caché el recurso para evitar recargas en cada ejecución de Streamlit
def load_and_preprocess_data():
    try:
        df = pd.read_csv('peru_match_results.csv') # Carga el archivo CSV en un DataFrame
        df.drop('match_id', axis=1, inplace=True) # Elimina la columna 'match_id'
        # Define un mapeo para renombrar columnas a español
        rename_mapping = {"rival_confederation":"confederacion_rival","peru_score":"puntuacion_peru","rival_score":"puntuacion_rival","peru_awarded_score":"goles_peru","rival_awarded_score":"goles_rival","result":"resultado","shootout_result":"resultado_penales","awarded_result":"resultado_final_oficial","tournament_name":"nombre_torneo","tournament_type":"tipo_torneo","official":"partido_oficial","stadium":"nombre_estadio","city":"ciudad","country":"pais","elevation":"altitud_sede","peru_condition":"localia_peru","coach":"entrenador","coach_nationality":"nacionalidad_entrenador","date":"fecha"}
        df.rename(columns=rename_mapping, inplace=True) # Renombra las columnas
        df = df[df['entrenador'] == 'Ricardo Gareca'].copy() # Filtra el DataFrame para incluir solo partidos de Ricardo Gareca
        # Elimina varias columnas que no se usarán para el modelo
        df.drop(['resultado_penales', 'nombre_torneo', 'tipo_torneo',
                 'goles_peru','goles_rival','puntuacion_peru','puntuacion_rival','resultado_final_oficial','nombre_estadio','ciudad','pais','entrenador','nacionalidad_entrenador','fecha'], axis=1, inplace=True)
        bins = [0, 500, 1500, 3000, 10000] # Define los rangos para la categorización de altitud
        labels = ['Bajo', 'Moderado', 'Alto', 'Extremo'] # Define las etiquetas para las categorías de altitud
        df['categoria_altitud'] = pd.cut(df['altitud_sede'], bins=bins, labels=labels, include_lowest=True) # Crea una nueva columna 'categoria_altitud'
        df['partido_oficial'] = df['partido_oficial'].astype(int) # Convierte la columna 'partido_oficial' a tipo entero
        altitude_le = LabelEncoder() # Inicializa un codificador de etiquetas
        df['categoria_altitud'] = altitude_le.fit_transform(df['categoria_altitud']) # Codifica las categorías de altitud a valores numéricos
        df.drop('altitud_sede',axis=1, inplace=True) # Elimina la columna 'altitud_sede' original
        # Codifica columnas categóricas a valores numéricos y guarda las categorías originales
        df['confederacion_rival_num'], confederacion_rival_categories = pd.factorize(df['confederacion_rival'])
        df['resultado_num'], resultado_categories = pd.factorize(df['resultado'])
        df['localia_peru_num'], localia_peru_categories = pd.factorize(df['localia_peru'])
        freq_encoding = df['rival'].value_counts(normalize=True).to_dict() # Calcula la frecuencia de cada rival
        df['rival_encoded'] = df['rival'].map(freq_encoding) # Codifica 'rival' usando la frecuencia
        # Elimina las columnas categóricas originales que ya fueron codificadas
        df = df.drop(['confederacion_rival', 'resultado', 'localia_peru', 'rival'], axis=1)

        y_encoded = df['resultado_num'] # Define la variable objetivo (resultado codificado)
        features = df.drop('resultado_num', axis=1) # Define las características (todas las columnas menos la variable objetivo)

        X_train_full = features # Asigna las características para entrenamiento
        y_train_full = y_encoded # Asigna la variable objetivo para entrenamiento

        scaler = MinMaxScaler() # Inicializa un escalador Min-Max
        # Define las características a escalar
        features_to_scale = ['partido_oficial', 'categoria_altitud', 'confederacion_rival_num', 'localia_peru_num', 'rival_encoded']
        X_train_full[features_to_scale] = scaler.fit_transform(X_train_full[features_to_scale]) # Escala las características seleccionadas


        preprocessing_pipeline = {
            'scaler': scaler, # Guarda el escalador
            'altitude_le': altitude_le, # Guarda el codificador de altitud
            'confederacion_rival_categories': confederacion_rival_categories, # Guarda las categorías de confederación rival
            'resultado_categories': resultado_categories, # Guarda las categorías de resultado (necesario para decodificar predicciones)
            'localia_peru_categories': localia_peru_categories, # Guarda las categorías de localía de Perú
            'freq_encoding': freq_encoding, # Guarda el mapeo de codificación por frecuencia
            'altitude_bins': bins, # Guarda los rangos de altitud
            'altitude_labels': labels # Guarda las etiquetas de altitud
        }

        return X_train_full, y_train_full, preprocessing_pipeline # Retorna los datos preprocesados y el pipeline

    except FileNotFoundError:
        st.error("Error: The file 'peru_match_results.csv' was not found.") # Maneja el error si el archivo no se encuentra
        return None, None, None # Retorna valores nulos en caso de error
    except Exception as e:
        st.error(f"An error occurred during data loading and preprocessing: {e}") # Maneja cualquier otra excepción
        return None, None, None # Retorna valores nulos en caso de error

X_train_full, y_train_full, preprocessing_pipeline = load_and_preprocess_data() # Carga los datos preprocesados y el pipeline de preprocesamiento

# Define la arquitectura del modelo PyTorch (necesario para cargar el modelo)
class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim, num_units1=64, num_units2=32, dropout_p=0.3):
        super(MLP_PyTorch, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_units1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(num_units1, num_units2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(num_units2, len(preprocessing_pipeline['resultado_categories']))
        )

    def forward(self, x):
        return self.layers(x)

@st.cache_resource # Almacena en caché el recurso
def train_pytorch_model(X_train, y_train, input_dim, num_classes):
     if X_train is None or y_train is None:
         return None

     # Configurar early stopping y otros callbacks si es necesario para entrenamiento final
     # Note: Skorch's NeuralNetClassifier trains the model when .fit() is called.
     # The GridSearchCV already did the hyperparameter tuning and fitting.
     # We just need to retrieve the best model from the GridSearchCV result.

     # Assuming gs_pytorch is available from the main notebook execution
     # We need to find a way to get the best_pytorch_model into the Streamlit context.
     # The simplest way is to save/load the model if it's not too large,
     # or re-run the GridSearchCV with refit=True if it's feasible (might be slow).

     # For demonstration, we'll hardcode hyperparameters found previously or load from file if saved.
     # A better way would be to save the best_pytorch_params dict.
     # Let's assume best_pytorch_params is available globally in the notebook context
     # and we've saved it or can access it.

     # Example of loading hardcoded or saved best params (adapt as needed)
     # For this example, let's define a placeholder for best params
     # A real implementation would load from 'existing_analysis_outputs/pytorch_results_best_hyperparameters.txt'
     # and parse it or load a saved params dictionary.
     # Let's use the values we found: {'batch_size': 64, 'max_epochs': 50, 'module__dropout_p': 0.2, 'module__num_units1': 128, 'module__num_units2': 32, 'optimizer__lr': 0.001}
     try:
        # Recreate and train the best PyTorch model based on the best hyperparameters
        best_pytorch_params_found = {
             'module__num_units1': 128,
             'module__num_units2': 32,
             'module__dropout_p': 0.2,
             'optimizer__lr': 0.001,
             'batch_size': 64,
             'max_epochs': 50
        }

        best_pytorch_model_skorch = NeuralNetClassifier(
            module=MLP_PyTorch,
            module__input_dim=input_dim,
            module__num_units1=best_pytorch_params_found['module__num_units1'],
            module__num_units2=best_pytorch_params_found['module__num_units2'],
            module__dropout_p=best_pytorch_params_found['module__dropout_p'],
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__lr=best_pytorch_params_found['optimizer__lr'],
            max_epochs=best_pytorch_params_found['max_epochs'],
            batch_size=best_pytorch_params_found['batch_size'],
            verbose=0 # Suppress verbose output during Streamlit loading
            # train_split=predefined_split(val_dataset) # Use predefined_split if validation set is available
        )

        # Train the model on the full training data (X_train_full, y_train_full)
        # Note: In the notebook, GridSearchCV with refit=True trains the best model.
        # To use that exact model here without re-tuning, you'd need to save/load it.
        # For simplicity in this Streamlit app, we'll re-train the best model architecture
        # with the best hyperparameters on the full training data.
        st.info("Entrenando el modelo PyTorch con los mejores hiperparámetros...")
        start_time = time.time()
        best_pytorch_model_skorch.fit(X_train.values.astype(np.float32), y_train.values.astype(np.int64))
        end_time = time.time()
        st.success(f"Entrenamiento de PyTorch completado en {end_time - start_time:.2f} segundos.")

        return best_pytorch_model_skorch

     except Exception as e:
         st.error(f"Error training PyTorch model: {e}")
         return None

# Determine input dimension and number of classes
if X_train_full is not None and y_train_full is not None:
    INPUT_DIM_PYTORCH = X_train_full.shape[1]
    NUM_CLASSES_PYTORCH = len(np.unique(y_train_full))
else:
    INPUT_DIM_PYTORCH = 5 # Default or handle error
    NUM_CLASSES_PYTORCH = 3 # Default or handle error

best_pytorch_model = train_pytorch_model(X_train_full, y_train_full, INPUT_DIM_PYTORCH, NUM_CLASSES_PYTORCH)


st.title("PARCIAL Ciencia de Datos 1") # Establece el título principal de la aplicación Streamlit
st.markdown("""
**Integrantes:** # Agrega un subtítulo para los integrantes
* Bravo Yataco Luiggi Mariano
* Llamoca León Israel
* Poma Gómez Diego Alonso
* Torres Rua Daniel Isaias
""")

st.header("Contexto") # Establece un encabezado para la sección de Contexto
st.markdown("""
Luego de un exhaustivo proceso de recopilación de datos, se ha creado una base de datos lista para usar que contiene los resultados de todos los partidos jugados por la selección nacional de fútbol del Perú, junto con detalles importantes para el análisis. Este proyecto nació de una pasión personal por las estadísticas del fútbol, con la intención inicial de generar gráficos y resúmenes para compartir con amigos. Ahora, el propósito es poner esta información a disposición de la comunidad de Kaggle.

Contenido del dataset:

El archivo principal peru_match_results.csv recopila los resultados de todos los partidos disputados por la selección peruana desde su primer encuentro oficial contra Uruguay el 1 de noviembre de 1927, hasta el partido más reciente registrado al momento de la última actualización.

Se incluyen tanto partidos oficiales como amistosos.

Existen tres partidos excluidos por motivos específicos (ver sección Not considered matches para más detalles).
""") # Muestra una descripción del contexto del proyecto y el dataset

# --- Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([ # Crea un conjunto de pestañas para organizar el contenido
    "1. Descripción del conjunto de datos", # Etiqueta de la primera pestaña
    "2. Análisis Exploratorio de Datos (EDA)", # Etiqueta de la segunda pestaña
    "3. Transformación de Variables", # Etiqueta de la tercera pestaña
    "4. Modelado con Keras", # Etiqueta de la cuarta pestaña
    "5. Modelado con PyTorch", # Etiqueta de la quinta pestaña
    "6. Evaluación y Selección del Modelo", # Etiqueta de la sexta pestaña
    "7. Predicción" # Etiqueta de la séptima pestaña
])

with tab1: # Define el contenido de la primera pestaña
    st.header("1. Descripción del conjunto de datos") # Establece un encabezado para la sección
    st.markdown("""
    Fuente: https://www.kaggle.com/datasets/arturoarias12/peruvian-national-football-team-results?resource=download
    """) # Muestra la fuente del conjunto de datos
    st.subheader("Variables") # Establece un subtítulo para las variables
    st.markdown("""
    * **match_id:** Comenzando con M seguido del número de coincidencia en orden cronológico.
    * **date:** Fecha del partido.
    * **rival:** Nombre del equipo contra el que jugó Perú.
    * **rival_confederation:** Confederación donde pertenece el rival.
    * **peru_score:** Goles marcados por Perú en el partido.
    * **rival_score:** Goles recibidos por Perú en el partido.
    * **peru_awarded_score:** Goles marcados por Perú luego de revisiones o sanciones (si las hubiera).
    * **rival_awarded_score:** Goles recibidos por Perú luego de revisiones o sanciones (si las hubiera).
    * **result:** Resultado del partido (G: victoria, D: empate, L: derrota).
    * **shootout_result:** Resultado de la tanda de penales (si aplica).
    * **awarded_result:** Resultado del partido después de revisiones o sanciones (si las hubiera).
    * **tournament_name:** Nombre específico del torneo (por ejemplo: Copa Mundial de la FIFA 2018).
    * **tournament_type:** Tipo de torneo (p. ej.: Copa Mundial de la FIFA).
    * **official:** Booleano que indica si el partido fue oficial.
    * **stadium:** Nombre del estadio donde se jugó el partido.
    * **city:** Ciudad donde se jugó el partido.
    * **country:** País donde se jugó el partido.
    * **elevation:** Elevación (sobre el nivel del mar) de la ciudad donde se jugó el partido.
    * **peru_condition:** Indica si Perú jugó como equipo local, visitante o neutral.
    * **coach:** Nombre del entrenador de la selección peruana al momento del partido.
    * **coach_nationality:** Nacionalidad del entrenador.
    """) # Lista y describe cada variable del dataset

    st.subheader("Primeras filas del DataFrame (después de cargar)") # Establece un subtítulo
    try:
        df_initial = pd.read_csv('peru_match_results.csv') # Intenta leer el CSV
        st.dataframe(df_initial.head()) # Muestra las primeras filas del DataFrame
    except FileNotFoundError:
        st.error("Error: The file 'peru_match_results.csv' was not found.") # Maneja el error si el archivo no existe
    except Exception as e:
        st.error(f"Error loading or displaying initial dataframe head: {e}") # Maneja otras excepciones

    st.subheader("Naturaleza de las columnas") # Establece un subtítulo
    try:
        dtypes_path = os.path.join(output_dir, 'data_info_dtypes.txt') # Construye la ruta al archivo de tipos de datos
        if os.path.exists(dtypes_path): # Verifica si el archivo existe
            with open(dtypes_path, 'r') as f:
                dtypes_content = f.read() # Lee el contenido del archivo
            st.code(dtypes_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {dtypes_path}. Please ensure you have run the cell to save outputs.") # Advierte si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying dtypes from {dtypes_path}: {e}") # Maneja otras excepciones

    st.subheader("Número de filas") # Establece un subtítulo
    try:
        df_initial = pd.read_csv('peru_match_results.csv') # Intenta leer el CSV
        st.write(df_initial.shape[0]) # Muestra el número de filas
    except FileNotFoundError:
        st.error("Error: The file 'peru_match_results.csv' was not found to determine row count.") # Maneja el error si el archivo no existe
    except Exception as e:
        st.error(f"Error determining number of rows: {e}") # Maneja otras excepciones

    st.subheader("Número de columnas") # Establece un subtítulo
    try:
        df_initial = pd.read_csv('peru_match_results.csv') # Intenta leer el CSV
        st.write(df_initial.shape[1]) # Muestra el número de columnas
    except FileNotFoundError:
        st.error("Error: The file 'peru_match_results.csv' was not found to determine column count.") # Maneja el error si el archivo no existe
    except Exception as e:
        st.error(f"Error determining number of columns: {e}") # Maneja otras excepciones

    st.subheader("Seleccionando una Muestra (Filtro por Ricardo Gareca)") # Establece un subtítulo
    st.markdown("Se filtró el dataset para incluir solo los partidos dirigidos por Ricardo Gareca.") # Explica el filtro aplicado

    st.subheader("Análisis de valores faltantes (después de eliminar columnas)") # Establece un subtítulo
    try:
        missing_values_path = os.path.join(output_dir, 'data_info_missing_values_after_drop.txt') # Construye la ruta al archivo de valores faltantes
        if os.path.exists(missing_values_path): # Verifica si el archivo existe
            st.markdown("Valores faltantes por columna (después de eliminar columnas con +15% NA y filtrar por entrenador):") # Muestra un mensaje
            with open(missing_values_path, 'r') as f:
                missing_values_content = f.read() # Lee el contenido del archivo
            st.code(missing_values_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {missing_values_path}. Please ensure you have run the cell to save outputs.") # Advierte si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying missing values from {missing_values_path}: {e}") # Maneja otras excepciones


with tab2: # Define el contenido para la segunda pestaña
    st.header("2. Análisis Exploratorio de Datos (EDA)") # Título de la sección EDA
    st.subheader("Distribución de resultados")
    try:
        plot_path = os.path.join(output_dir, 'eda_plots_result_distribution.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Resultados por condición")
    try:
        plot_path = os.path.join(output_dir, 'eda_plots_results_by_condition.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Resultados por confederación rival")
    try:
        plot_path = os.path.join(output_dir, 'eda_plots_results_by_confederation.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error


with tab3: # Define el contenido para la tercera pestaña
    st.header("3. Transformación de Variables") # Título de la sección de Transformación de Variables

    st.subheader("Porcentajes de las categorías por variable nominal (después del filtro por entrenador)") # Subtítulo
    try:
        df_value_counts = pd.read_csv('peru_match_results.csv') # Carga el DataFrame para calcular los conteos de valores

        df_value_counts = df_value_counts[df_value_counts['coach'] == 'Ricardo Gareca'].copy() # Filtra el DataFrame por el entrenador Ricardo Gareca
        # Mapea y renombra columnas relevantes para la visualización de porcentajes
        rename_mapping_vc = {"rival_confederation":"confederacion_rival","result":"resultado","peru_condition":"localia_peru"}
        df_value_counts.rename(columns=rename_mapping_vc, inplace=True) # Aplica el renombramiento de columnas

        nominal_vars_to_show = ['rival', 'confederacion_rival', 'resultado', 'localia_peru'] # Define las variables nominales a mostrar
        for var_name in nominal_vars_to_show: # Itera sobre cada variable nominal
            st.write(f'\n📊 Porcentajes para "{var_name}":') # Muestra el título para la variable actual
            if var_name in df_value_counts.columns: # Verifica si la columna existe en el DataFrame
                try:
                    counts_series = df_value_counts[var_name].value_counts(normalize=True) * 100 # Calcula los porcentajes
                    counts_df = counts_series.reset_index() # Convierte la serie a DataFrame
                    counts_df.columns = [var_name, 'Porcentaje (%)'] # Renombra las columnas del DataFrame de porcentajes
                    st.dataframe(counts_df) # Muestra el DataFrame de porcentajes
                except Exception as e:
                    st.error(f"Error calculating or displaying value counts for {var_name}: {e}") # Maneja errores al calcular o mostrar los conteos
            else:
                st.warning(f"Column '{var_name}' not found in the DataFrame used for value counts calculation in Streamlit.") # Advertencia si la columna no se encuentra

    except FileNotFoundError:
        st.error("Error: The file 'peru_match_results.csv' was not found in Streamlit app to calculate value counts.") # Maneja el error si el archivo no se encuentra
    except Exception as e:
        st.error(f"An error occurred during direct value counts calculation in Streamlit: {e}") # Maneja otras excepciones

    st.subheader("Discretizando variables numéricas continuas a categóricas") # Subtítulo
    st.markdown("La variable 'altitud_sede' fue discretizada en categorías: Bajo, Moderado, Alto, Extremo.") # Descripción de la discretización

    st.subheader("Convirtiendo Booleano a numérico") # Subtítulo
    st.markdown("La variable 'partido_oficial' fue convertida a numérico (0/1).") # Descripción de la conversión

    st.subheader("Transformando variables categoricas usando Label Encoding y Frequency Encoding") # Subtítulo
    st.markdown("""
    Las variables categóricas 'categoria_altitud', 'confederacion_rival', 'resultado' y 'localia_peru' fueron transformadas usando Label Encoding o Factorize.
    La variable 'rival' fue transformada usando Frequency Encoding.
    """) # Descripción de las transformaciones categóricas
    st.subheader("Head del DataFrame después de transformaciones") # Subtítulo
    try:
        df_head_trans_path = os.path.join(output_dir, 'data_info_df_head_after_transformations.txt') # Ruta del archivo con el head transformado (saved as txt)
        if os.path.exists(df_head_trans_path): # Verifica si el archivo existe
            with open(df_head_trans_path, 'r') as f:
                df_head_trans_content = f.read() # Lee el contenido del archivo
            st.code(df_head_trans_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {df_head_trans_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying dataframe head from {df_head_trans_path}: {e}") # Maneja errores al cargar o mostrar el DataFrame

    st.subheader("Head del DataFrame después del escalado") # Subtítulo
    try:
        df_head_scaled_path = os.path.join(output_dir, 'data_info_df_head_after_scaling.txt') # Ruta del archivo con el head escalado (saved as txt)
        if os.path.exists(df_head_scaled_path): # Verifica si el archivo existe
            with open(df_head_scaled_path, 'r') as f:
                df_head_scaled_content = f.read() # Lee el contenido del archivo
            st.code(df_head_scaled_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {df_head_scaled_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying dataframe head from {df_head_scaled_path}: {e}") # Maneja errores al cargar o mostrar el DataFrame


with tab4: # Define el contenido para la cuarta pestaña
    st.header("4. Modelado con Keras / TensorFlow") # Título de la sección
    st.subheader("División del Conjunto de Datos") # Subtítulo
    st.markdown("""
    Los datos se dividen en conjuntos de entrenamiento y prueba (80/20) con estratificación. # Describe la estrategia de división de datos
    El conjunto de entrenamiento se subdivide para validación (para KerasTuner). # Explica la subdivisión para KerasTuner
    """)

    st.code("""
    Resumen de división de datos (Originalmente calculado):
    Entrenamiento completo: (76, 5)
    Subconjunto de Entrenamiento (para KerasTuner): (60, 5)
    Subconjunto de Validación (para KerasTuner): (16, 5)
    Prueba (evaluación final): (20, 5)
    """) # Muestra un resumen de las dimensiones de los conjuntos de datos

    st.subheader("Implementación con Keras (MLP)") # Subtítulo
    st.markdown("""
    Se define y entrena un modelo de red neuronal (MLP) utilizando Keras. # Describe el tipo de modelo y la herramienta
    Se utilizó KerasTuner para optimizar hiperparámetros (número de neuronas, dropout, tasa de aprendizaje). # Menciona el uso de KerasTuner para optimización
    """)

    st.subheader("Mejores hiperparámetros encontrados por KerasTuner") # Subtítulo
    try:
        keras_hps_path = os.path.join(output_dir, 'keras_results_best_hyperparameters.txt') # Ruta al archivo de hiperparámetros
        if os.path.exists(keras_hps_path): # Verifica si el archivo existe
            st.markdown("Detalle de los mejores hiperparámetros (Keras):") # Mensaje descriptivo
            with open(keras_hps_path, 'r') as f:
                keras_hps_content = f.read() # Lee el contenido del archivo
            st.code(keras_hps_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {keras_hps_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying Keras hyperparameters from {keras_hps_path}: {e}") # Maneja errores al cargar o mostrar los hiperparámetros


with tab5: # Define el contenido para la quinta pestaña
    st.header("5. Modelado con PyTorch") # Título de la sección
    st.subheader("Preparación de Datos para PyTorch") # Subtítulo
    st.markdown("Los datos de entrenamiento y validación fueron convertidos a Tensores de PyTorch.") # Describe la preparación de datos para PyTorch

    st.subheader("Arquitectura y Bucle de Entrenamiento en PyTorch") # Subtítulo
    st.markdown("""
    Se define una red neuronal (MLP) utilizando `torch.nn.Module` y se entrena utilizando `skorch` para la integración con scikit-learn y GridSearchCV.
    """) # Describe la arquitectura y el proceso de entrenamiento en PyTorch

    st.subheader("Mejores hiperparámetros encontrados por GridSearchCV (PyTorch)") # Subtítulo
    try:
        pytorch_hps_path = os.path.join(output_dir, 'pytorch_results_best_hyperparameters.txt') # Ruta al archivo de hiperparámetros
        if os.path.exists(pytorch_hps_path): # Verifica si el archivo existe
            st.markdown("Detalle de los mejores hiperparámetros (PyTorch):") # Mensaje descriptivo
            with open(pytorch_hps_path, 'r') as f:
                pytorch_hps_content = f.read() # Lee el contenido del archivo
            st.code(pytorch_hps_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {pytorch_hps_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying PyTorch hyperparameters from {pytorch_hps_path}: {e}") # Maneja errores al cargar o mostrar los hiperparámetros


with tab6: # Define el contenido para la sexta pestaña
    st.header("6. Evaluación del modelo y selección") # Título de la sección

    st.subheader("Curvas de Aprendizaje - Keras Optimizado") # Subtítulo
    try:
        plot_path = os.path.join(output_dir, 'keras_results_learning_curves_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Reporte de Clasificación - Keras Optimizado") # Subtítulo
    try:
        keras_report_path = os.path.join(output_dir, 'keras_results_classification_report.txt') # Ruta del archivo del reporte
        if os.path.exists(keras_report_path): # Verifica si el archivo existe
            with open(keras_report_path, 'r') as f:
                keras_report_content = f.read() # Lee el contenido del reporte
            st.code(keras_report_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {keras_report_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying report from {keras_report_path}: {e}") # Mensaje de error

    st.subheader("Matriz de Confusión - Keras Optimizado") # Subtítulo
    try:
        plot_path = os.path.join(output_dir, 'keras_results_confusion_matrix_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Curvas de Aprendizaje - PyTorch Optimizado") # Subtítulo
    try:
        plot_path = os.path.join(output_dir, 'pytorch_results_learning_curves_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Comparación de Métricas de Evaluación") # Subtítulo
    try:
        metrics_comp_path = os.path.join(output_dir, 'comparisons_metrics_table.csv') # Ruta del archivo de la tabla
        if os.path.exists(metrics_comp_path): # Verifica si el archivo existe
            metrics_comp_df = pd.read_csv(metrics_comp_path, index_col=0) # Carga la tabla de métricas
            st.dataframe(metrics_comp_df) # Muestra el DataFrame
        else:
            st.warning(f"File not found: {metrics_comp_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying metrics comparison table from {metrics_comp_path}: {e}") # Mensaje de error

    st.subheader("Matriz de Confusión - PyTorch Optimizado") # Subtítulo
    try:
        plot_path = os.path.join(output_dir, 'pytorch_results_confusion_matrix_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Reporte de Clasificación - PyTorch Optimizado") # Subtítulo
    try:
        pytorch_report_path = os.path.join(output_dir, 'pytorch_results_classification_report.txt') # Ruta del archivo del reporte
        if os.path.exists(pytorch_report_path): # Verifica si el archivo existe
            with open(pytorch_report_path, 'r') as f:
                pytorch_report_content = f.read() # Lee el contenido del reporte
            st.code(pytorch_report_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {pytorch_report_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying report from {pytorch_report_path}: {e}") # Mensaje de error

    st.subheader("Comparación de Tiempos de Ajuste (Tuning)") # Subtítulo
    try:
        time_comp_path = os.path.join(output_dir, 'comparisons_time_table.csv') # Ruta del archivo de la tabla
        if os.path.exists(time_comp_path): # Verifica si el archivo existe
            try:
                time_comp_df = pd.read_csv(time_comp_path, index_col=0) # Carga la tabla de tiempos
                st.dataframe(time_comp_df) # Muestra el DataFrame
            except pd.errors.EmptyDataError:
                st.warning(f"File '{time_comp_path}' is empty.") # Advertencia si el archivo está vacío
            except Exception as e:
                st.error(f"Error loading or displaying time comparison table from {time_comp_path}: {e}") # Mensaje de error
        else:
            st.warning(f"File not found: {time_comp_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"An unexpected error occurred with time comparison table loading: {e}") # Mensaje de error

    st.subheader("Comparación Cualitativa entre Librerías (Keras vs PyTorch)") # Subtítulo
    try:
        qual_comp_path = os.path.join(output_dir, 'comparisons_qualitative_table.csv') # Ruta del archivo de la tabla
        if os.path.exists(qual_comp_path): # Verifica si el archivo existe
            qual_comp_df = pd.read_csv(qual_comp_path, index_col=0) # Carga la tabla cualitativa
            st.dataframe(qual_comp_df) # Muestra el DataFrame
        else:
            st.warning(f"File not found: {qual_comp_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying qualitative comparison table from {qual_comp_path}: {e}") # Mensaje de error

    st.subheader("Comparación de Curvas de Aprendizaje (Keras vs PyTorch)") # Subtítulo
    st.write("Evolución de Pérdida y Precisión durante el Entrenamiento") # Descripción de la gráfica
    try:
        plot_path = os.path.join(output_dir, 'comparisons_learning_curves_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Visualización de Comparación de Métricas") # Subtítulo
    st.write("Comparación Gráfica de Métricas Clave (Accuracy, Precision, Recall, F1-Score)") # Descripción de la gráfica
    try:
        plot_path = os.path.join(output_dir, 'comparisons_metrics_bar_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Visualización de Tiempos de Ajuste") # Subtítulo
    st.write("Tiempo Requerido para la Búsqueda de Hiperparámetros") # Descripción de la gráfica
    try:
        plot_path = os.path.join(output_dir, 'comparisons_tuning_time_bar_plot.png') # Ruta del archivo de la gráfica
        if os.path.exists(plot_path): # Verifica si el archivo de la gráfica existe
            st.image(plot_path) # Muestra la imagen de la gráfica
        else:
            st.warning(f"Plot file not found: {plot_path}. Please ensure you have run the cell to save outputs.") # Advertencia si la gráfica no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying plot from {plot_path}: {e}") # Mensaje de error

    st.subheader("Hiperparámetros Óptimos Encontrados") # Subtítulo
    st.write("Mejores Hiperparámetros para Keras:") # Mensaje descriptivo
    try:
        keras_hps_path = os.path.join(output_dir, 'keras_results_best_hyperparameters.txt') # Ruta del archivo de hiperparámetros
        if os.path.exists(keras_hps_path): # Verifica si el archivo existe
            with open(keras_hps_path, 'r') as f:
                keras_hps_content = f.read() # Lee el contenido
            st.code(keras_hps_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {keras_hps_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying Keras hyperparameters from {keras_hps_path}: {e}") # Mensaje de error

    st.write("Mejores Hiperparámetros para PyTorch:") # Mensaje descriptivo
    try:
        pytorch_hps_path = os.path.join(output_dir, 'pytorch_results_best_hyperparameters.txt') # Ruta del archivo de hiperparámetros
        if os.path.exists(pytorch_hps_path): # Verifica si el archivo existe
            with open(pytorch_hps_path, 'r') as f:
                pytorch_hps_content = f.read() # Lee el contenido
            st.code(pytorch_hps_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {pytorch_hps_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying PyTorch hyperparameters from {pytorch_hps_path}: {e}") # Mensaje de error

    st.subheader("Comparación Detallada de Métricas y Tiempos") # Subtítulo
    try:
        detailed_comp_path = os.path.join(output_dir, 'comparisons_detailed_comparison_string.txt') # Ruta del archivo de comparación detallada
        if os.path.exists(detailed_comp_path): # Verifica si el archivo existe
            with open(detailed_comp_path, 'r') as f:
                detailed_comp_content = f.read() # Lee el contenido
            st.code(detailed_comp_content) # Muestra el contenido como código
        else:
            st.warning(f"File not found: {detailed_comp_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying detailed comparison from {detailed_comp_path}: {e}") # Mensaje de error

    st.subheader("Recomendación y Selección del Mejor Modelo") # Subtítulo
    try:
        recommendation_path = os.path.join(output_dir, 'comparisons_recommendation_string.txt') # Ruta del archivo de recomendación
        if os.path.exists(recommendation_path): # Verifica si el archivo existe
            with open(recommendation_path, 'r') as f:
                recommendation_content = f.read() # Lee el contenido
            recommendation_content_cleaned = recommendation_content.replace("================================================================================", "").strip() # Limpia el contenido
            st.markdown(recommendation_content_cleaned) # Muestra el contenido como Markdown
        else:
            st.warning(f"File not found: {recommendation_path}. Please ensure you have run the cell to save outputs.") # Advertencia si el archivo no se encuentra
    except Exception as e:
        st.error(f"Error loading or displaying recommendation from {recommendation_path}: {e}") # Mensaje de error


with tab7: # Define el contenido para la séptima pestaña
    st.header("7. Predicción") # Título de la sección
    st.markdown("Utilice el modelo PyTorch entrenado para predecir el resultado de un partido basado en las características ingresadas.") # Descripción de la funcionalidad

    if best_pytorch_model is None or preprocessing_pipeline is None: # Verifica si el modelo o el pipeline de preprocesamiento están cargados
        st.warning("El modelo o los datos de preprocesamiento no se pudieron cargar/entrenar. No se puede realizar la predicción.") # Mensaje de advertencia si no están disponibles
    else:
        st.subheader("Ingrese las características del partido:") # Subtítulo para la entrada de datos

        try:
            df_original_cols = pd.read_csv('peru_match_results.csv') # Carga el DataFrame original para obtener opciones
            df_original_cols = df_original_cols[df_original_cols['coach'] == 'Ricardo Gareca'].copy() # Filtra por entrenador
            # Mapea y renombra columnas relevantes para la interfaz de usuario
            rename_mapping_pred = {"rival_confederation":"confederacion_rival","peru_condition":"localia_peru","elevation":"altitud_sede","official":"partido_oficial"}
            df_original_cols.rename(columns=rename_mapping_pred, inplace=True) # Aplica el renombramiento

            # Use categories from the preprocessed data for selection options where possible
            rival_options = sorted(preprocessing_pipeline['freq_encoding'].keys()) # Get rivals from freq_encoding keys
            confederation_options = sorted(preprocessing_pipeline['confederacion_rival_categories'].tolist()) # Get confederations from categories
            condition_options = sorted(preprocessing_pipeline['localia_peru_categories'].tolist()) # Get conditions from categories

            min_elevation = int(df_original_cols['altitud_sede'].min()) # Obtiene la elevación mínima
            max_elevation = int(df_original_cols['altitud_sede'].max()) # Obtiene la elevación máxima
            official_options = [True, False] # Opciones para partido oficial

        except FileNotFoundError:
            st.error("Error loading original data to populate prediction input options.") # Maneja el error si el archivo no se encuentra
            # Establece opciones predeterminadas vacías o seguras en caso de error
            rival_options = []
            confederation_options = []
            condition_options = []
            min_elevation = 0
            max_elevation = 10000
            official_options = [True, False]
        except Exception as e:
            st.error(f"An error occurred while preparing prediction input options: {e}") # Maneja otras excepciones
            # Establece opciones predeterminadas vacías o seguras en caso de error
            rival_options = []
            confederation_options = []
            condition_options = []
            min_elevation = 0
            max_elevation = 10000
            official_options = [True, False]

        # Widgets de entrada de usuario
        input_rival = st.selectbox("Rival:", rival_options) # Selector para el rival
        input_confederation = st.selectbox("Confederación del Rival:", confederation_options) # Selector para la confederación del rival
        input_condition = st.selectbox("Condición de Perú:", condition_options) # Selector para la condición de Perú
        input_elevation = st.number_input(f"Altitud de la Sede (metros):", min_value=min_elevation, max_value=max_elevation, value=int(min_elevation + (max_elevation - min_elevation)/2)) # Entrada numérica para la altitud
        input_official = st.selectbox("¿Partido Oficial?:", official_options) # Selector para si es partido oficial

        predict_button = st.button("Predecir Resultado") # Botón para iniciar la predicción

        if predict_button: # Si el botón de predicción es presionado
            # Crea un DataFrame con los datos de entrada del usuario
            input_data = pd.DataFrame([{
                'rival': input_rival,
                'confederacion_rival': input_confederation,
                'localia_peru': input_condition,
                'altitud_sede': input_elevation,
                'partido_oficial': input_official,
            }])

            try:
                input_data['partido_oficial'] = input_data['partido_oficial'].astype(int) # Convierte 'partido_oficial' a entero

                # Discretiza la altitud de la sede
                input_data['categoria_altitud'] = pd.cut(
                    input_data['altitud_sede'],
                    bins=preprocessing_pipeline['altitude_bins'],
                    labels=preprocessing_pipeline['altitude_labels'],
                    include_lowest=True
                )
                # Transforma la categoría de altitud usando el LabelEncoder pre-entrenado
                input_data['categoria_altitud'] = preprocessing_pipeline['altitude_le'].transform(input_data['categoria_altitud'])

                input_data.drop('altitud_sede', axis=1, inplace=True) # Elimina la columna 'altitud_sede' original

                # Crea mapeos para codificar las variables categóricas
                confederacion_rival_mapping = {cat: i for i, cat in enumerate(preprocessing_pipeline['confederacion_rival_categories'])}
                localia_peru_mapping = {cat: i for i, cat in enumerate(preprocessing_pipeline['localia_peru_categories'])}

                # Aplica la codificación a las variables de entrada, manejando valores no vistos
                # For categories not seen in training, map to -1, then replace with a default (e.g., 0)
                input_data['confederacion_rival_num'] = input_data['confederacion_rival'].map(confederacion_rival_mapping).fillna(-1).astype(int)
                input_data['localia_peru_num'] = input_data['localia_peru'].map(localia_peru_mapping).fillna(-1).astype(int)

                if (input_data['confederacion_rival_num'] == -1).any():
                    st.warning("Confederación del Rival no vista en los datos de entrenamiento. Usando valor predeterminado (0).")
                    input_data['confederacion_rival_num'].replace(-1, 0, inplace=True) # Replace unseen with 0

                if (input_data['localia_peru_num'] == -1).any():
                    st.warning("Condición de Perú no vista en los datos de entrenamiento. Usando valor predeterminado (0).")
                    input_data['localia_peru_num'].replace(-1, 0, inplace=True) # Replace unseen with 0


                # Aplica Frequency Encoding a 'rival', usando la frecuencia mínima si el rival no se vio en el entrenamiento
                # Ensure freq_encoding has values before calling min()
                min_freq = min(preprocessing_pipeline['freq_encoding'].values()) if preprocessing_pipeline['freq_encoding'] else 0.0
                input_data['rival_encoded'] = input_data['rival'].map(preprocessing_pipeline['freq_encoding']).fillna(min_freq)


                # Elimina las columnas categóricas originales que ya fueron codificadas
                input_data.drop(['confederacion_rival', 'localia_peru', 'rival'], axis=1, inplace=True)

                # Asegura que las columnas del input_data estén en el mismo orden que las usadas para entrenar el modelo
                # Get the list of columns the model was trained on from the preprocessed data
                expected_cols = X_train_full.columns.tolist()
                input_data = input_data[expected_cols]

                # Escala los datos de entrada usando el scaler pre-entrenado
                input_data[expected_cols] = preprocessing_pipeline['scaler'].transform(input_data[expected_cols])

                # Convert input data to PyTorch tensor
                input_tensor = torch.tensor(input_data.values.astype(np.float32))


                # Perform prediction with the PyTorch model
                # Use the trained skorch model (best_pytorch_model)
                prediction_pytorch = best_pytorch_model.predict(input_tensor) # Skorch predict returns numpy array of predictions

                # The prediction is the class index (0, 1, or 2)
                predicted_class_index = prediction_pytorch[0] # Get the single prediction

                # Decodifica el resultado predicho a su etiqueta original (W, D, or L)
                predicted_result = preprocessing_pipeline['resultado_categories'][predicted_class_index]

                st.subheader("Resultado Predicho:") # Subtítulo para el resultado
                result_mapping_text = {'W': 'Victoria', 'D': 'Empate', 'L': 'Derrota'} # Mapeo de resultados a texto legible
                st.success(f"El resultado predicho es: **{result_mapping_text.get(predicted_result, predicted_result)}**") # Muestra el resultado predicho

            except Exception as e:
                st.error(f"Error durante la predicción: {e}") # Maneja cualquier error durante el proceso de predicción
