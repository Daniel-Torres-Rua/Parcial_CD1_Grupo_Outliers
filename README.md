# Parcial_CD1_Grupo_Outliers
Trabajo de medio curso del curso de Ciencia de Datos 1
# Análisis y Modelado de Resultados de Partidos de la Selección Peruana bajo la Dirección de Ricardo Gareca

## Introducción

Este proyecto tiene como objetivo analizar el rendimiento de la selección nacional de fútbol del Perú durante el período en que fue dirigida por el entrenador Ricardo Gareca, utilizando datos históricos de partidos. Se explora el conjunto de datos para entender las características de los partidos y los resultados obtenidos. Posteriormente, se desarrollan y evalúan modelos de clasificación utilizando las librerías de aprendizaje profundo Keras (con TensorFlow) y PyTorch (con Skorch) para predecir el resultado de un partido. Finalmente, se presenta una aplicación interactiva construida con Streamlit para permitir la predicción de resultados basada en nuevas entradas.

## Contenido del Repositorio

Este repositorio contiene los siguientes archivos y directorios:

-   `peru_match_results.csv`: El conjunto de datos original utilizado para el análisis y modelado.
-   `Nombre_del_Notebook.ipynb`: El cuaderno de Google Colab que documenta todo el proceso, desde la carga y preprocesamiento de datos hasta el modelado, evaluación y comparación de los enfoques con Keras y PyTorch.
-   `mi_app.py`: El script de Python para la aplicación interactiva Streamlit.
-   `existing_analysis_outputs/`: Directorio que contiene los resultados del análisis y modelado, incluyendo gráficos (EDA, curvas de aprendizaje, matrices de confusión), tablas comparativas de métricas y tiempos, y detalles de los hiperparámetros óptimos encontrados.

## Metodología

El proyecto siguió las siguientes etapas:

1.  **Recopilación y Carga de Datos:** Se utilizó el dataset `peru_match_results.csv` de Kaggle.
2.  **Análisis Exploratorio de Datos (EDA):** Se realizaron visualizaciones para comprender la distribución de resultados y su relación con la condición de localía y la confederación del rival.
3.  **Preprocesamiento y Transformación de Datos:**
    *   Filtrado de partidos bajo la dirección de Ricardo Gareca.
    *   Eliminación de columnas irrelevantes o con alta proporción de valores faltantes.
    *   Discretización de la altitud de la sede.
    *   Conversión de variables booleanas a numéricas.
    *   Aplicación de Label Encoding y Frequency Encoding para variables categóricas.
    *   Escalado de las características numéricas.
4.  **Modelado:**
    *   Se implementaron y entrenaron modelos de Red Neuronal Multicapa (MLP) utilizando Keras (con KerasTuner para optimización de hiperparámetros) y PyTorch (con Skorch y GridSearchCV).
5.  **Evaluación y Comparación:** Se evaluaron los modelos optimizados utilizando métricas cuantitativas (Accuracy, Precision, Recall, F1-Score) y visualizaciones (curvas de aprendizaje, matrices de confusión). Se compararon los resultados y los tiempos de entrenamiento/tuning.
6.  **Selección del Modelo:** Se seleccionó el modelo con el mejor rendimiento basado en las métricas evaluadas (en este caso, el modelo PyTorch mostró un mejor F1-Score en el conjunto de prueba).
7.  **Aplicación Interactiva:** Se desarrolló una aplicación Streamlit para demostrar el uso del modelo seleccionado (PyTorch) en la predicción de resultados de partidos futuros.

## Resultados Clave

-   El análisis exploratorio reveló patrones interesantes en los resultados de los partidos de Perú bajo Gareca, relacionados con la localía y la confederación del rival.
-   Ambos enfoques (Keras y PyTorch) lograron construir modelos capaces de predecir el resultado del partido.
-   La comparación detallada de métricas y tiempos, así como las curvas de aprendizaje y matrices de confusión (disponibles en el directorio `existing_analysis_outputs/`), permitieron justificar la selección del modelo PyTorch como el de mejor rendimiento para este problema específico, aunque Keras demostró ser más rápido en el proceso de tuning en este caso particular.

## Cómo Ejecutar el Proyecto

### Usando Google Colab

La forma más sencilla de explorar y ejecutar este proyecto es a través del cuaderno de Google Colab (`Nombre_del_Notebook.ipynb`).

1.  Abra el cuaderno en Google Colab.
2.  Ejecute todas las celdas en orden. Esto cargará los datos, realizará el preprocesamiento, entrenará los modelos, generará los resultados de evaluación y creará los archivos necesarios para la aplicación Streamlit.
3.  Para ejecutar la aplicación Streamlit, siga las instrucciones en la sección "7. Aplicacion Interactiva" del cuaderno, que incluye instalar las dependencias y ejecutar el script `mi_app.py` con localtunnel.

### Localmente (Requiere Python y las librerías instaladas)

Para ejecutar el código localmente, necesitará tener Python instalado junto con las librerías listadas al inicio del cuaderno (pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, keras, torch, skorch, streamlit, etc.).

1.  Clone este repositorio.
2.  Instale las dependencias: `pip install -r requirements.txt` (Nota: Deberá crear un archivo `requirements.txt` listando las dependencias).
3.  Ejecute el script de preprocesamiento y modelado (puede adaptar el cuaderno `.ipynb` a un script `.py`).
4.  Ejecute la aplicación Streamlit: `streamlit run mi_app.py`

## Visualizaciones y Resultados Adicionales

El directorio `existing_analysis_outputs/` contiene los gráficos generados durante el EDA, las curvas de aprendizaje y matrices de confusión de los modelos, así como tablas y resúmenes de la comparación de métricas y tiempos.

## Conclusión

Este proyecto demuestra la aplicación de técnicas de análisis de datos y modelado con redes neuronales utilizando Keras y PyTorch para predecir resultados de partidos de fútbol. La comparación entre ambas librerías resalta sus fortalezas y debilidades en un caso de uso práctico. La aplicación Streamlit proporciona una interfaz interactiva para poner el modelo en funcionamiento.

---

**Autor:** [* Bravo Yataco Luiggi Mariano
* Llamoca León Israel
* Poma Gómez Diego Alonso
* Torres Rua Daniel Isaias]
**Fecha:** [1/07/2025]
