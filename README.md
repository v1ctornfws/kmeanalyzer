📊 K-Means Interactivo con PCA (Aplicación Streamlit)

Este proyecto ofrece una aplicación web interactiva desarrollada con Streamlit que permite a los usuarios cargar un conjunto de datos CSV y aplicar el algoritmo de K-Means para realizar clustering (agrupamiento). Los resultados del clustering se visualizan inmediatamente en 2D o 3D utilizando Análisis de Componentes Principales (PCA) para la reducción de dimensionalidad.

✨ Características Principales

Carga de Datos: Sube cualquier archivo CSV que contenga datos numéricos.

Clustering K-Means: Aplica el algoritmo K-Means a las columnas numéricas seleccionadas.

Ajuste de Hiperparámetros: Control total sobre los parámetros clave de K-Means:

k (Número de Clusters)

init (Método de inicialización)

max_iter (Máximo de iteraciones)

n_init (Número de ejecuciones)

random_state (Semilla para reproducibilidad)

Reducción de Dimensionalidad: Utiliza PCA para proyectar los datos en un espacio de 2 o 3 dimensiones para una visualización clara.

Visualización Interactiva: Gráficos dinámicos creados con Plotly, que permiten rotar los resultados 3D y hacer zoom.

Método del Codo: Herramienta incluida para ayudar a estimar el número óptimo de clusters (k).

Exportación de Resultados: Descarga un CSV con los datos originales más la columna del cluster asignado a cada fila.

🚀 Cómo Ejecutar la Aplicación

Requisitos

Asegúrate de tener Python instalado (versión 3.8+ recomendada) y las siguientes librerías instaladas.

El archivo requirements.txt necesario para este proyecto es:

streamlit==1.50.0
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
plotly==6.4.0
matplotlib==3.10.7
scipy==1.16.2


Pasos de Instalación y Ejecución

Clonar el Repositorio:

git clone [TU_URL_DEL_REPOSITORIO]
cd [nombre-del-repositorio]


Crear y Activar un Entorno Virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate  # En Windows


Instalar Dependencias:

pip install -r requirements.txt


Ejecutar la Aplicación Streamlit:

streamlit run main.py


Esto abrirá la aplicación en tu navegador predeterminado (normalmente en http://localhost:8501).

⚙️ Estructura del Proyecto

main.py: Contiene toda la lógica de la aplicación Streamlit, el modelo K-Means, PCA y las visualizaciones.

requirements.txt: Lista de dependencias de Python necesarias para la ejecución.

README.md: Este archivo.

🤝 Contribuciones

Si encuentras algún error o tienes sugerencias de mejora (como añadir métricas de evaluación o preprocesamiento de datos), ¡no dudes en abrir un issue o enviar un pull request!