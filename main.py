import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# Configuracion de la app
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("Clustering Interactivo con K-Means y PCA")
st.write(
    """
Sube tus datos, aplica **K-Means**, y observa como el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
"""
)

# --- Subir archivo ---
st.sidebar.header("Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numÃ©ricas
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols,
        )

        # ParÃ¡metros de clustering
        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # --- Datos y modelo ---
        X = data[selected_cols]
        kmeans = KMeans(
            n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        # kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        data["Cluster"] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f"PCA{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df["Cluster"] = data["Cluster"]

        # --- VisualizaciÃ³n antes del clustering ---
        st.subheader("Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df,
                x="PCA1",
                y="PCA2",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"],
            )
        else:
            fig_before = px.scatter_3d(
                pca_df,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"],
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- VisualizaciÃ³n despuÃ©s del clustering ---
        st.subheader(f"Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df,
                x="PCA1",
                y="PCA2",
                color=pca_df["Cluster"].astype(str),
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                color=pca_df["Cluster"].astype(str),
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides ---
        st.subheader("Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(
            pca.transform(kmeans.cluster_centers_), columns=pca_cols
        )
        st.dataframe(centroides_pca)

        # --- MÃ©todo del Codo ---
        st.subheader("Método del Codo (Elbow Method)")
        if st.button("Calcular número Óptimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plt.plot(K, inertias, "bo-")
            plt.title("Método del Codo")
            plt.xlabel("Número de Clusters (k)")
            plt.ylabel("Inercia (SSE)")
            plt.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv",
        )

else:
    st.info("Carga un archivo CSV en la barra lateral para comenzar.")
    st.write(
        """
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |----------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """
    )
