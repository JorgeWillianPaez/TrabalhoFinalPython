import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import time

def generate_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    # Distribuição de valores (numéricos)
    for col in df.select_dtypes(include="number").columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plot_path = os.path.join(output_dir, f"dist_{col}.png")
        plt.title(f"Distribuição de {col}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots[col] = plot_path

    # Correlação
    corr_path = os.path.join(output_dir, "correlation.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    plots["correlation"] = corr_path

    mapa_path = os.path.join(output_dir, "mapa_vendas.html")
    try:
        mapa = generate_sales_map(df)
        mapa.save(mapa_path)
        plots["mapa_vendas"] = mapa_path
    except Exception as e:
        print(f"Erro ao gerar mapa de vendas: {e}")

    return plots


# mapa de vendas
def generate_sales_map(df):
    import pandas as pd
    import numpy as np

    df.columns = df.columns.str.lower()

    if "city" not in df.columns:
        raise ValueError("ERRO: A base não contém coluna 'city'.")

    # Agrupa vendas por cidade
    vendas = df.groupby("city").size().reset_index(name="qtd_vendas")

    # Carrega coordenadas 
    coord_path = os.path.join("data", "coordenadas.csv")
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"Arquivo {coord_path} não encontrado.")

    coordenadas = pd.read_csv(coord_path, sep=";")
    coordenadas.columns = coordenadas.columns.str.lower()

    obrigatorias = {"city", "latitude", "longitude"}
    if not obrigatorias.issubset(coordenadas.columns):
        raise ValueError("ERRO: O CSV de coordenadas precisa ter city, latitude e longitude.")

    # Merge das vendas com coordenadas
    vendas_coord = vendas.merge(coordenadas, on="city", how="left")

    # Criação do mapa
    mapa = folium.Map(location=[-15.78, -47.93], zoom_start=5, tiles="cartodbpositron")
    cluster = MarkerCluster().add_to(mapa)

    for _, row in vendas_coord.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        radius = np.log1p(row["qtd_vendas"]) * 4

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.25,
            popup=f"{row['city']}: {row['qtd_vendas']} vendas",
            tooltip=row["city"]
        ).add_to(cluster)

    return mapa