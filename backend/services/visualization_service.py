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

    # 1. Distribuição de valores (numéricos)
    for col in df.select_dtypes(include="number").columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plot_path = os.path.join(output_dir, f"dist_{col}.png")
        plt.title(f"Distribuição de {col}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots[col] = plot_path

    # 2. Correlação
    corr_path = os.path.join(output_dir, "correlation.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()
    plots["correlation"] = corr_path

    city_col = None
    for candidate in ["city", "cidade", "City", "Cidade"]:
        if candidate in df.columns:
            city_col = candidate
            break

    if city_col:
        mapa_path = os.path.join(output_dir, "mapa_cidades.html")
        mapa = generate_city_map(df, city_col)
        mapa.save(mapa_path)
        plots["mapa_cidades"] = mapa_path

    return plots


def generate_city_map(df, city_col):
    """
    Gera um mapa interativo com base na coluna de cidades.
    As cidades são geocodificadas e exibidas com contagem agregada.
    """
    geolocator = Nominatim(user_agent="data_analysis_app")
    mapa = folium.Map(location=[-15.78, -47.93], zoom_start=4, tiles="cartodbpositron")  # centro do Brasil
    cluster = MarkerCluster().add_to(mapa)

    # Agrupar por cidade (contagem de registros)
    city_counts = df[city_col].value_counts().to_dict()

    for city, count in city_counts.items():
        try:
            location = geolocator.geocode(city + ", Brazil")
            if location:
                folium.Marker(
                    location=[location.latitude, location.longitude],
                    popup=f"{city}: {count} registros",
                    tooltip=city
                ).add_to(cluster)
                time.sleep(1)  # evita bloqueio do serviço de geocodificação
        except Exception as e:
            print(f"Erro ao localizar {city}: {e}")

    return mapa
