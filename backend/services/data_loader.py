import pandas as pd
import numpy as np
import re
from pathlib import Path


def limpar_dataset(caminho):
    """
    Limpa e prepara o dataset para uso em machine learning.
    
    Args:
        caminho: caminho para o arquivo CSV
        
    Returns:
        DataFrame limpo e processado
    """
    print(f"📂 Lendo arquivo: {caminho}\n")
    
    df = pd.read_csv(caminho, encoding="utf-8")
    print("✅ Arquivo lido com sucesso!\n")
    
    # === Corrige cabeçalhos ===
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.lower()
    )
    
    # === Conversão automática para datas ===
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # === Nulos ===
    print("💧 Valores nulos por coluna:")
    print(df.isnull().sum(), "\n")
    
    # === Preenchimento ===
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col].fillna(df[col].min(), inplace=True)
        else:
            df[col].fillna("desconhecido", inplace=True)
    
    # === Remove duplicatas ===
    duplicadas = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"🧬 Duplicatas removidas: {duplicadas}")
    
    # === Limpa strings ===
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    
    # === Converte números ===
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors='ignore')
            except:
                pass
    
    # === Correção específica ===
    if 'delivery_time_days' in df.columns:
        col = 'delivery_time_days'
        
        df[col] = df[col].astype(str)
        
        df[col] = df[col].apply(
            lambda x: int(re.search(r'\.(\d+)$', x).group(1)) if re.search(r'\.(\d+)$', x) else np.nan
        )
        
        df[col].fillna(df[col].median(), inplace=True)
        df[col] = df[col].astype(int)
        
        print(f"\n✅ Coluna '{col}' corrigida!")
        print(df[col].head(10))
    else:
        print("⚠ Coluna 'delivery_time_days' não encontrada.\n")
    
    # === Info final ===
    print("\n📊 Informações finais:")
    print(df.info())
    print("\nPrimeiras linhas:")
    print(df.head(), "\n")
    
    return df


def load_csv(filepath):
    """
    Carrega e limpa um arquivo CSV usando a função limpar_dataset().
    
    Args:
        filepath: caminho para o arquivo CSV
        
    Returns:
        DataFrame limpo e processado
        
    Raises:
        ValueError: Se o arquivo estiver vazio ou houver erro no processamento
    """
    try:
        df = limpar_dataset(filepath)  # Integração principal
        
        if df.empty:
            raise ValueError("O arquivo CSV está vazio após limpeza.")
        
        return df
    except Exception as e:
        raise ValueError(f"Erro ao processar CSV: {e}")


