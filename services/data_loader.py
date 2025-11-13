import pandas as pd
import numpy as np
import re

def clean_dataset(filepath):
    print(f"Lendo arquivo: {filepath}\n")
    
    df = pd.read_csv(filepath, encoding="utf-8")
    print("Arquivo lido com sucesso!\n")
    
    # corrige cabeçalhos 
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.lower()
    )
    
    # conversão automática para datas 
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    
    print("Valores nulos por coluna:")
    print(df.isnull().sum(), "\n")
    
    # preenchimento 
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col].fillna(df[col].min(), inplace=True)
        else:
            df[col].fillna("desconhecido", inplace=True)
    
    # remover duplicatas
    duplicadas = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Duplicatas removidas: {duplicadas}")
    
    # limpar strings
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    
    # converte números em strings para numéricos
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors='ignore')
            except Exception:
                pass
    
    # correção específica para 'delivery_time_days'
    if 'delivery_time_days' in df.columns:
        col = 'delivery_time_days'
        
        df[col] = df[col].astype(str)
        
        df[col] = df[col].apply(
            lambda x: int(re.search(r'\.(\d+)$', x).group(1)) if re.search(r'\.(\d+)$', x) else np.nan
        )
        
        df[col].fillna(df[col].median(), inplace=True)
        df[col] = df[col].astype(int)
        
        print(f"\nColuna '{col}' corrigida")
        print(df[col].head(10))
    else:
        print("Coluna 'delivery_time_days' não encontrada.\n")
    
    print("\nInformações finais:")
    print(df.info())
    print("\nPrimeiras linhas:")
    print(df.head(), "\n")
    
    return df


# carrega e limpa o arquivo CSV
def load_csv(filepath):
    try:
        df = clean_dataset(filepath)  
        
        if df.empty:
            raise ValueError("O arquivo CSV está vazio após limpeza.")
        
        return df
    except Exception as e:
        raise ValueError(f"Erro ao processar CSV: {e}")


