import json
import os
import joblib
from datetime import datetime
from pathlib import Path
from services.data_loader import load_csv

from ml.ml_module import (
    train_regression_model,
    train_classification_model,
    train_all_models,
    predict_regression,
    predict_classification
)

# MODEL_DIR deve estar no diretório backend
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Treina um modelo de machine learning usando o ml_module.py
def train_model(csv_path, model_type="regression", target_col=None, 
                algorithm="rf", params=None, test_size=0.2, random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Arquivo CSV não encontrado para treinamento.")

    df = load_csv(csv_path)
    
    if target_col is None:
        raise ValueError("target_col é obrigatório para treinamento.")
    
    # Normaliza o nome da coluna target (mesmo padrão usado no load_csv)
    target_col_normalized = target_col.strip().lower().replace(' ', '_').replace('-', '_')
    
    if target_col_normalized not in df.columns:
        # Lista colunas disponíveis para ajudar no debug
        available_cols = ', '.join(df.columns.tolist()[:10])  # Mostra até 10 colunas
        raise ValueError(
            f"Coluna '{target_col}' (normalizada: '{target_col_normalized}') não encontrada no DataFrame. "
            f"Colunas disponíveis: {available_cols}{'...' if len(df.columns) > 10 else ''}"
        )
    
    # Usa a coluna normalizada
    target_col = target_col_normalized

    timestamp = datetime.now().isoformat()
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        if model_type == "regression":
            # Mapeia algoritmo para formato do ml_module
            ml_algorithm = "rf" if algorithm == "rf" or algorithm == "random_forest_reg" else "linreg"
            
            result = train_regression_model(
                df=df,
                target_col=target_col,
                model_type=ml_algorithm,
                params=params,
                test_size=test_size,
                random_state=random_state
            )
            
            # Salva o modelo
            model_path = MODEL_DIR / f"{model_id}_regression.pkl"
            joblib.dump(result["model"], model_path)
            
            # Salva metadados
            metadata = {
                "model_id": model_id,
                "model_type": "regression",
                "algorithm": algorithm,
                "target_col": target_col,
                "timestamp": timestamp,
                "model_path": str(model_path),
                "metrics": result["metrics"],
                "numeric_features": result["numeric_features"],
                "categorical_features": result["categorical_features"],
                "n_samples_train": result["n_samples_train"],
                "n_samples_test": result["n_samples_test"],
                "y_test_sample": result.get("y_test_sample", []),
                "y_pred_sample": result.get("y_pred_sample", [])
            }
            
        elif model_type == "classification":
            # Mapeia algoritmo para formato do ml_module
            algorithm_map = {
                "rf": "rf",
                "random_forest": "rf",
                "logreg": "logreg",
                "logistic_regression": "logreg",
                "knn": "knn"
            }
            ml_algorithm = algorithm_map.get(algorithm, "rf")
            
            result = train_classification_model(
                df=df,
                target_col=target_col,
                model_type=ml_algorithm,
                params=params,
                test_size=test_size,
                random_state=random_state
            )
            
            # Salva o modelo e o label encoder
            model_path = MODEL_DIR / f"{model_id}_classification.pkl"
            encoder_path = MODEL_DIR / f"{model_id}_encoder.pkl"
            joblib.dump(result["model"], model_path)
            joblib.dump(result["label_encoder"], encoder_path)
            
            # Salva metadados
            metadata = {
                "model_id": model_id,
                "model_type": "classification",
                "algorithm": algorithm,
                "target_col": target_col,
                "timestamp": timestamp,
                "model_path": str(model_path),
                "encoder_path": str(encoder_path),
                "metrics": result["metrics"],
                "classes": result["classes_"],
                "numeric_features": result["numeric_features"],
                "categorical_features": result["categorical_features"],
                "n_samples_train": result["n_samples_train"],
                "n_samples_test": result["n_samples_test"],
                "n_classes": result["n_classes"],
                "y_test_sample": result.get("y_test_sample", []),
                "y_pred_sample": result.get("y_pred_sample", []),
                "y_proba_sample": result.get("y_proba_sample", [])
            }
            
        else:
            raise ValueError(f"model_type '{model_type}' não suportado. Use 'regression' ou 'classification'.")
        
        # Salva metadados em JSON
        metadata_path = MODEL_DIR / f"{model_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, default=str)
        
        metadata["status"] = "treinado com sucesso"
        return metadata
        
    except Exception as e:
        raise RuntimeError(f"Erro ao treinar modelo: {str(e)}") from e


# Treina modelos de regressão e classificação simultaneamente
def train_both_models(csv_path, target_reg, target_clf, 
                     reg_algorithm="rf", clf_algorithm="rf",
                     reg_params=None, clf_params=None,
                     test_size=0.2, random_state=42): 
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Arquivo CSV não encontrado para treinamento.")

    df = load_csv(csv_path)

    # Normaliza os nomes das colunas target (mesmo padrão usado no load_csv)
    target_reg_normalized = target_reg.strip().lower().replace(' ', '_').replace('-', '_')
    target_clf_normalized = target_clf.strip().lower().replace(' ', '_').replace('-', '_')
    
    if target_reg_normalized not in df.columns:
        available_cols = ', '.join(df.columns.tolist()[:10])
        raise ValueError(
            f"Coluna de regressão '{target_reg}' (normalizada: '{target_reg_normalized}') não encontrada. "
            f"Colunas disponíveis: {available_cols}{'...' if len(df.columns) > 10 else ''}"
        )
    
    if target_clf_normalized not in df.columns:
        available_cols = ', '.join(df.columns.tolist()[:10])
        raise ValueError(
            f"Coluna de classificação '{target_clf}' (normalizada: '{target_clf_normalized}') não encontrada. "
            f"Colunas disponíveis: {available_cols}{'...' if len(df.columns) > 10 else ''}"
        )

    timestamp = datetime.now().isoformat()
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Mapeia algoritmos
        reg_ml_algorithm = "rf" if reg_algorithm == "random_forest_reg" else "linreg"
        
        algorithm_map = {
            "random_forest": "rf",
            "logistic_regression": "logreg",
            "knn": "knn"
        }
        clf_ml_algorithm = algorithm_map.get(clf_algorithm, "rf")
        
        results = train_all_models(
            df=df,
            target_reg=target_reg_normalized,
            target_clf=target_clf_normalized,
            reg_model_type=reg_ml_algorithm,
            clf_model_type=clf_ml_algorithm,
            reg_params=reg_params,
            clf_params=clf_params,
            test_size=test_size,
            random_state=random_state
        )
        
        # Salva modelos
        reg_model_path = MODEL_DIR / f"{model_id}_regression.pkl"
        clf_model_path = MODEL_DIR / f"{model_id}_classification.pkl"
        clf_encoder_path = MODEL_DIR / f"{model_id}_encoder.pkl"
        
        joblib.dump(results["regression"]["model"], reg_model_path)
        joblib.dump(results["classification"]["model"], clf_model_path)
        joblib.dump(results["classification"]["label_encoder"], clf_encoder_path)
        
        # Salva metadados
        metadata = {
            "model_id": model_id,
            "timestamp": timestamp,
            "regression": {
                "model_path": str(reg_model_path),
                "target_col": target_reg_normalized,
                "algorithm": reg_algorithm,
                "metrics": results["regression"]["metrics"],
                "n_samples_train": results["regression"]["n_samples_train"],
                "n_samples_test": results["regression"]["n_samples_test"]
            },
            "classification": {
                "model_path": str(clf_model_path),
                "encoder_path": str(clf_encoder_path),
                "target_col": target_clf_normalized,
                "algorithm": clf_algorithm,
                "metrics": results["classification"]["metrics"],
                "classes": results["classification"]["classes_"],
                "n_samples_train": results["classification"]["n_samples_train"],
                "n_samples_test": results["classification"]["n_samples_test"],
                "n_classes": results["classification"]["n_classes"]
            }
        }
        
        metadata_path = MODEL_DIR / f"{model_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, default=str)
        
        metadata["status"] = "ambos modelos treinados com sucesso"
        return metadata
        
    except Exception as e:
        raise RuntimeError(f"Erro ao treinar modelos: {str(e)}") from e

# Carrega um modelo treinado e seus metadados
def load_model(model_id):
    metadata_path = MODEL_DIR / f"{model_id}_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Modelo '{model_id}' não encontrado.")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    model_path = Path(metadata.get("model_path", metadata.get("regression", {}).get("model_path")))
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
    
    model = joblib.load(model_path)
    
    # Se for classificação, carrega também o encoder
    label_encoder = None
    if metadata.get("model_type") == "classification":
        encoder_path = Path(metadata.get("encoder_path"))
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
    elif "classification" in metadata:
        encoder_path = Path(metadata["classification"].get("encoder_path"))
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
    
    return model, metadata, label_encoder


#Faz predições usando um modelo treinado
def predict_with_model(model_id, data, model_type=None):
    import pandas as pd
    
    model, metadata, label_encoder = load_model(model_id)
    
    # Converte dict para DataFrame se necessário
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("data deve ser um DataFrame ou dicionário")
    
    # Detecta tipo do modelo se não fornecido
    if model_type is None:
        model_type = metadata.get("model_type")
        if model_type is None and "regression" in metadata:
            # Modelo com ambos os tipos
            raise ValueError("model_type deve ser especificado quando o modelo tem ambos os tipos")
    
    try:
        if model_type == "regression":
            predictions = predict_regression(model, data)
            return {
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "model_type": "regression",
                "model_id": model_id
            }
        elif model_type == "classification":
            predictions = predict_classification(
                model, 
                data, 
                label_encoder=label_encoder,
                return_proba=True
            )
            # Adiciona classes do metadata
            classes = metadata.get("classes", [])
            return {
                **predictions,
                "model_type": "classification",
                "model_id": model_id,
                "classes": classes
            }
        else:
            raise ValueError(f"model_type '{model_type}' não suportado")
            
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer predições: {str(e)}") from e

# Lista todos os modelos treinados
def list_models():
    models = []
    
    for metadata_file in MODEL_DIR.glob("*_metadata.json"):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                models.append(metadata)
        except Exception as e:
            print(f"Erro ao carregar {metadata_file}: {e}")
    
    # Ordena por timestamp (mais recente primeiro)
    models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return models
