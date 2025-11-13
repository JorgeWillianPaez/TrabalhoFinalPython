from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from config import Config
from utils.file_utils import save_file
from services.data_loader import load_csv
from services.data_analysis import get_basic_stats
from services.visualization_service import generate_visualizations
from services.model_training import (
    train_model, 
    train_both_models, 
    list_models,
    load_model
)
import os
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Permite requisições do frontend

@app.route("/")
def index():
    return jsonify({
        "message": "API de Análise de Dados com Flask e Machine Learning",
        "endpoints": {
            "/upload": "POST - envia um arquivo CSV para análise",
            "/analyze": "GET - exibe estatísticas e gráficos do último arquivo enviado",
            "/train": "POST - treina um modelo de ML",
            "/train/both": "POST - treina modelos de regressão e classificação",
            "/models": "GET - lista todos os modelos treinados",
            "/models/<model_id>": "GET - obtém informações de um modelo específico",
            "/columns": "GET - lista as colunas do último arquivo enviado"
        }
    })

# Guarda o último arquivo carregado em memória simples (poderia ser DB futuramente)
last_uploaded_file = None

@app.route("/upload", methods=["POST"])
def upload_file():
    global last_uploaded_file
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    try:
        filepath = save_file(file, app.config["UPLOAD_FOLDER"])
        last_uploaded_file = filepath
        return jsonify({"message": "Arquivo enviado com sucesso", "path": filepath})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/analyze", methods=["GET"])
def analyze():
    global last_uploaded_file
    if not last_uploaded_file or not os.path.exists(last_uploaded_file):
        return jsonify({"error": "Nenhum arquivo CSV disponível para análise"}), 400

    df = load_csv(last_uploaded_file)
    stats = get_basic_stats(df)
    plots = generate_visualizations(df, os.path.join("static", "plots"))

    return jsonify({
        "status": "ok",
        "arquivo_analisado": os.path.basename(last_uploaded_file),
        "estatisticas": stats,
        "graficos_gerados": list(plots.values())
    })


@app.route("/download/<path:filename>")
def download_plot(filename):
    filepath = os.path.join("static", "plots", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "Arquivo não encontrado"}), 404


@app.route("/columns", methods=["GET"])
def get_columns():
    """Retorna as colunas do último arquivo enviado."""
    global last_uploaded_file
    if not last_uploaded_file or not os.path.exists(last_uploaded_file):
        return jsonify({"error": "Nenhum arquivo CSV disponível"}), 400
    
    try:
        df = load_csv(last_uploaded_file)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        
        return jsonify({
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "shape": df.shape
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    """Treina um modelo de regressão ou classificação."""
    global last_uploaded_file
    if not last_uploaded_file or not os.path.exists(last_uploaded_file):
        return jsonify({"error": "Nenhum arquivo CSV disponível para treinamento"}), 400

    try:
        data = request.get_json() or {}
        
        model_type = data.get("model_type", "regression")  # 'regression' ou 'classification'
        target_col = data.get("target_col")
        algorithm = data.get("algorithm", "rf")
        params = data.get("params")
        test_size = data.get("test_size", 0.2)
        random_state = data.get("random_state", 42)
        
        if not target_col:
            return jsonify({"error": "target_col é obrigatório"}), 400
        
        # Mapeia nomes de algoritmos do frontend para o backend
        algorithm_map = {
            "random_forest": "rf",
            "random_forest_reg": "rf",
            "linear_regression": "linreg",
            "logistic_regression": "logreg",
            "knn": "knn"
        }
        algorithm = algorithm_map.get(algorithm, algorithm)
        
        model_info = train_model(
            csv_path=last_uploaded_file,
            model_type=model_type,
            target_col=target_col,
            algorithm=algorithm,
            params=params,
            test_size=test_size,
            random_state=random_state
        )
        
        return jsonify({
            "message": "Treinamento concluído com sucesso!",
            "model": model_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train/both", methods=["POST"])
def train_both():
    """Treina modelos de regressão e classificação simultaneamente."""
    global last_uploaded_file
    if not last_uploaded_file or not os.path.exists(last_uploaded_file):
        return jsonify({"error": "Nenhum arquivo CSV disponível para treinamento"}), 400

    try:
        data = request.get_json() or {}
        
        target_reg = data.get("target_reg")
        target_clf = data.get("target_clf")
        reg_algorithm = data.get("reg_algorithm", "rf")
        clf_algorithm = data.get("clf_algorithm", "rf")
        reg_params = data.get("reg_params")
        clf_params = data.get("clf_params")
        test_size = data.get("test_size", 0.2)
        random_state = data.get("random_state", 42)
        
        if not target_reg or not target_clf:
            return jsonify({"error": "target_reg e target_clf são obrigatórios"}), 400
        
        model_info = train_both_models(
            csv_path=last_uploaded_file,
            target_reg=target_reg,
            target_clf=target_clf,
            reg_algorithm=reg_algorithm,
            clf_algorithm=clf_algorithm,
            reg_params=reg_params,
            clf_params=clf_params,
            test_size=test_size,
            random_state=random_state
        )
        
        return jsonify({
            "message": "Ambos os modelos treinados com sucesso!",
            "models": model_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def get_models():
    """Lista todos os modelos treinados."""
    try:
        models = list_models()
        return jsonify({
            "models": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models/<model_id>", methods=["GET"])
def get_model(model_id):
    """Obtém informações de um modelo específico."""
    try:
        model, metadata, label_encoder = load_model(model_id)
        # Remove o modelo do retorno (muito grande para JSON)
        metadata.pop("model_path", None)
        if "regression" in metadata:
            metadata["regression"].pop("model_path", None)
        if "classification" in metadata:
            metadata["classification"].pop("model_path", None)
            metadata["classification"].pop("encoder_path", None)
        
        return jsonify({
            "model_id": model_id,
            "metadata": metadata,
            "has_model": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/predict", methods=["POST"])
def predict():
    """Faz predições usando um modelo treinado."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Dados não fornecidos"}), 400
        
        model_id = data.get("model_id")
        model_type = data.get("model_type")  # 'regression' ou 'classification'
        features = data.get("features")  # Dicionário com valores das features
        
        if not model_id:
            return jsonify({"error": "model_id é obrigatório"}), 400
        
        if not features:
            return jsonify({"error": "features é obrigatório"}), 400
        
        if not model_type:
            return jsonify({"error": "model_type é obrigatório"}), 400
        
        from services.model_training import predict_with_model
        result = predict_with_model(model_id, features, model_type=model_type)
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao fazer predição: {str(e)}"}), 500


@app.route("/models/<model_id>/features", methods=["GET"])
def get_model_features(model_id):
    """Obtém a lista de features necessárias para um modelo."""
    try:
        model, metadata, label_encoder = load_model(model_id)
        
        # Determina o tipo do modelo
        model_type = metadata.get("model_type")
        
        # Se o modelo tem ambos os tipos, precisa especificar
        if model_type is None:
            model_type_param = request.args.get("model_type")
            if not model_type_param:
                return jsonify({
                    "error": "model_type é obrigatório para modelos com ambos os tipos"
                }), 400
            model_type = model_type_param
        
        # Obtém features do metadata
        # Se o modelo tem ambos os tipos (regression e classification), busca dentro da chave correspondente
        if "regression" in metadata and "classification" in metadata:
            # Modelo com ambos os tipos
            if model_type == "regression":
                numeric_features = metadata.get("regression", {}).get("numeric_features", [])
                categorical_features = metadata.get("regression", {}).get("categorical_features", [])
            else:  # classification
                numeric_features = metadata.get("classification", {}).get("numeric_features", [])
                categorical_features = metadata.get("classification", {}).get("categorical_features", [])
        else:
            # Modelo único (regression ou classification)
            numeric_features = metadata.get("numeric_features", [])
            categorical_features = metadata.get("categorical_features", [])
        
        all_features = numeric_features + categorical_features
        
        return jsonify({
            "model_id": model_id,
            "model_type": model_type,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_features": all_features
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao obter features: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
