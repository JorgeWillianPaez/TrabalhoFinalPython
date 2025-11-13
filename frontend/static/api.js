// Configuração da API
const API_BASE_URL = 'http://localhost:5000'; // Ajuste conforme necessário

// Funções de API
const API = {
    // Upload de arquivo
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao fazer upload');
        }
        
        return await response.json();
    },
    
    // Obter colunas do dataset
    async getColumns() {
        const response = await fetch(`${API_BASE_URL}/columns`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao obter colunas');
        }
        
        return await response.json();
    },
    
    // Treinar modelo
    async trainModel(config) {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao treinar modelo');
        }
        
        return await response.json();
    },
    
    // Treinar ambos os modelos
    async trainBothModels(config) {
        const response = await fetch(`${API_BASE_URL}/train/both`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao treinar modelos');
        }
        
        return await response.json();
    },
    
    
    // Listar modelos
    async listModels() {
        const response = await fetch(`${API_BASE_URL}/models`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao listar modelos');
        }
        
        return await response.json();
    },
    
    // Obter informações de um modelo
    async getModel(modelId) {
        const response = await fetch(`${API_BASE_URL}/models/${modelId}`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao obter modelo');
        }
        
        return await response.json();
    },
    
    // Fazer predição individual
    async predict(modelId, modelType, features) {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                model_type: modelType,
                features: features
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao fazer predição');
        }
        
        return await response.json();
    },
    
    // Obter features de um modelo
    async getModelFeatures(modelId, modelType = null) {
        let url = `${API_BASE_URL}/models/${modelId}/features`;
        if (modelType) {
            url += `?model_type=${modelType}`;
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao obter features do modelo');
        }
        
        return await response.json();
    },
    
    // Análise de dados
    async analyze() {
        const response = await fetch(`${API_BASE_URL}/analyze`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erro ao analisar dados');
        }
        
        return await response.json();
    }
};

// Funções auxiliares
function showNotification(message, type = 'info') {
    // Cria uma notificação simples
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} alert-dismissible fade show`;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Remove após 5 segundos
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="spinner-border" role="status"><span class="visually-hidden">Carregando...</span></div>';
    }
}

function hideLoading(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content || '';
    }
}

