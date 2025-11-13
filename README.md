# Projeto Final — Sistema de Análise de Dados e Predição em E-commerce

## Descrição do Projeto

Este projeto tem como objetivo desenvolver uma **aplicação web completa** para análise e predição de dados de e-commerce, permitindo que o usuário **faça upload de arquivos CSV**, visualize **análises interativas** e, futuramente, realize **predições com modelos de machine learning**.

A aplicação foi dividida em **frontend e backend**, de forma modular e escalável, garantindo flexibilidade e fácil manutenção.

---

## Funcionalidades Principais

### **1. Upload e Flexibilidade dos Dados**

- Upload de arquivos `.csv` com dados estruturados.
- Validação automática do formato e armazenamento local em `/uploads`.
- Suporte a datasets variados (ex: cidades diferentes, novas bases de e-commerce).

### **2. Análise de Dados e Visualização**

- Geração automática de análises estatísticas via **Pandas**.
- Criação de gráficos com **Matplotlib** e **Seaborn**:
  - Distribuições (histogramas)
  - Correlação entre variáveis
- Criação de **mapas interativos com Folium**, baseados na coluna de cidade (`city` ou `cidade`), permitindo a visualização geográfica dos dados.

### **3. Machine Learning com Treinamento Dinâmico**

- Endpoint de re-treinamento dinâmico (`/prediction-page`), simulando o treinamento do modelo.
- Registro de informações sobre o dataset e características do modelo treinado.
- Estrutura modular pronta para integração com algoritmos reais (Scikit-learn).

### **4. Interface Web Interativa**

- Frontend desenvolvido com **HTML, Bootstrap, FontAwesome e Chart.js**.
- Páginas já integradas:
  - `upload.html`: Upload de dados CSV.
  - `analysis.html`: Exibição de análises visuais.
  - `results.html`: Exibição dos resultados e performance do modelo.
- Navegação amigável e responsiva, ideal para apresentações e usuários não técnicos.

## ⚙️ Tecnologias Utilizadas

### 🧩 Backend

- **Python 3.10+**
- **Flask 3.1.2**
- **Flask-CORS**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Folium**
- **Geopy**

### 🎨 Frontend

- **HTML5 + CSS3**
- **Bootstrap 5**
- **Chart.js**
- **FontAwesome**
- **Templates Jinja2**

---

## 🧰 Instalação e Execução

### 1️⃣ Clonar o Repositório

```bash
git clone https://github.com/JorgeWillianPaez/TrabalhoFinalPython.git
cd TrabalhoFinalPython
```

### 2️⃣ Criar Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Executar o Servidor Flask

```bash
python app.py
```

O servidor será iniciado em:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 💡 Como Usar

### **1. Upload do Arquivo CSV**

- Acesse: [http://127.0.0.1:5000/upload](http://127.0.0.1:5000/upload)
- Selecione seu arquivo CSV (ex: `ecommerce_customer_behavior_dataset.csv`)
- Clique em **“Fazer Upload e Analisar”**

### **2. Visualização e Análise**

- Para análise dos dados, acesse `/analysis`.
- Serão exibidos:
  - Gráficos automáticos (histogramas e correlação)
  - Mapa interativo com localização das cidades (via Folium)

### **3. Treinamento Dinâmico**

- Acesse: [http://127.0.0.1:5000/prediction-page](http://127.0.0.1:5000/prediction-page)
- A aplicação executará um **treinamento**, gerando:
  - Número de amostras e colunas do dataset
  - Timestamp do treinamento
  - Relatório salvo em `models/trained_model_info.json`

---

## 🧠 Módulos do Backend Explicados

| Módulo                     | Função               | Descrição                                 |
| -------------------------- | -------------------- | ----------------------------------------- |
| `data_loader.py`           | Leitura de dados     | Carrega o CSV enviado e valida formato.   |
| `data_analysis.py`         | Estatísticas         | Gera resumo estatístico com Pandas.       |
| `visualization_service.py` | Gráficos e Mapas     | Cria gráficos (Seaborn) e mapas (Folium). |
| `model_training.py`        | Treinamento Dinâmico | Treinamento e gera arquivo JSON.          |

---

## 🧩 Estrutura das Rotas Flask

| Rota                   | Método   | Descrição                    |
| ---------------------- | -------- | ---------------------------- |
| `/`                    | GET      | Página inicial               |
| `/upload`              | GET/POST | Upload de arquivo CSV        |
| `/analysis`            | GET      | Exibe análises e gráficos    |
| `/prediction-page`     | GET      | Executa treinamento dinâmico |
| `/download/<filename>` | GET      | Baixa gráficos gerados       |

---

## 👨‍💻 Desenvolvedores

| Nome                   | Responsabilidade                       |
| ---------------------- | -------------------------------------- |
| **Milena Leonardi**    | Backend (Flask, Mapas)                 |
| **Jorge Willian Páez** | Frontend (Templates, Integração UI/UX) |
| **Isabela Class**      | Machine Learning                       |
| **Vinicius Prado**     | Análise de Dados                       |

---

## 📜 Licença

Este projeto é de uso acadêmico e educacional.  
Sinta-se livre para adaptar e expandir para outras aplicações analíticas.

---

## 📎 Conclusão

Este projeto representa uma aplicação web **completa e modular**, integrando:

- Backend analítico com **Python e Flask**
- Visualizações ricas com **Seaborn, Folium e Chart.js**
- Frontend moderno com **Bootstrap e templates Jinja2**
- Base sólida para **expansão com machine learning real**

O sistema entrega **insights visuais e reprodutíveis**, e está pronto para evoluir para predições reais com modelos configuráveis.
