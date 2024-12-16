# ComputerVisionModelAPI

**ComputerVisionModelAPI** é uma API construída com **FastAPI** que oferece funcionalidades para classificação de imagens utilizando um modelo de aprendizado de máquina. O modelo é baseado no **K-Nearest Neighbors (KNN)** e treinado com o conjunto de dados **Fashion MNIST**, que contém imagens de roupas e acessórios. A API oferece duas funcionalidades principais: a predição de classe para imagens enviadas e a obtenção de um relatório de classificação do modelo.

## Funcionalidades

1. **Predição de Classe**: Envie uma imagem através de um **POST request** para receber a classe prevista para a imagem.
2. **Relatório de Classificação**: Obtenha um relatório detalhado sobre a performance do modelo em relação ao conjunto de dados de teste, incluindo métricas como precisão, recall, f1-score e acurácia.

## Requisitos

Antes de executar o projeto, certifique-se de que você tenha os seguintes pacotes instalados:

- Python 3.8 ou superior
- **FastAPI** para criação da API.
- **Uvicorn** como servidor ASGI para rodar a aplicação FastAPI.
- **scikit-learn** para a implementação do modelo de aprendizado de máquina.
- **numpy** para manipulação de arrays e dados numéricos.
- **joblib** para salvar e carregar o modelo treinado.
- **Pillow** para processamento de imagens.
- **tensorflow** para carregar o dataset Fashion MNIST.
- **scikit-image** para pré-processamento de imagens.

Para instalar as dependências, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

A estrutura de diretórios do projeto é a seguinte:

```bash
ComputerVisionModelAPI/
│
├── app/
│   ├── models/
│   │   └── knn_fashion_mnist_model.pkl       # Modelo treinado KNN
│   |   └── classification_report.json        # Relatório de classificação do modelo
│   ├── routes/
│   │   ├── predict.py                        # Rota de predição de classe
│   │   └── classification_report.py          # Rota para obter o relatório de classificação
│   ├── main.py                               # Arquivo principal que inicializa a API
│   └── test/
│   │   └── data/                             # Imagens para teste
│   |   └── predict_test.py                   # Script para um teste de requisicao

├── requirements.txt                          # Arquivo com dependências do projeto
└── README.md                                 # Este arquivo

```

## Como Rodar

1. Clone este repositório para o seu ambiente local:

```bash
git clone
cd ComputerVisionModelAPI
```

2. Instale as dependências necessárias:

```bash
python -m venv env
env/Scripts/actiavte (windoes mode)
pip install -r requirements.txt
```

3. Inicie o servidor da API com o Uvicorn:

```bash
uvicorn app.main:app --reload
```

Isso iniciará o servidor localmente em http://127.0.0.1:8000.

4. Acesse as rotas da API:

* Predição de classe: Envie uma imagem através de uma requisição POST para a rota /predict/.

* Relatório de classificação: Acesse o relatório do modelo através da rota /classification-report/.

## Como Usar

### Predição de Classe

Para realizar uma predição de classe, envie uma imagem para a rota /predict/:

```bash
POST http://127.0.0.1:8000/predict/
Content-Type: multipart/form-data
Body:
  file: [imagem_aqui]
```

### Relatório de Classificação

Para obter o relatório de classificação, faça uma requisição GET para a rota /classification-report/:
```bash
GET http://127.0.0.1:8000/classification-report/
```

## Como Treinar o Modelo

Para treinar o modelo KNN, execute o script **train_model.py**. Este script carregará o conjunto de dados Fashion MNIST, treinará o modelo KNN e salvará o modelo treinado em **app/models/knn_fashion_mnist_model.pkl**. O script também gera um arquivo **classification_report.json** com as métricas do modelo.