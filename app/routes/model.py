# app/routes/model.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from skimage.transform import resize
from PIL import Image
from io import BytesIO

router = APIRouter()

class_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Função de pré-processamento da imagem
def preprocess_image(image, image_size=(32, 32)):
    # Abrir a imagem usando PIL
    image = Image.open(BytesIO(image))  # Carregar imagem da memória
    image = image.convert('L')  # Converter para escala de cinza (Fashion MNIST é em escala de cinza)
    image = image.resize(image_size)  # Redimensionar para o tamanho desejado
    return np.array(image).flatten() / 255.0  # Achatar e normalizar a imagem

# Função para carregar o modelo KNN treinado
def load_knn_model(model_path):
    return joblib.load(model_path)

# Rota para previsão
@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler o arquivo de imagem enviado
        image_data = await file.read()

        # Pré-processar a imagem
        image_normalized = preprocess_image(image_data)

        # Carregar o modelo treinado
        model_path = "app/models/knn_fashion_mnist_model.pkl"
        model = load_knn_model(model_path)

        # Fazer a predição
        prediction = model.predict([image_normalized])  # O modelo KNN prevê a classe

        # Verificar se a predição é um índice ou um nome de classe
        predicted_class = prediction[0]  # Obtém a primeira (e única) predição

        # Se a predição for um índice (inteiro), converta para o nome da classe
        if isinstance(predicted_class, int):
            predicted_class = class_names[predicted_class]  # Converte para o nome da classe

        # Retornar a resposta com a previsão
        return JSONResponse(content={"prediction": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a imagem: {str(e)}")
