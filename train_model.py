import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from skimage.transform import resize
import joblib  # For saving and loading models
import json  # Para manipulação de arquivos JSON

# Função para mapear os labels numéricos para os nomes das classes
class_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def map_labels_to_class_names(labels, class_names):
    return np.array([class_names[label] for label in labels])

# 1. Carregar o dataset Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Dividir em 10% do dataset, de forma estratificada
x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.95, stratify=y_train, random_state=42)
x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.95, stratify=y_test, random_state=42)

# Mapear labels para nomes de classes
y_train = map_labels_to_class_names(y_train, class_names)
y_test = map_labels_to_class_names(y_test, class_names)

# 2. Pré-processamento
# Redimensionar imagens para 32x32 e achatar
image_size = (32, 32)

def preprocess_images(images, image_size):
    return np.array([resize(img, image_size).flatten() for img in images])

x_train_flatten = preprocess_images(x_train, image_size)
x_test_flatten = preprocess_images(x_test, image_size)

# Normalizar os valores de pixel para [0, 1]
x_train_flatten = x_train_flatten / 255.0
x_test_flatten = x_test_flatten / 255.0

# 3. Treinar o modelo KNN
def train_knn(x_train, y_train, x_test, y_test, output_path, report_json_path, k=5):
    # Criar o modelo
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # n_jobs=-1 para usar múltiplos núcleos
    knn.fit(x_train, y_train)

    # Avaliar o modelo
    y_test_pred = knn.predict(x_test)
    report = classification_report(y_test, y_test_pred, output_dict=True)  # Gerar o relatório de classificação em formato dicionário

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_test_pred) * 100

    # Adicionar a acurácia ao relatório
    report['accuracy'] = accuracy

    # Salvar o relatório em um arquivo JSON
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Relatório de classificação salvo em {report_json_path}")

    # Salvar o modelo
    joblib.dump(knn, output_path)
    print(f"Modelo KNN salvo em {output_path}")

# Caminho para salvar o modelo e o relatório
output_path = "app/models/knn_fashion_mnist_model.pkl"
report_json_path = "app/models/classification_report.json"

# Treinar e salvar o modelo e o relatório
train_knn(x_train_flatten, y_train, x_test_flatten, y_test, output_path, report_json_path)

# 4. Carregar o modelo salvo
def load_knn_model(model_path):
    return joblib.load(model_path)

# Carregar o modelo treinado
loaded_knn = load_knn_model(output_path)
print("Modelo KNN carregado com sucesso!")
