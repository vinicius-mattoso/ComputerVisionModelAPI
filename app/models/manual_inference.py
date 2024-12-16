from joblib import load
import numpy as np
from skimage.io import imread
from skimage.transform import resize

model = load("./knn_fashion_mnist_model.pkl")
image = imread("C:/Users/vinicius/Documents/repositorios/ComputerVisionModelAPI/app/test/data/Trouser.png", as_gray=True)
image_resized = resize(image, (32, 32)).flatten()
image_normalized = image_resized / 255.0
prediction = model.predict([image_normalized])
print("Prediction:", prediction)
