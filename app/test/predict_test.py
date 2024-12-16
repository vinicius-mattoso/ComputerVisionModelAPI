import requests

url = "http://127.0.0.1:8000/predict/"

# Abra o arquivo de imagem que você quer testar
with open("./data/Trouser.png", "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

# Exibir a resposta
print(response.json())
