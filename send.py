import requests

# Ruta de la imagen que quieres predecir
image_path = "numero.png"  # Reemplázalo con la ruta correcta de tu imagen

# URL de la API Flask
url = "http://127.0.0.1:5000/predict"

# Abrir la imagen en modo binario y enviarla como archivo
with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

# Mostrar la respuesta
print("Respuesta de la API:", response.json())