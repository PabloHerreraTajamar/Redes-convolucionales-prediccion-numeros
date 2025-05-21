# 🧠 Predicción de Dígitos con Redes Neuronales Convolucionales

Este proyecto implementa redes neuronales convolucional (CNN) para el reconocimiento de dígitos manuscritos utilizando el conjunto de datos MNIST. El objetivo es entrenar un modelo capaz de identificar correctamente los números del 0 al 9 a partir de imágenes en escala de grises de 28x28 píxeles.

## 📂 Estructura del Proyecto

- `Modelo_Convolutional_reconocimiento_imagenes.ipynb`: Notebook de Jupyter que contiene el desarrollo completo del modelo CNN, desde la carga de datos hasta la evaluación del rendimiento.
- `modelo_mnist.h5`: Archivo del modelo entrenado guardado en formato H5, listo para ser cargado y utilizado en aplicaciones de predicción.
- `api.py`: Script que implementa una API para realizar predicciones utilizando el modelo entrenado.
- `send.py`: Script auxiliar para enviar imágenes a la API y obtener predicciones.
- `numero.png`: Imagen de ejemplo utilizada para probar la predicción del modelo.
- `Image.jpg`: Otra imagen de ejemplo para pruebas adicionales.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## 🚀 Tecnologías Utilizadas

- **Lenguaje de programación:** Python
- **Bibliotecas principales:**
  - [TensorFlow](https://www.tensorflow.org/): Para la construcción y entrenamiento del modelo CNN.
  - [Keras](https://keras.io/): API de alto nivel para la creación de modelos de aprendizaje profundo.
  - [NumPy](https://numpy.org/): Para operaciones numéricas y manejo de matrices.
  - [Matplotlib](https://matplotlib.org/): Para la visualización de datos y resultados.
  - [Flask](https://flask.palletsprojects.com/): Para la creación de la API REST que permite realizar predicciones a través de solicitudes HTTP.
- **Entorno de desarrollo:** Jupyter Notebook

## 🛠️ Requisitos de Instalación

Para ejecutar el proyecto, asegúrate de tener instalado Python 3.x y las dependencias listadas en `requirements.txt`. Puedes instalarlas utilizando pip:

```bash
pip install -r requirements.txt
