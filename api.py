from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("modelo_mnist.h5")

# Crear la aplicación Flask
app = Flask(__name__)

# Umbral para decidir si invertir la imagen
background_threshold = 127

# Función de preprocesamiento con segmentación múltiple
def preprocess_and_segment_digits(image_bytes):
    # Convertir los bytes de la imagen a un arreglo de numpy y decodificarla en escala de grises
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("No se pudo cargar la imagen.")  # Verificar si la imagen se cargó correctamente

    # Invertir la imagen si el fondo es claro y los dígitos oscuros
    avg_intensity = np.mean(img)
    if avg_intensity > background_threshold:
        img = cv2.bitwise_not(img)

    # Umbralizar la imagen para convertirla en una imagen binaria
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar los contornos (dígitos) en la imagen
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se detectaron dígitos.")  # Asegurarse de que haya contornos

    # Ordenar los contornos de izquierda a derecha
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    digit_images = []

    # Recortar, redimensionar y centrar cada dígito en una imagen 28x28
    for (x, y, w, h) in bounding_boxes:
        if w < 5 or h < 5:
            continue  # Ignorar contornos demasiado pequeños

        # Recortar el dígito
        digit_crop = img[y:y+h, x:x+w]

        # Redimensionar a 20x20 y centrarlo en una imagen 28x28
        digit_resized = cv2.resize(digit_crop, (20, 20))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized

        canvas = canvas / 255.0  # Normalizar la imagen
        digit_images.append(canvas.reshape(28, 28, 1))

    return np.array(digit_images)  # Retorna un arreglo de imágenes 28x28

# Ruta para predecir varios dígitos en una imagen
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()

        digits_array = preprocess_and_segment_digits(image_bytes)

        predictions = model.predict(digits_array)
        predicted_labels = np.argmax(predictions, axis=1).tolist()

        return jsonify({"Prediccion": predicted_labels}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
