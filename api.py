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

# --- NUEVA función de preprocesamiento con segmentación múltiple ---
def preprocess_and_segment_digits(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("No se pudo cargar la imagen.")

    avg_intensity = np.mean(img)
    if avg_intensity > background_threshold:
        img = cv2.bitwise_not(img)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No se detectaron dígitos.")

    # Ordenar contornos de izquierda a derecha
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    digit_images = []

    for (x, y, w, h) in bounding_boxes:
        if w < 5 or h < 5:
            continue  # ignorar cosas muy pequeñas

        digit_crop = img[y:y+h, x:x+w]

        # Redimensionar a 20x20 y centrar en 28x28
        digit_resized = cv2.resize(digit_crop, (20, 20))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized

        canvas = canvas / 255.0
        digit_images.append(canvas.reshape(28, 28, 1))

    return np.array(digit_images)  # (N, 28, 28, 1)

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
