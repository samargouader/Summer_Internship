import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Chemin absolu vers votre modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'accuracy 0.8871.h5')

# Charger le modèle
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

def predict_image(image_path):
    """Fonction de prédiction principale"""
    if model is None:
        raise ValueError("Modèle non chargé")
    
    # Prétraitement
    img = Image.open(image_path)
    img = preprocess_image(img)
    
    # Prédiction
    predictions = model.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Mapping des classes
    class_names = ['healthy', 'mild', 'moderate', 'proliferate', 'severe']
    return class_names[predicted_class], confidence

def preprocess_image(img, target_size=(224, 224)):
    """Prétraitement standard"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img