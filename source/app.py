import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model import predict_image  # Import direct depuis model.py

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    stats = {
        "num_classes": 5,
        "healthy": 400,
        "mild": 200,
        "moderate": 400,
        "proliferate": 195,
        "severe": 150
    }
    return render_template('index.html', stats=stats)

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction, confidence = predict_image(filepath)
            return jsonify({
                'success': True,
                'filename': filename,
                'prediction': prediction,
                'confidence': float(confidence),
                'message': get_prediction_message(prediction)
            })
        except Exception as e:
            return jsonify({
                'error': f"Erreur lors de la prédiction: {str(e)}"
            }), 500
    else:
        return jsonify({'error': 'Type de fichier non autorisé'}), 400

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_prediction_message(prediction):
    messages = {
        'healthy': "Aucun signe de maladie détecté",
        'mild': "Signes légers de maladie détectés",
        'moderate': "Maladie à un stade modéré",
        'proliferate': "Maladie en phase de prolifération",
        'severe': "Maladie à un stade sévère"
    }
    return messages.get(prediction, "Résultat inconnu")

if __name__ == "__main__":
    app.run(debug=True)