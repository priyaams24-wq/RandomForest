from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load the model at startup
MODEL_PATH = 'random_forest_model.joblib'
INFO_PATH = 'model_info.json'

try:
    model = joblib.load(MODEL_PATH)
    with open(INFO_PATH, 'r') as f:
        model_info = json.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_info = None

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Random Forest Prediction API',
        'status': 'active',
        'endpoints': {
            '/': 'API information',
            '/predict': 'POST - Make predictions',
            '/model-info': 'GET - Model information',
            '/health': 'GET - Health check'
        },
        'model_loaded': model is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/model-info')
def info():
    """Get model information"""
    if model_info is None:
        return jsonify({'error': 'Model info not available'}), 500
    
    return jsonify({
        'feature_names': model_info['feature_names'],
        'target_names': model_info['target_names'],
        'model_type': 'RandomForestClassifier',
        'input_format': {
            'sepal_length': 'float (cm)',
            'sepal_width': 'float (cm)',
            'petal_length': 'float (cm)',
            'petal_width': 'float (cm)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" key in request',
                'expected_format': {
                    'features': [5.1, 3.5, 1.4, 0.2]
                }
            }), 400
        
        features = data['features']
        
        # Validate feature length
        if len(features) != 4:
            return jsonify({
                'error': 'Expected 4 features',
                'received': len(features),
                'feature_names': model_info['feature_names']
            }), 400
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        prediction_proba = model.predict_proba(features_array)
        
        # Get class name
        predicted_class = model_info['target_names'][prediction[0]]
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'predicted_class': predicted_class,
            'confidence': {
                model_info['target_names'][i]: float(prob)
                for i, prob in enumerate(prediction_proba[0])
            },
            'input_features': {
                model_info['feature_names'][i]: features[i]
                for i in range(len(features))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
