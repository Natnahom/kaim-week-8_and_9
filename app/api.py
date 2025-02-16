from flask import Flask, request, jsonify
from app.model import load_model, predict
from app.logging_config import setup_logging
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
setup_logging()

# Load the model
model = load_model('../model/cred_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint to make predictions on incoming data.
    
    Returns:
    - JSON response containing predictions.
    """
    data = request.json

    # Ensure data is provided
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert JSON to DataFrame
    input_data = pd.DataFrame(data)

    # Make predictions
    predictions = predict(model, input_data)

    return jsonify(predictions.tolist())

@app.route('/', methods=['GET'])
def home():
    """
    Home route for testing the API.
    
    Returns:
    - Simple message indicating the API is running.
    """
    return "<h1>Fraud Detection API is running!</h1>"