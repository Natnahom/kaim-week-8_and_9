import joblib

def load_model(model_path):
    """
    Load the trained model from the specified path.
    
    Parameters:
    - model_path: Path to the saved model file.
    
    Returns:
    - model: Loaded model object.
    """
    model = joblib.load(model_path)
    return model

def predict(model, data):
    """
    Make predictions using the loaded model.
    
    Parameters:
    - model: Loaded model object.
    - data: Data for which predictions are to be made.
    
    Returns:
    - predictions: Predicted values.
    """
    predictions = model.predict(data)
    return predictions