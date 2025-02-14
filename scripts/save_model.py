import joblib

def save_model(model, model_path):
    """
    Train a model and save it as a .pkl file.
    
    Parameters:
    - model: models.
    - model_path: Path where the model will be saved.
    """
    # Save the trained model to a .pkl file
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")