import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

def shap_expl(model, X_test, K=100):
    """
    Generate SHAP values and create a summary plot.
    
    Parameters:
    - model: Trained model for prediction.
    - X_test: Test dataset.
    - K: Number of samples to use for the background data summary.
    
    Returns:
    - shap_values: Calculated SHAP values for the test set.
    """
    # Summarize background data using K-means
    X_kmeans = shap.kmeans(X_test, K)
    explainer = shap.KernelExplainer(model.predict, X_kmeans)
    shap_values = explainer.shap_values(X_test)
    
    # Create a summary plot
    shap.summary_plot(shap_values, X_test)
    
    return shap_values

def shap_force_plot(model, X_test, instance_index, K=100):
    """
    Create a SHAP force plot for a specific instance.
    
    Parameters:
    - model: Trained model for prediction.
    - X_test: Test dataset.
    - instance_index: Index of the instance to explain.
    - K: Number of samples to use for the background data summary.
    """
    # Summarize background data using K-means
    X_kmeans = shap.kmeans(X_test, K)
    explainer = shap.KernelExplainer(model.predict, X_kmeans)
    shap_values = explainer.shap_values(X_test)
    
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[instance_index], X_test.iloc[instance_index])

def shap_dependence_plot(model, X_test, feature_name, K=100):
    """
    Create a SHAP dependence plot for a specific feature.
    
    Parameters:
    - model: Trained model for prediction.
    - X_test: Test dataset.
    - feature_name: The feature to plot.
    - K: Number of samples to use for the background data summary.
    """
    # Summarize background data using K-means
    X_kmeans = shap.kmeans(X_test, K)
    explainer = shap.KernelExplainer(model.predict, X_kmeans)
    shap_values = explainer.shap_values(X_test)
    
    shap.dependence_plot(feature_name, shap_values, X_test)

def lime_expl(model, X_train, X_test, instance_index):
    """
    Explain a prediction using LIME.
    
    Parameters:
    - model: Trained model for prediction.
    - X_train: Training dataset.
    - X_test: Test dataset.
    - instance_index: Index of the instance to explain.
    
    Returns:
    - exp: LIME explanation object.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=X_train.columns, 
        class_names=['Not Fraud', 'Fraud'], 
        mode='classification'
    )
    
    exp = explainer.explain_instance(X_test.values[instance_index], model.predict_proba)
    return exp

def plot_lime_explanation(exp):
    """
    Plot the LIME explanation for a specific instance.
    
    Parameters:
    - exp: LIME explanation object.
    """
    exp.as_pyplot_figure()
    plt.show()