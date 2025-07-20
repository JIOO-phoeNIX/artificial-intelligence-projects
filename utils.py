import csv
import matplotlib.pyplot as plt
import numpy as np

def load_csv_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]
    
def prepare_data(data, feature_columns, target_column):
    """
    Prepare data for training by extracting features and target variable.
    
    Args:
        data (list): List of dictionaries containing the data.
        feature_columns (list): List of feature column names.
        target_column (str): The target column name.
        
    Returns:
        tuple: A tuple containing features and target variable as NumPy arrays.
    """
    
    features = np.array([[float(row[col]) for col in feature_columns] for row in data])
    target = np.array([float(row[target_column]) for row in data])
    
    return features, target

def plot_features_vs_target(X, y, feature_names, target_name):
    """
    Plot features against the target variable.
    
    Args:
        X (np.ndarray): Feature matrix
        Y (np.ndarray): Target array
        feature_names (list): List of feature names
        target_name (str): Target name for the plot labels
    """    
    
    for i, feature in enumerate(feature_names):
        x = X[:, i]
        y = y
        
        plt.figure(figsize=(3, 3))
        plt.scatter(x, y, color='blue', alpha=0.5)
        plt.title(f"{feature.capitalize()} vs {target_name.capitalize()}")
        plt.xlabel(feature.capitalize())
        plt.ylabel(target_name.capitalize())
        plt.grid(True)
        plt.show()
        
def plot_loss_curve(training_loss_history: list[float], validation_loss_history: list[float]):
    """
    Plot the loss curve over training epochs.
    
    Args:
        training_loss_history (list): List of training loss values per epoch
        validation_loss_history (list): List of validation loss values per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(training_loss_history, label='Training Loss', color='red')
    plt.plot(validation_loss_history, label='validation Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, x_label: str, y_label: str):
    """
    Plot predicted vs actual target values for a regression model.

    Args:
        y_true (np.ndarray): Actual target values (ground truth)
        y_pred (np.ndarray): Predicted target values by the model
        title (str): Title of the plot
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis        
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color='green', alpha=0.7, label='Predicted Points')
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        color='red',
        linestyle='--',
        label='Perfect Prediction Line'
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()