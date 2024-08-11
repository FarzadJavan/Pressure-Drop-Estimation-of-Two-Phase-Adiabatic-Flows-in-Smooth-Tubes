import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from model.tpot_exported_pipeline import exported_pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data():
    """Load training and testing data."""
    cwd = sys.path[0]

    X_train_path = os.path.join(cwd, "data", "X_train.csv")
    X_test_path = os.path.join(cwd, "data", "X_test.csv")
    y_train_path = os.path.join(cwd, "data", "y_train.csv")
    y_test_path = os.path.join(cwd, "data", "y_test.csv")

    X_train = pd.read_csv(X_train_path, index_col=0)
    X_test = pd.read_csv(X_test_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0)
    y_test = pd.read_csv(y_test_path, index_col=0)

    return X_train, X_test, y_train, y_test

def deviation_calculator(y_true, y_pred):
    """Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values. """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ensure no division by zero
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, which would lead to division by zero.")
    
    # Compute the deviation
    deviation = (y_pred - y_true) / y_true
    mean_deviation = np.mean(deviation)
    mean_absolute_deviation = np.mean(np.abs(deviation))
    
    # Convert to percentage
    mean_deviation_percentage = 100 * mean_deviation
    mean_absolute_deviation_percentage = 100 * mean_absolute_deviation

    return round( float(mean_deviation_percentage),2), round(float(mean_absolute_deviation_percentage),2)

def evaluate_model(exported_pipeline, X_train_selected, y_train):
    """Evaluate the model using cross-validation and calculate deviations."""
    cv = 10
    y_train_pred_cv = cross_val_predict(exported_pipeline, X_train_selected, y_train.values.ravel(), cv=cv)
    
    MAPE_train_cv, MPE_train_cv = deviation_calculator(y_train, pd.DataFrame(y_train_pred_cv))
    print(f'{" Cross-Validation Performance on Training Set ":#^100}')
    print(f"  - Mean Absolute Percentage Error (MAPE): {MAPE_train_cv:.2f}%\n"
          f"  - Mean Percentage Error (MPE): {MPE_train_cv:.2f}%\n")

    return exported_pipeline.fit(X_train_selected, y_train.values.ravel())

def test_model(model, X_test, y_test):
    """Make predictions with the trained model and evaluate performance."""
    y_test_pred = model.predict(X_test[model.feature_names_in_])
    MAPE_test, MPE_test = deviation_calculator(y_test.values.ravel(), y_test_pred)
    print(f'{" Model Performance on Test Set ":#^100}')
    print(f"  - Mean Absolute Percentage Error (MAPE): {MAPE_test:.2f}%\n"
          f"  - Mean Percentage Error (MPE): {MPE_test:.2f}%\n")

def save_model(model, filepath='model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)

def main():
    """Main function to execute the script."""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Define selected features
    selected_features = ["x [-]", "Re_g [-]", "X [-]"]
    X_train_selected = X_train[selected_features]
    
    # Evaluate and train model
    trained_model = evaluate_model(exported_pipeline, X_train_selected, y_train)
    
    # Predict and evaluate on test set
    test_model(trained_model, X_test, y_test)
    
    # Save the trained model
    filepath=os.path.join(sys.path[0], "model", "model.pkl")
    save_model(trained_model, filepath=filepath)

if __name__ == "__main__":
    main()
