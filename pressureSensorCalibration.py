import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import RANSACRegressor
import xgboost as xgb
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = 'recordings/test recording - 12-13@19-16.json'
#DATA_FILE = os.path.join(os.path.dirname(__file__), DATA_FILE)
SENSOR1_CALIBRATION_OUTPUT = 'calibrations/pressure_calibrations/sensor1_calibration.json'
SENSOR2_CALIBRATION_OUTPUT = 'calibrations/pressure_calibrations/sensor2_calibration.json'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_prepare_data(file_path):
    """Load data from JSON, align by time, and prepare it for modeling."""
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in file '{file_path}': {e}")

    depth_data = []
    pressure_data = []
    
    for item in data:
        if item.get('mpu_unit') == 1 and 'encoder' in item and 'distance' in item['encoder']:
            depth_data.append({'timestamp': item['encoder']['timestamp'], 'distance': item['encoder']['distance']})
        
        if item.get('mpu_unit') == 0 and 'pressure' in item and 'adc_value0' in item['pressure'] and 'adc_value1' in item['pressure']:
            pressure_data.append({'timestamp': item['pressure']['timestamp'], 'sensor0': item['pressure']['adc_value0'], 'sensor1': item['pressure']['adc_value1']})
             
    if not depth_data or not pressure_data:
        raise ValueError("No valid sensor or depth data found in the JSON file")

    # Convert to Pandas DataFrame
    depth_df = pd.DataFrame(depth_data)
    pressure_df = pd.DataFrame(pressure_data)

    # Sort data by timestamp
    depth_df = depth_df.sort_values('timestamp').reset_index(drop=True)
    pressure_df = pressure_df.sort_values('timestamp').reset_index(drop=True)
    
    # Get values for interpolation
    depth_x = depth_df['timestamp'].values
    depth_y = depth_df['distance'].values
    pressure_x = pressure_df['timestamp'].values
    pressure_sensor_0 = pressure_df['sensor0'].values
    pressure_sensor_1 = pressure_df['sensor1'].values
    
    if len(np.unique(depth_x)) < 2:
        raise ValueError("Not enough unique timestamps in the depth data for interpolation.")

    # Identify valid timestamps within the depth range
    valid_mask = (pressure_x >= depth_x.min()) & (pressure_x <= depth_x.max())

    # Filter pressure data to only those within the depth range
    pressure_x = pressure_x[valid_mask]
    pressure_sensor_0 = pressure_sensor_0[valid_mask]
    pressure_sensor_1 = pressure_sensor_1[valid_mask]

    if len(pressure_x) == 0:
        raise ValueError("No pressure data falls within the depth timestamp range; nothing to interpolate.")
    
    # Interpolate depth
    if len(depth_x) > 1:
        interp_func = interp1d(depth_x, depth_y, bounds_error=True)
        try:
            interpolated_depths = interp_func(pressure_x)
        except ValueError as e:
            raise ValueError(f"Error interpolating depth data: {e}")
        
        return np.array(interpolated_depths), np.array(pressure_sensor_0), np.array(pressure_sensor_1)
    else:
        raise ValueError("Not enough unique timestamps in the encoder data for interpolation.")

def create_models():
    """Create dictionary of regression models to test."""
    models = {
        'Linear': LinearRegression(),
        'Poly2': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Poly3': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Poly4': make_pipeline(PolynomialFeatures(4), LinearRegression()),
        'SVR': SVR(kernel='rbf'),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'MLP': MLPRegressor(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=3),  # Adjusted n_neighbors
        'GaussianProcess': GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0)),
        'RANSAC': RANSACRegressor(random_state=42),
        'TheilSen': TheilSenRegressor(random_state=42),
        'Spline3': lambda x, y: UnivariateSpline(x, y, k=2)
    }
    return models

def evaluate_models(X_train, X_test, y_train, y_test, models):
    """Evaluate all models and return the best one."""
    results = {}
    fitted_models = {}
    best_score = -np.inf
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        try:
            if name == 'Spline3':
                # Ensure strictly increasing x-values with no duplicates
                X_train_unique, indices = np.unique(X_train, return_index=True)
                y_train_unique = y_train[indices]
                
                if len(X_train_unique) > 2:
                    model_fitted = model(X_train_unique, y_train_unique)
                    y_pred = model_fitted(X_test.ravel())
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results[name] = {'r2': r2, 'rmse': rmse}
                    fitted_models[name] = model_fitted
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model_fitted
                        best_model_name = name
                else:
                    raise ValueError("Not enough unique data points for Spline3.")
                
            else:
                if name == 'KNN':
                    n_samples = len(X_train)
                    if n_samples < 5:
                        model.set_params(n_neighbors=max(1, n_samples - 1))
                        
                model.fit(X_train.reshape(-1, 1), y_train)
                y_pred = model.predict(X_test.reshape(-1, 1))
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {'r2': r2, 'rmse': rmse}
                fitted_models[name] = model
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
                
        except Exception as e:
            print(f"Error fitting {name}: {str(e)}")
            continue
    
    return results, fitted_models, best_model, best_model_name

def save_calibration(model, model_name, sensor_type, file_path):
    """Save calibration parameters to JSON file."""
    if model_name == 'Spline3':
        # UnivariateSpline does not have .get_params(), .coef_ or .intercept_ in the same way as linear models.
        # We'll just store the knots and coefficients.
        calibration_data = {
            'sensor_type': sensor_type,
            'model_name': model_name,
            'knots': model.get_knots().tolist(),
            'coefficients': model.get_coeffs().tolist()
        }
    else:
        calibration_data = {
            'sensor_type': sensor_type,
            'model_name': model_name,
            'model_params': model.get_params() if hasattr(model, 'get_params') else None,
            'coefficients': model.coef_.tolist() if hasattr(model, 'coef_') else None,
            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None
        }
    
    with open(file_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)

def visualize_models(X_train, X_test, y_train, y_test, results, fitted_models):
    """
    Visualize the data points and the fitted lines for the top 3 models based on R² score.
    Plots the training and testing data points and prediction lines of the best and two runner-ups.
    """
    # Sort models by R² in descending order
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]['r2'], reverse=True)
    
    # Top 3 models
    top_three = sorted_models[:3]
    
    # Create a plot range
    X_all = np.concatenate([X_train, X_test])
    X_plot = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Train Data')
    plt.scatter(X_test, y_test, color='red', alpha=0.7, label='Test Data')
    
    # Plot the top three models
    for i, (model_name, metrics) in enumerate(top_three):
        model = fitted_models[model_name]
        if model_name == 'Spline3':
            # Spline model: use model(X_plot.ravel()) directly
            y_plot = model(X_plot.ravel())
        else:
            # Other models: use predict method
            y_plot = model.predict(X_plot)
        
        colors = ['green', 'orange', 'purple']  # For up to 3 lines
        plt.plot(X_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.3f})")
    
    plt.title("Data vs. Top 3 Model Predictions")
    plt.xlabel("Sensor Value")
    plt.ylabel("Depth")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_two_models(X1_train, X1_test, y1_train, y1_test, results1, fitted_models1,
                          X2_train, X2_test, y2_train, y2_test, results2, fitted_models2):
    """
    Visualize the data points and the fitted lines for the top 3 models of both sensors side-by-side.
    Left subplot for Sensor 1, Right subplot for Sensor 2.
    """
    # Sort models by R² in descending order for sensor 1
    sorted_models_1 = sorted(results1.items(), key=lambda kv: kv[1]['r2'], reverse=True)
    top_three_1 = sorted_models_1[:3]

    # Sort models by R² in descending order for sensor 2
    sorted_models_2 = sorted(results2.items(), key=lambda kv: kv[1]['r2'], reverse=True)
    top_three_2 = sorted_models_2[:3]

    # Create a plot range for sensor 1
    X1_all = np.concatenate([X1_train, X1_test])
    X1_plot = np.linspace(X1_all.min(), X1_all.max(), 200).reshape(-1, 1)

    # Create a plot range for sensor 2
    X2_all = np.concatenate([X2_train, X2_test])
    X2_plot = np.linspace(X2_all.min(), X2_all.max(), 200).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Sensor 1
    ax1 = axes[0]
    ax1.scatter(X1_train, y1_train, color='blue', alpha=0.7, label='Train Data')
    ax1.scatter(X1_test, y1_test, color='red', alpha=0.7, label='Test Data')

    colors = ['green', 'orange', 'purple']
    for i, (model_name, metrics) in enumerate(top_three_1):
        model = fitted_models1[model_name]
        if model_name == 'Spline3':
            y_plot = model(X1_plot.ravel())
        else:
            y_plot = model.predict(X1_plot)
        ax1.plot(X1_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.3f})")

    ax1.set_title("Sensor 0 (adc_value0) - Top 3 Models")
    ax1.set_xlabel("Sensor Value")
    ax1.set_ylabel("Depth")
    ax1.legend()
    ax1.grid(True)

    # Right subplot: Sensor 2
    ax2 = axes[1]
    ax2.scatter(X2_train, y2_train, color='blue', alpha=0.7, label='Train Data')
    ax2.scatter(X2_test, y2_test, color='red', alpha=0.7, label='Test Data')

    for i, (model_name, metrics) in enumerate(top_three_2):
        model = fitted_models2[model_name]
        if model_name == 'Spline3':
            y_plot = model(X2_plot.ravel())
        else:
            y_plot = model.predict(X2_plot)
        ax2.plot(X2_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.3f})")

    ax2.set_title("Sensor 1 (adc_value1) - Top 3 Models")
    ax2.set_xlabel("Sensor Value")
    ax2.set_ylabel("Depth")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load data
        depth, sensor1, sensor2 = load_and_prepare_data(DATA_FILE)
        
        # Create and evaluate models for both sensors
        models = create_models()
        
        # Split data for both sensors
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            sensor1, depth, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            sensor2, depth, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Evaluate models for sensor 0
        print("Evaluating models for Sensor 0 (adc_value0)...")
        results1, fitted_models1, best_model1, best_model_name1 = evaluate_models(
            X1_train, X1_test, y1_train, y1_test, models)
        
        # Evaluate models for sensor 1
        print("Evaluating models for Sensor 1 (adc_value1)...")
        results2, fitted_models2, best_model2, best_model_name2 = evaluate_models(
            X2_train, X2_test, y2_train, y2_test, models)
        
        # Save calibration parameters
        save_calibration(best_model1, best_model_name1, 'adc_value0', file_path=SENSOR1_CALIBRATION_OUTPUT)
        save_calibration(best_model2, best_model_name2, 'adc_value1', file_path=SENSOR2_CALIBRATION_OUTPUT)
        
        print("\nCalibration Results:")
        print(f"Sensor 0 (adc_value0) - Best Model: {best_model_name1}")
        print(f"R² Score: {results1[best_model_name1]['r2']:.4f}")
        print(f"RMSE: {results1[best_model_name1]['rmse']:.4f}")
        
        print(f"\nSensor 1 (adc_value1) - Best Model: {best_model_name2}")
        print(f"R² Score: {results2[best_model_name2]['r2']:.4f}")
        print(f"RMSE: {results2[best_model_name2]['rmse']:.4f}")
        
        # Visualize the top 3 models for both sensors side-by-side
        visualize_two_models(X1_train, X1_test, y1_train, y1_test, results1, fitted_models1,
                              X2_train, X2_test, y2_train, y2_test, results2, fitted_models2)
        
        '''
        # Visualize the top 3 models for sensor 1
        print("\nVisualizing top 3 models for Sensor 1 (adc_value0)...")
        visualize_models(X1_train, X1_test, y1_train, y1_test, results1, fitted_models1)
        
        # Visualize the top 3 models for sensor 2
        print("Visualizing top 3 models for Sensor 2 (adc_value1)...")
        visualize_models(X2_train, X2_test, y2_train, y2_test, results2, fitted_models2)
        '''
        
    except Exception as e:
        print(f"Error in calibration process: {str(e)}")

if __name__ == "__main__":
    main()
