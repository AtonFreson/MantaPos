import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load calibration data and prepare it for modeling."""
    try:
        df = pd.read_csv(file_path)
        return df['depth'].values, df['sensor1'].values, df['sensor2'].values
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

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
        'KNN': KNeighborsRegressor(),
        'GaussianProcess': GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0)),
        'RANSAC': RANSACRegressor(random_state=42),
        'TheilSen': TheilSenRegressor(random_state=42),
        'Spline3': lambda x, y: UnivariateSpline(x, y, k=3)
    }
    return models

def evaluate_models(X_train, X_test, y_train, y_test, models):
    """Evaluate all models and return the best one."""
    results = {}
    best_score = -np.inf
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        try:
            if name == 'Spline3':
                # Special handling for spline
                model_fitted = model(X_train.ravel(), y_train)
                y_pred = model_fitted(X_test.ravel())
            else:
                model.fit(X_train.reshape(-1, 1), y_train)
                y_pred = model.predict(X_test.reshape(-1, 1))
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = {'r2': r2, 'rmse': rmse}
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"Error fitting {name}: {str(e)}")
            continue
    
    return results, best_model, best_model_name

def save_calibration(model, model_name, scaler=None, file_path='calibration.json'):
    """Save calibration parameters to JSON file."""
    calibration_data = {
        'model_name': model_name,
        'model_params': model.get_params() if hasattr(model, 'get_params') else None,
        'coefficients': model.coef_.tolist() if hasattr(model, 'coef_') else None,
        'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None
    }
    
    with open(file_path, 'w') as f:
        json.dump(calibration_data, f)

def main():
    # Configuration
    DATA_FILE = 'pressure_calibration_data.csv'
    CALIBRATION_OUTPUT = 'calibration.json'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

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
        
        # Evaluate models for sensor 1
        print("Evaluating models for Sensor 1...")
        results1, best_model1, best_model_name1 = evaluate_models(
            X1_train, X1_test, y1_train, y1_test, models)
        
        # Evaluate models for sensor 2
        print("Evaluating models for Sensor 2...")
        results2, best_model2, best_model_name2 = evaluate_models(
            X2_train, X2_test, y2_train, y2_test, models)
        
        # Save calibration parameters
        calibration = {
            'sensor1': {
                'model_name': best_model_name1,
                'performance': results1[best_model_name1]
            },
            'sensor2': {
                'model_name': best_model_name2,
                'performance': results2[best_model_name2]
            }
        }
        
        # Save models and calibration data
        save_calibration(best_model1, best_model_name1, file_path='sensor1_calibration.json')
        save_calibration(best_model2, best_model_name2, file_path='sensor2_calibration.json')
        
        print("\nCalibration Results:")
        print(f"Sensor 1 - Best Model: {best_model_name1}")
        print(f"R² Score: {results1[best_model_name1]['r2']:.4f}")
        print(f"RMSE: {results1[best_model_name1]['rmse']:.4f}")
        
        print(f"\nSensor 2 - Best Model: {best_model_name2}")
        print(f"R² Score: {results2[best_model_name2]['r2']:.4f}")
        print(f"RMSE: {results2[best_model_name2]['rmse']:.4f}")
        
    except Exception as e:
        print(f"Error in calibration process: {str(e)}")

if __name__ == "__main__":
    main()
