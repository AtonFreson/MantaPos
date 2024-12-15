import numpy as np
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import RANSACRegressor
import xgboost as xgb
from scipy.interpolate import BSpline

class PressureSensorSystem:
    def __init__(self, surface_sensor_depth=0.02):
        self.surface_sensor_depth = surface_sensor_depth
        self.sensor1_model = None
        self.sensor2_model = None
        self.reference_surface_pressure = None
        
    def load_calibration(self, 
                         sensor1_file='calibrations/pressure_calibrations/pressure1_calibration.json', 
                         sensor2_file='calibrations/pressure_calibrations/pressure2_calibration.json'):
        """Load calibration parameters for both sensors."""
        try:
            # Load sensor 1 calibration
            with open(sensor1_file, 'r') as f:
                sensor1_cal = json.load(f)
            
            # Load sensor 2 calibration
            with open(sensor2_file, 'r') as f:
                sensor2_cal = json.load(f)
            
            # Initialize models based on calibration data
            self.sensor1_model = self.initialize_model(sensor1_cal)
            self.sensor2_model = self.initialize_model(sensor2_cal)
            
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            return False
    
    def initialize_model(self, calibration_data):
        """Initialize a model with saved parameters."""
        model_name = calibration_data['model_name']
        
        # Mapping model names to classes
        model_map = {
            'Linear': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'SVR': SVR,
            'DecisionTree': DecisionTreeRegressor,
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'XGBoost': xgb.XGBRegressor,
            'MLP': MLPRegressor,
            'KNN': KNeighborsRegressor,
            'GaussianProcess': GaussianProcessRegressor,
            'RANSAC': RANSACRegressor,
            'TheilSen': TheilSenRegressor,
            'Spline3': None  # Will handle separately
        }
        
        if model_name == 'Spline3':
            # Reconstruct the spline from knots and coefficients
            knots = np.array(calibration_data['knots'])
            coeffs = np.array(calibration_data['coefficients'])
            # The calibration used k=2 for the spline
            k = 2
            model = BSpline(knots, coeffs, k)
            return model
        else:
            # For other models
            model_class = model_map.get(model_name)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_name}")
            
            model_params = calibration_data.get('model_params', {})
            model = model_class(**model_params) if model_params else model_class()
            
            # If linear-like model, try to set coefficients and intercept
            if 'coefficients' in calibration_data and calibration_data['coefficients'] is not None and hasattr(model, 'coef_'):
                model.coef_ = np.array(calibration_data['coefficients'])
            if 'intercept' in calibration_data and calibration_data['intercept'] is not None and hasattr(model, 'intercept_'):
                model.intercept_ = calibration_data['intercept']
            
            return model
    
    def calibrate_surface_pressure(self, surface_sensor_value):
        """Calibrate system using current surface pressure reading."""
        try:
            self.reference_surface_pressure = surface_sensor_value
            return True
        except Exception as e:
            print(f"Error calibrating surface pressure: {str(e)}")
            return False
    
    def get_depth(self, depth_sensor_value, surface_sensor_value):
        """
        Calculate depth using both sensor values and compensating for atmospheric pressure.
        
        Args:
            depth_sensor_value: ADC reading from the depth sensor
            surface_sensor_value: Current ADC reading from the surface sensor
        
        Returns:
            float: Calculated depth in meters
        """
        try:
            if self.reference_surface_pressure is None:
                raise ValueError("Surface pressure not calibrated")
            
            # Calculate pressure difference due to atmospheric changes
            pressure_correction = surface_sensor_value - self.reference_surface_pressure
            
            # Compensate depth sensor reading for atmospheric pressure change
            compensated_depth_reading = depth_sensor_value - pressure_correction
            
            # Determine how to get depth depending on model type
            # If the model is callable (like a spline), just call it
            if callable(self.sensor2_model):
                depth = self.sensor2_model(compensated_depth_reading)
            else:
                # Otherwise, assume a sklearn-like model
                depth = self.sensor2_model.predict([[compensated_depth_reading]])[0]
            
            return max(0.0, depth)  # Ensure non-negative depth
            
        except Exception as e:
            print(f"Error calculating depth: {str(e)}")
            return None

def main():
    # Example usage
    try:
        # Initialize system
        sensor_system = PressureSensorSystem(surface_sensor_depth=0.02)
        
        # Load calibration
        if not sensor_system.load_calibration():
            print("Failed to load calibration")
            return
        
        # Calibrate surface pressure (should be done at startup)
        surface_sensor_value = 1000  # Example value
        sensor_system.calibrate_surface_pressure(surface_sensor_value)
        
        # Get depth reading (in a loop during actual usage)
        # For demonstration purposes, we'll just do a single calculation here
        current_depth_sensor_value = 2000  # Example value
        current_surface_value = 1050       # Example value
        
        depth = sensor_system.get_depth(
            current_depth_sensor_value, 
            current_surface_value
        )
        
        if depth is not None:
            print(f"Current depth: {depth:.2f} meters")
            
    except KeyboardInterrupt:
        print("\nStopping depth measurements")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()
