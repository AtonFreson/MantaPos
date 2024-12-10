import numpy as np
import json
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

class PressureSensorSystem:
    def __init__(self, surface_sensor_depth=0.02):
        self.surface_sensor_depth = surface_sensor_depth
        self.sensor1_model = None
        self.sensor2_model = None
        self.reference_surface_pressure = None
        
    def load_calibration(self, sensor1_file='sensor1_calibration.json', 
                        sensor2_file='sensor2_calibration.json'):
        """Load calibration parameters for both sensors."""
        try:
            # Load sensor 1 calibration
            with open(sensor1_file, 'r') as f:
                sensor1_cal = json.load(f)
            
            # Load sensor 2 calibration
            with open(sensor2_file, 'r') as f:
                sensor2_cal = json.load(f)
            
            # Initialize models based on calibration data
            self.sensor1_model = self._initialize_model(sensor1_cal)
            self.sensor2_model = self._initialize_model(sensor2_cal)
            
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            return False
    
    def _initialize_model(self, calibration_data):
        """Initialize a model with saved parameters."""
        model_name = calibration_data['model_name']
        model_params = calibration_data.get('model_params', {})
        
        # Create appropriate model based on name
        model_map = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'MLP': MLPRegressor(),
            'KNN': KNeighborsRegressor(),
            'GaussianProcess': GaussianProcessRegressor(),
            'RANSAC': RANSACRegressor(),
            'TheilSen': TheilSenRegressor(),
            'Spline3': UnivariateSpline
        }
        
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Set model parameters if available
        if model_params and hasattr(model, 'set_params'):
            model.set_params(**model_params)
        
        # Set coefficients and intercept if available
        if 'coefficients' in calibration_data and hasattr(model, 'coef_'):
            model.coef_ = np.array(calibration_data['coefficients'])
        if 'intercept' in calibration_data and hasattr(model, 'intercept_'):
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
            
            # Calculate depth using the calibrated model
            if isinstance(self.sensor2_model, UnivariateSpline):
                depth = self.sensor2_model(compensated_depth_reading)
            else:
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
        while True:
            # These would be actual sensor readings in practice
            current_depth_sensor_value = 2000  # Example value
            current_surface_value = 1050  # Example value
            
            depth = sensor_system.get_depth(
                current_depth_sensor_value, 
                current_surface_value
            )
            
            if depth is not None:
                print(f"Current depth: {depth:.2f} meters")
            
            # Add appropriate delay here
            
    except KeyboardInterrupt:
        print("\nStopping depth measurements")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()
