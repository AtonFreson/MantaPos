import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, TheilSenRegressor, RANSACRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import xgboost as xgb
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy import optimize

# Configuration
DATA_FILE = 'recordings/depth 3 4 calib - 02-14@15-40.json'
sensor1_name = 'adc_value0'
sensor2_name = 'adc_value1'

CALIBRATION_FOLDER = 'calibrations/pressure_calibrations/'
os.makedirs(CALIBRATION_FOLDER, exist_ok=True)
SENSOR1_CALIBRATION_OUTPUT = os.path.join(CALIBRATION_FOLDER, f"{sensor1_name}_calibration.pkl")
SENSOR2_CALIBRATION_OUTPUT = os.path.join(CALIBRATION_FOLDER, f"{sensor2_name}_calibration.pkl")
TEST_SIZE = 0.001
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
        
        if item.get('mpu_unit') == 0 and 'pressure' in item and sensor1_name in item['pressure'] and sensor2_name in item['pressure']:
            pressure_data.append({
                'timestamp': item['pressure']['timestamp'],
                'sensor1': item['pressure'][sensor1_name],
                'sensor2': item['pressure'][sensor2_name],
            })
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
    pressure_sensor_1 = pressure_df['sensor1'].values
    pressure_sensor_2 = pressure_df['sensor2'].values
    
    if len(np.unique(depth_x)) < 2:
        raise ValueError("Not enough unique timestamps in the depth data for interpolation.")

    # Identify valid timestamps within the depth range
    valid_mask = (pressure_x >= depth_x.min()) & (pressure_x <= depth_x.max())

    # Filter pressure data to only those within the depth range
    pressure_x = pressure_x[valid_mask]
    pressure_sensor_1 = pressure_sensor_1[valid_mask]
    pressure_sensor_2 = pressure_sensor_2[valid_mask]

    if len(pressure_x) == 0:
        raise ValueError("No pressure data falls within the depth timestamp range; nothing to interpolate.")
    
    # Interpolate depth
    if len(depth_x) > 1:
        interp_func = interp1d(depth_x, depth_y, bounds_error=True)
        try:
            interpolated_depths = interp_func(pressure_x)
        except ValueError as e:
            raise ValueError(f"Error interpolating depth data: {e}")
    else:
        raise ValueError("Not enough unique timestamps in the encoder data for interpolation.")
    
    # Helper function to filter sensor data
    def filter_sensor(depth_vals, sensor_vals):
        window_size = 0.01
        half_window = 0.005
        ranges = []
        counts = []
        d_min, d_max = depth_vals.min(), depth_vals.max()
        # First pass: compute sensor range and count per window.
        current = d_min
        while current < d_max:
            win_mask = (depth_vals >= current) & (depth_vals < current + window_size)
            count = np.sum(win_mask)
            if count > 0:
                win_data = sensor_vals[win_mask]
                ranges.append(win_data.max() - win_data.min())
                counts.append(count)
            current += window_size
        avg_range = np.mean(ranges) if ranges else 0
        avg_count = np.mean(counts) if counts else 0
        threshold = int(avg_count) if avg_count >= 1 else 1

        # Second pass: mark outliers deviating too much
        good_mask = np.ones(len(sensor_vals), dtype=bool)
        for i in range(len(sensor_vals)):
            local_mask = (depth_vals >= depth_vals[i] - half_window) & (depth_vals <= depth_vals[i] + half_window)
            if np.any(local_mask):
                local_median = np.median(sensor_vals[local_mask])
                if abs(sensor_vals[i] - local_median) > avg_range:
                    good_mask[i] = False
        removed_pass2_idx = np.where(~good_mask)[0]

        # Third pass: remove random excess points in each window.
        removed_pass3_idx = []
        current = d_min
        while current < d_max:
            win_mask = (depth_vals >= current) & (depth_vals < current + window_size)
            valid_indices = np.where(win_mask & good_mask)[0]
            if valid_indices.size > threshold:
                remove_count = valid_indices.size - threshold
                remove_indices = np.random.choice(valid_indices, size=remove_count, replace=False)
                removed_pass3_idx.extend(remove_indices.tolist())
                good_mask[remove_indices] = False
            current += window_size
        removed_pass3_idx = np.array(removed_pass3_idx)
        return good_mask, removed_pass2_idx, removed_pass3_idx

    # Filter sensor data for sensor 1
    sensor1_mask, r2_idx1, r3_idx1 = filter_sensor(interpolated_depths, pressure_sensor_1)
    print(f"Sensor 1: Removed {len(r2_idx1)} outliers and then {len(r3_idx1)} excess points.")
    filtered_depth1 = interpolated_depths[sensor1_mask]
    filtered_sensor1 = np.array(pressure_sensor_1)[sensor1_mask]
    # Compute removed datapoints for sensor 1 from original arrays
    removed_green1_depth = interpolated_depths[r2_idx1]
    removed_green1_sensor = np.array(pressure_sensor_1)[r2_idx1]
    removed_yellow1_depth = interpolated_depths[r3_idx1]
    removed_yellow1_sensor = np.array(pressure_sensor_1)[r3_idx1]

    # Filter sensor data for sensor 2
    sensor2_mask, r2_idx2, r3_idx2 = filter_sensor(interpolated_depths, pressure_sensor_2)
    print(f"Sensor 2: Removed {len(r2_idx2)} outliers and then {len(r3_idx2)} excess points.")
    filtered_depth2 = interpolated_depths[sensor2_mask]
    filtered_sensor2 = np.array(pressure_sensor_2)[sensor2_mask]
    removed_green2_depth = interpolated_depths[r2_idx2]
    removed_green2_sensor = np.array(pressure_sensor_2)[r2_idx2]
    removed_yellow2_depth = interpolated_depths[r3_idx2]
    removed_yellow2_sensor = np.array(pressure_sensor_2)[r3_idx2]

    # Return filtered data along with removed datapoints for each sensor.
    return (
        filtered_depth1, filtered_sensor1, 
        removed_green1_depth, removed_green1_sensor, 
        removed_yellow1_depth, removed_yellow1_sensor,
        filtered_depth2, filtered_sensor2,
        removed_green2_depth, removed_green2_sensor,
        removed_yellow2_depth, removed_yellow2_sensor
    )

# Piecewise linear regression model
class PiecewiseLinearAlternate:
    def __init__(self, max_segments=20, allowEdits=False):
        self.max_segments = max_segments
        self.px = None
        self.py = None
        self.allowEdits = allowEdits
 
    def segments_fit(self, X, Y, count):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()    

        seg = np.full(count - 1, (xmax - xmin) / count)

        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

        def func(p):
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2)**2)

        r = optimize.minimize(
            err, x0=np.r_[seg, py_init], method='SLSQP', 
            bounds=[(0, None)] * len(seg) + [(ymin, ymax)] * len(py_init), 
            constraints=[
                optimize.LinearConstraint([1] * len(seg) + [0] * len(py_init), 0, xmax - xmin)
            ])
        
        return func(r.x)

    def fit(self, X, y):
        # Flatten input arrays to ensure 1D data
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()
        self.px, self.py = self.segments_fit(X, y, self.max_segments)

        if self.allowEdits:
            # Save self.px and self.py to a temporary text file in JSON format
            temp_file = "piecewise_params.json"
            params = {
                "px": self.px.tolist() if hasattr(self.px, "tolist") else self.px,
                "py": self.py.tolist() if hasattr(self.py, "tolist") else self.py
            }
            with open(temp_file, "w") as f:
                json.dump(params, f, indent=4)
            print(f"Parameters saved to {temp_file}. You may edit the file if you wish.")
            input("Press Enter to continue and load parameters...")

            # Load the parameters back from the file
            with open(temp_file, "r") as f:
                params = json.load(f)
            self.px = np.array(params.get("px", self.px))
            self.py = np.array(params.get("py", self.py))
            print("Parameters loaded. Continuing with model fitting...")

    def predict(self, X):
        return np.interp(X, self.px, self.py)

# Piecewise linear regression model
class PiecewiseLinear:
    def __init__(self, min_width=25, max_width=100):
        self.min_width = min_width
        self.max_width = max_width
        self.segments = []      # Each segment: (start, end, slope, intercept)
        self.boundaries = []    # List of segment boundaries

    def fit(self, X, y):
        X = np.array(X).ravel()
        y = np.array(y).ravel()
        idx = np.argsort(X)
        X_sorted = X[idx]
        y_sorted = y[idx]
        self.segments = []
        self.boundaries = [X_sorted[0]]
        while self.boundaries[-1] < X_sorted[-1]:
            seg_start = self.boundaries[-1]
            seg_end_limit = seg_start + self.max_width
            seg_mask = (X_sorted >= seg_start) & (X_sorted <= seg_end_limit)
            seg_indices = np.where(seg_mask)[0]
            if len(seg_indices) == 0:
                break
            if X_sorted[seg_indices[-1]] - seg_start < self.min_width:
                seg_end = min(seg_start + self.min_width, X_sorted[-1])
                seg_points_mask = (X_sorted >= seg_start) & (X_sorted <= seg_end)
                seg_indices = np.where(seg_points_mask)[0]
            else:
                seg_end = X_sorted[seg_indices[-1]]
            X_seg = X_sorted[seg_indices]
            y_seg = y_sorted[seg_indices]
            if self.boundaries[0] == seg_start:
                m, c = np.polyfit(X_seg, y_seg, 1)
            else:
                yb = self._predict_at(seg_start)
                denom = np.sum((X_seg - seg_start)**2)
                m = np.sum((X_seg - seg_start) * (y_seg - yb)) / denom if denom != 0 else 0
                c = yb - m * seg_start
            self.segments.append((seg_start, seg_end, m, c))
            self.boundaries.append(seg_end)
            if seg_end >= X_sorted[-1]:
                break
                
    def _predict_at(self, x):
        for seg in self.segments:
            start, end, m, c = seg
            if start <= x <= end:
                return m * x + c
        if self.segments:
            m, c = self.segments[-1][2], self.segments[-1][3]
            return m * x + c
        return None

    def predict(self, X):
        X = np.array(X).ravel()
        return np.array([self._predict_at(x) for x in X])

def create_models():
    models = {
        'Linear': LinearRegression(),
        #'Poly2': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        #'Poly3': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        #'Poly4': make_pipeline(PolynomialFeatures(4), LinearRegression()),
        #'Poly20': make_pipeline(PolynomialFeatures(20), LinearRegression()),
        #'SVR': SVR(kernel='rbf', C=5, gamma=0.005, epsilon=0.1),
        #'DecisionTree': DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5, min_samples_leaf=5),
        #'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=70, max_depth=6, min_samples_leaf=15, max_features=0.8), 
        #'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=150, max_depth=3, learning_rate=0.05), 
        #'RANSAC': RANSACRegressor(random_state=RANDOM_STATE, estimator=DecisionTreeRegressor(max_depth=5), min_samples=0.5),
        #'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_estimators=100, max_depth=3, reg_alpha=0.1, reg_lambda=1, verbosity=0),
        #'MLP': MLPRegressor(random_state=RANDOM_STATE, hidden_layer_sizes=(50,), max_iter=1000),
        #'Ridge': Ridge(random_state=RANDOM_STATE),
        #'Lasso': Lasso(random_state=RANDOM_STATE),
        #'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
        #'KNN': KNeighborsRegressor(n_neighbors=10, weights='uniform'),
        #'GaussianProcess': GaussianProcessRegressor(kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds=(0.1, 10.0)), alpha=0.1),
        #'TheilSen': TheilSenRegressor(random_state=RANDOM_STATE),
        #'Spline3': lambda x, y: UnivariateSpline(x, y, k=2),
        #'KerasNN': create_nn_model(),
        'PiecewiseLinear': PiecewiseLinear(20, 200),
        'PiecewiseLinearAlternate': PiecewiseLinearAlternate(max_segments=100)#200
    }
    return models

def create_nn_model():
    # Create the ML based neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='linear', input_shape=[1]),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

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
                    #print(f"Model: {name}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model_fitted
                        best_model_name = name
                else:
                    raise ValueError("Not enough unique data points for Spline3.")
                
            elif name == 'KerasNN':
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                X_train_norm = scaler_x.fit_transform(X_train.reshape(-1, 1))
                X_test_norm = scaler_x.transform(X_test.reshape(-1, 1))
                y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1))
                
                # Train the Keras neural network on normalized data
                model.fit(X_train_norm, y_train_norm, epochs=100, verbose=1)
                
                # Store scalers in the model for later un-normalization
                model.scaler_x = scaler_x
                model.scaler_y = scaler_y
                
                y_pred_norm = model.predict(X_test_norm).ravel()
                y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {'r2': r2, 'rmse': rmse}
                fitted_models[name] = model
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
                
            else:
                model.fit(X_train.reshape(-1, 1), y_train)
                y_pred = model.predict(X_test.reshape(-1, 1))
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {'r2': r2, 'rmse': rmse}
                fitted_models[name] = model
                #print(f"Model: {name}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
                
        except Exception as e:
            print(f"Error fitting {name}: {str(e)}")
            continue
    
    return results, fitted_models, best_model, best_model_name


def save_calibration(model, model_name, sensor_type, file_path):
    """Save calibration model using pickle."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump({
                'sensor_type': sensor_type,
                'model_name': model_name,
                'model': model
            }, f)
        print(f"Saved {model_name} model for {sensor_type} to {file_path}")
    except Exception as e:
        print(f"Error saving calibration model: {str(e)}")


# Modify visualize_models to accept removed datapoints (green and yellow)
def visualize_models(X_train, X_test, y_train, y_test, results, fitted_models, removed_green, removed_yellow):
    """
    Visualize the data points and the fitted lines for the top 3 models based on R² score.
    Plots the training and testing data points and prediction lines of the best and two runner-ups.
    """
    # removed_green and removed_yellow are tuples: (green_depth, green_sensor) and (yellow_depth, yellow_sensor)
    # Sort models by R² in descending order
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]['r2'], reverse=True)
    
    # Top 3 models
    top_three = sorted_models[:3]
    
    # Create a plot range
    X_all = np.concatenate([X_train, X_test])
    X_plot = np.linspace(X_all.min(), X_all.max(), 1000).reshape(-1, 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot removed points: green unfilled circles (high deviation) and yellow unfilled circles (excess)
    green_depth, green_sensor = removed_green
    yellow_depth, yellow_sensor = removed_yellow
    plt.scatter(green_sensor, green_depth, facecolors='none', edgecolors='green', s=50, 
                linewidths=1.5, label='High Deviation Removed')
    plt.scatter(yellow_sensor, yellow_depth, facecolors='none', edgecolors='yellow', s=50, 
                linewidths=1.5, label='Excess Points Removed')
    
    # Plot data points
    plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Train Data')
    plt.scatter(X_test, y_test, color='red', alpha=0.7, label='Test Data')
    
    # Plot the top three models
    for i, (model_name, metrics) in enumerate(top_three):
        model = fitted_models[model_name]
        if model_name == 'Spline3':
            # Spline model: use model(X_plot.ravel()) directly
            y_plot = model(X_plot.ravel())
        elif model_name == 'KerasNN' and hasattr(model, 'scaler_x') and hasattr(model, 'scaler_y'):
            X_plot_norm = model.scaler_x.transform(X_plot)
            y_plot_norm = model.predict(X_plot_norm)
            y_plot = model.scaler_y.inverse_transform(y_plot_norm)
        else:
            # Other models: use predict method
            y_plot = model.predict(X_plot)
        
        colors = ['green', 'orange', 'purple']  # For up to 3 lines
        plt.plot(X_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.6f})")
    
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
        elif model_name == 'KerasNN' and hasattr(model, 'scaler_x') and hasattr(model, 'scaler_y'):
            X1_plot_norm = model.scaler_x.transform(X1_plot)
            y_plot_norm = model.predict(X1_plot_norm)
            y_plot = model.scaler_y.inverse_transform(y_plot_norm)
        else:
            y_plot = model.predict(X1_plot)
        ax1.plot(X1_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.6f})")

    ax1.set_title("Sensor 1 (adc_value0) - Top 3 Models")
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
        elif model_name == 'KerasNN' and hasattr(model, 'scaler_x') and hasattr(model, 'scaler_y'):
            X2_plot_norm = model.scaler_x.transform(X2_plot)
            y_plot_norm = model.predict(X2_plot_norm)
            y_plot = model.scaler_y.inverse_transform(y_plot_norm)
        else:
            y_plot = model.predict(X2_plot)
        ax2.plot(X2_plot, y_plot, color=colors[i], linewidth=2, label=f"{model_name} (R²={metrics['r2']:.6f})")

    ax2.set_title("Sensor 2 (adc_value1) - Top 3 Models")
    ax2.set_xlabel("Sensor Value")
    ax2.set_ylabel("Depth")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        # Unpack filtered data and removed points for both sensors.
        (depth1, sensor1, 
         green_depth1, green_sensor1, yellow_depth1, yellow_sensor1,
         depth2, sensor2,
         green_depth2, green_sensor2, yellow_depth2, yellow_sensor2) = load_and_prepare_data(DATA_FILE)
        
        # Create and evaluate models for both sensors
        models_1 = create_models()
        models_2 = create_models()
        
        # Split data for both sensors
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            sensor1, depth1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            sensor2, depth2, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Evaluate models for sensor 1
        print("\nEvaluating models for Sensor 1 (adc_value0)...")
        #results1, fitted_models1, best_model1, best_model_name1 = evaluate_models(
        #    X1_train, X1_test, y1_train, y1_test, models_1)
        results1, fitted_models1, best_model1, best_model_name1 = evaluate_models(
            X1_train,X1_train, y1_train,y1_train, models_1)
        
        # Evaluate models for sensor 2
        print("Evaluating models for Sensor 2 (adc_value1)...\n")
        #results2, fitted_models2, best_model2, best_model_name2 = evaluate_models(
        #    X2_train, X2_test, y2_train, y2_test, models_2)
        results2, fitted_models2, best_model2, best_model_name2 = evaluate_models(
            X2_train,X2_train, y2_train,y2_train, models_2)
        
        # Save calibration models using pickle
        save_calibration(best_model1, best_model_name1, sensor1_name, file_path=SENSOR1_CALIBRATION_OUTPUT)
        save_calibration(best_model2, best_model_name2, sensor2_name, file_path=SENSOR2_CALIBRATION_OUTPUT)
        
        print("\nCalibration Results:")
        print(f"Sensor 1 (adc_value0) - Best Model: {best_model_name1}")
        print(f"R² Score: {results1[best_model_name1]['r2']:.6f}")
        print(f"RMSE: {results1[best_model_name1]['rmse']:.4f}")
        
        print(f"\nSensor 2 (adc_value1) - Best Model: {best_model_name2}")
        print(f"R² Score: {results2[best_model_name2]['r2']:.6f}")
        print(f"RMSE: {results2[best_model_name2]['rmse']:.4f}")
        
        
        # Visualize the top 3 models for both sensors side-by-side
        #visualize_two_models(X1_train, X1_test, y1_train, y1_test, results1, fitted_models1,
        #                      X2_train, X2_test, y2_train, y2_test, results2, fitted_models2)
        
        
        # Visualize the top 3 models for sensor 1
        print("\nVisualizing top 3 models for Sensor 1 (adc_value0)...")
        visualize_models(
            X1_train, X1_test, y1_train, y1_test, results1, fitted_models1,
            removed_green=(green_depth1, green_sensor1),
            removed_yellow=(yellow_depth1, yellow_sensor1)
        )
        
        # Visualize the top 3 models for sensor 2
        print("Visualizing top 3 models for Sensor 2 (adc_value1)...")
        visualize_models(
            X2_train, X2_test, y2_train, y2_test, results2, fitted_models2,
            removed_green=(green_depth2, green_sensor2),
            removed_yellow=(yellow_depth2, yellow_sensor2)
        )
        
        
    except Exception as e:
        print(f"Error in calibration process: {str(e)}")