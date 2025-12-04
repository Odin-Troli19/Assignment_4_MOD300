"""
ebola_ml_functions.py

Assignment 4 - TOPIC 2: Supervised learning: Machines versus human models
some of the functions were giving an error when imported so we had to rewrite them in the Notebook again


This module contains all functions for:
- Loading and preprocessing Ebola data
- SEIR model implementation (from Project 2)
- Linear regression
- Polynomial/better fitting functions
- Neural Network models
- LSTM for time series prediction
- Evaluation and visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ebola_data(filepath):
    """
    Load Ebola data from a .dat file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    
    Returns:
    --------
    days : np.array
        Days since first outbreak
    new_cases : np.array
        Number of new cases reported each period
    cumulative_cases : np.array
        Cumulative number of cases
    """
    data = np.genfromtxt(filepath, skip_header=1, usecols=(1, 2))
    days = data[:, 0]
    new_cases = data[:, 1]
    new_cases = np.nan_to_num(new_cases, nan=0.0)
    new_cases = np.maximum(new_cases, 0)
    cumulative_cases = np.cumsum(new_cases)
    return days, new_cases, cumulative_cases


def load_all_countries(base_path):
    """
    Load Ebola data for all three countries.
    
    Parameters:
    -----------
    base_path : str
        Base path to the data files
    
    Returns:
    --------
    dict : Dictionary with country data
    """
    countries = {
        'Guinea': 'ebola_cases_guinea.dat',
        'Liberia': 'ebola_cases_liberia.dat',
        'Sierra Leone': 'ebola_cases_sierra_leone.dat'
    }
    
    data = {}
    for country, filename in countries.items():
        filepath = f"{base_path}/{filename}"
        days, new_cases, cumulative = load_ebola_data(filepath)
        data[country] = {
            'days': days,
            'new_cases': new_cases,
            'cumulative': cumulative
        }
    return data


# ============================================================================
# SEIR MODEL (FROM PROJECT 2 - Exercise 5)
# ============================================================================

def seir_model(t, y, beta0, lambda_param, sigma, gamma, N):
    """
    SEIR model right-hand side for Ebola.
    
    dS/dt = -β(t) * S * I / N
    dE/dt = β(t) * S * I / N - σ * E
    dI/dt = σ * E - γ * I
    dR/dt = γ * I
    
    where β(t) = β₀ * exp(-λ * t)
    """
    S, E, I, R = y
    beta_t = beta0 * np.exp(-lambda_param * t)
    
    dS = -beta_t * S * I / N
    dE = beta_t * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    
    return [dS, dE, dI, dR]


def solve_seir_model(t_span, t_eval, beta0, lambda_param, sigma, gamma, N, I0=1, E0=1):
    """
    Solve the SEIR model using scipy.integrate.solve_ivp.
    
    Parameters:
    -----------
    t_span : tuple
        (t_start, t_end)
    t_eval : array
        Times at which to evaluate the solution
    beta0 : float
        Initial transmission rate
    lambda_param : float
        Decay rate of transmission
    sigma : float
        Rate from exposed to infectious (1/incubation_period)
    gamma : float
        Recovery/death rate (1/infectious_period)
    N : float
        Total population
    I0 : int
        Initial number of infected
    E0 : int
        Initial number of exposed
    
    Returns:
    --------
    t : array
        Time points
    y : array
        Solution array [S, E, I, R]
    """
    S0 = N - E0 - I0
    R0 = 0
    y0 = [S0, E0, I0, R0]
    
    sol = solve_ivp(
        seir_model, 
        t_span, 
        y0, 
        args=(beta0, lambda_param, sigma, gamma, N),
        t_eval=t_eval,
        method='RK45'
    )
    
    return sol.t, sol.y.T


def fit_seir_to_data(days, cumulative_cases, sigma, gamma, N, 
                     beta0_init=0.5, lambda_init=0.001):
    """
    Fit SEIR model to cumulative case data using curve_fit.
    
    Returns fitted parameters (beta0, lambda) and model predictions.
    """
    def model_cumulative(t, beta0, lambda_param):
        t_span = (0, max(t) + 10)
        _, y = solve_seir_model(t_span, t, beta0, lambda_param, sigma, gamma, N)
        # R compartment represents cumulative cases (recovered + dead)
        return y[:, 3]
    
    try:
        popt, pcov = curve_fit(
            model_cumulative, 
            days, 
            cumulative_cases,
            p0=[beta0_init, lambda_init],
            bounds=([0.01, 0.0001], [2.0, 0.1]),
            maxfev=5000
        )
        beta0_fit, lambda_fit = popt
        cumulative_pred = model_cumulative(days, beta0_fit, lambda_fit)
        return beta0_fit, lambda_fit, cumulative_pred
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None


# ============================================================================
# TASK 0: Reproduce Exercise 5 plots
# ============================================================================

def plot_exercise5_style(days, new_cases, cumulative_cases, country_name, 
                         model_cumulative=None, ax=None):
    """
    Reproduce the Exercise 5 style plot with dual y-axis.
    Shows new cases (circles) and cumulative cases (line/squares).
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    else:
        ax1 = ax
        fig = ax.figure
    
    # Plot new cases on left axis
    color1 = 'red'
    ax1.plot(days, new_cases, 'o', color=color1, alpha=0.6, 
             markersize=5, label='New cases')
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for cumulative
    ax2 = ax1.twinx()
    color2 = 'black'
    ax2.plot(days, cumulative_cases, 's', color=color2, alpha=0.6,
             markersize=4, label='Cumulative (data)')
    
    if model_cumulative is not None:
        ax2.plot(days, model_cumulative, '-', color='blue', linewidth=2,
                 label='Model prediction')
    
    ax2.set_ylabel('Cumulative number of outbreaks', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(f'Ebola outbreaks in {country_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return fig, ax1, ax2


# ============================================================================
# TASK 1: Linear Regression
# ============================================================================

def linear_regression_single_country(days, cumulative_cases, country_name):
    """
    Train a linear regression model on cumulative cases for one country.
    
    Parameters:
    -----------
    days : array
        Days since first outbreak
    cumulative_cases : array
        Cumulative number of cases
    country_name : str
        Name of the country
    
    Returns:
    --------
    model : LinearRegression
        Fitted model
    predictions : array
        Model predictions
    metrics : dict
        R², RMSE, coefficients
    """
    X = days.reshape(-1, 1)
    y = cumulative_cases
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'country': country_name
    }
    
    return model, predictions, metrics


def linear_regression_all_countries(data):
    """
    Train separate linear regression models for each country.
    
    Parameters:
    -----------
    data : dict
        Dictionary with country data from load_all_countries()
    
    Returns:
    --------
    results : dict
        Dictionary with models, predictions, and metrics for each country
    """
    results = {}
    
    for country, country_data in data.items():
        model, pred, metrics = linear_regression_single_country(
            country_data['days'],
            country_data['cumulative'],
            country
        )
        results[country] = {
            'model': model,
            'predictions': pred,
            'metrics': metrics
        }
    
    return results


def plot_linear_regression(data, results):
    """
    Plot linear regression results for all three countries.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (country, country_data) in enumerate(data.items()):
        ax = axes[idx]
        days = country_data['days']
        cumulative = country_data['cumulative']
        pred = results[country]['predictions']
        metrics = results[country]['metrics']
        
        ax.scatter(days, cumulative, color='red', alpha=0.6, label='Actual data')
        ax.plot(days, pred, 'b-', linewidth=2, label='Linear fit')
        
        ax.set_xlabel('Days since first outbreak')
        ax.set_ylabel('Cumulative cases')
        ax.set_title(f'{country}\nR²={metrics["r2"]:.4f}, RMSE={metrics["rmse"]:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# TASK 2: Better Fitting Function (Polynomial/Sigmoid)
# ============================================================================

def polynomial_fit(days, cumulative_cases, degree=3):
    """
    Fit polynomial regression to cumulative cases.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(days.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly, cumulative_cases)
    predictions = model.predict(X_poly)
    
    r2 = r2_score(cumulative_cases, predictions)
    rmse = np.sqrt(mean_squared_error(cumulative_cases, predictions))
    
    return model, poly, predictions, {'r2': r2, 'rmse': rmse, 'degree': degree}


def sigmoid_function(t, L, k, t0, b):
    """
    Generalized logistic (sigmoid) function.
    y = L / (1 + exp(-k*(t-t0))) + b
    
    L: maximum value (carrying capacity)
    k: steepness
    t0: midpoint
    b: baseline
    """
    return L / (1 + np.exp(-k * (t - t0))) + b


def fit_sigmoid(days, cumulative_cases):
    """
    Fit sigmoid function to cumulative cases.
    """
    # Initial guesses
    L_init = max(cumulative_cases)
    k_init = 0.02
    t0_init = days[len(days)//2]
    b_init = 0
    
    try:
        popt, _ = curve_fit(
            sigmoid_function, 
            days, 
            cumulative_cases,
            p0=[L_init, k_init, t0_init, b_init],
            bounds=([0, 0.001, 0, -1000], [L_init*2, 0.5, max(days), 1000]),
            maxfev=10000
        )
        predictions = sigmoid_function(days, *popt)
        r2 = r2_score(cumulative_cases, predictions)
        rmse = np.sqrt(mean_squared_error(cumulative_cases, predictions))
        
        return popt, predictions, {'r2': r2, 'rmse': rmse, 'params': popt}
    except Exception as e:
        print(f"Sigmoid fitting failed: {e}")
        return None, None, None


def combined_fit_all_countries(data):
    """
    Fit multiple models (linear, polynomial, sigmoid) to all countries
    combined or separately for comparison.
    """
    # Combine all data
    all_days = []
    all_cumulative = []
    all_countries = []
    
    for country, country_data in data.items():
        all_days.extend(country_data['days'])
        all_cumulative.extend(country_data['cumulative'])
        all_countries.extend([country] * len(country_data['days']))
    
    all_days = np.array(all_days)
    all_cumulative = np.array(all_cumulative)
    
    # Sort by days
    sort_idx = np.argsort(all_days)
    all_days = all_days[sort_idx]
    all_cumulative = all_cumulative[sort_idx]
    
    results = {}
    
    # Linear regression (single line for all)
    model_lin, pred_lin, metrics_lin = linear_regression_single_country(
        all_days, all_cumulative, 'All Countries'
    )
    results['linear'] = {'predictions': pred_lin, 'metrics': metrics_lin}
    
    # Polynomial regression
    for degree in [2, 3, 4]:
        _, _, pred_poly, metrics_poly = polynomial_fit(all_days, all_cumulative, degree)
        results[f'poly_{degree}'] = {'predictions': pred_poly, 'metrics': metrics_poly}
    
    return all_days, all_cumulative, results


# ============================================================================
# TASK 3: Neural Network
# ============================================================================

def create_nn_model(input_dim=1, hidden_layers=[64, 32], output_dim=1):
    """
    Create a simple feedforward neural network using TensorFlow/Keras.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_layers : list
        Number of neurons in each hidden layer
    output_dim : int
        Number of outputs
    
    Returns:
    --------
    model : keras.Model
        Compiled neural network model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for neurons in hidden_layers:
            model.add(layers.Dense(neurons, activation='relu'))
            model.add(layers.Dropout(0.1))
        
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    except ImportError:
        print("TensorFlow not available. Using sklearn MLPRegressor instead.")
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=tuple(hidden_layers), 
                           max_iter=1000, random_state=42)


def train_nn_model(X_train, y_train, X_test, y_test, epochs=500, batch_size=4, verbose=0):
    """
    Train a neural network model optimized for small epidemic datasets.
    
    Returns:
    --------
    model : trained model
    history : training history
    metrics : dict with evaluation metrics
    """
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1))
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        
        # Simpler model with regularization for small datasets
        model = keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), 
                      loss='mse', metrics=['mae'])
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss', patience=50, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=20, min_lr=0.0001
        )
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        # Predictions
        y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
    except ImportError:
        # Fallback to sklearn
        from sklearn.neural_network import MLPRegressor
        
        model = MLPRegressor(hidden_layer_sizes=(32, 16, 8), 
                            max_iter=2000, random_state=42, 
                            early_stopping=True, alpha=0.01,
                            learning_rate='adaptive')
        model.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        history = None
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {'r2': r2, 'rmse': rmse}
    
    return model, scaler_X, scaler_y, history, y_pred, metrics


def nn_predict_epidemic(data, test_size=0.2):
    """
    Train and evaluate neural network for epidemic prediction.
    
    IMPORTANT: For time series, we use temporal split (not random).
    """
    results = {}
    
    for country, country_data in data.items():
        days = country_data['days']
        cumulative = country_data['cumulative']
        
        # Temporal split (not random!)
        split_idx = int(len(days) * (1 - test_size))
        X_train, X_test = days[:split_idx], days[split_idx:]
        y_train, y_test = cumulative[:split_idx], cumulative[split_idx:]
        
        model, scaler_X, scaler_y, history, y_pred, metrics = train_nn_model(
            X_train, y_train, X_test, y_test
        )
        
        results[country] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics,
            'history': history
        }
    
    return results


# ============================================================================
# TASK 4: LSTM for Time Series
# ============================================================================

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM training.
    
    Parameters:
    -----------
    data : array
        Time series data
    seq_length : int
        Length of input sequences
    
    Returns:
    --------
    X : array of shape (n_samples, seq_length, 1)
    y : array of shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def create_lstm_model(seq_length, units=[50, 25]):
    """
    Create an LSTM model for time series prediction.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=(seq_length, 1)),
            layers.LSTM(units[0], return_sequences=True if len(units) > 1 else False),
            layers.Dropout(0.2),
        ])
        
        if len(units) > 1:
            model.add(layers.LSTM(units[1]))
            model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        return model
    except ImportError:
        print("TensorFlow not available for LSTM.")
        return None


def train_lstm_model(cumulative_cases, seq_length=10, test_size=0.2, epochs=100, verbose=0):
    """
    Train LSTM model for epidemic time series prediction.
    
    Parameters:
    -----------
    cumulative_cases : array
        Cumulative case data
    seq_length : int
        Sequence length for LSTM input
    test_size : float
        Fraction of data for testing
    epochs : int
        Training epochs
    
    Returns:
    --------
    model : trained LSTM model
    scaler : fitted MinMaxScaler
    metrics : dict with evaluation metrics
    predictions : array of predictions
    """
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(cumulative_cases.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    X = X.reshape(-1, seq_length, 1)
    
    # Temporal split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    try:
        from tensorflow import keras
        
        model = create_lstm_model(seq_length)
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=8,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Predictions
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        
        return model, scaler, {'r2': r2, 'rmse': rmse}, y_pred, y_test_orig, history
        
    except ImportError:
        print("TensorFlow not available for LSTM.")
        return None, scaler, None, None, None, None


def lstm_predict_all_countries(data, seq_length=10, test_size=0.2, epochs=100):
    """
    Train and evaluate LSTM models for all countries.
    """
    results = {}
    
    for country, country_data in data.items():
        cumulative = country_data['cumulative']
        
        if len(cumulative) < seq_length + 10:
            print(f"Skipping {country}: not enough data for seq_length={seq_length}")
            continue
        
        model, scaler, metrics, y_pred, y_test, history = train_lstm_model(
            cumulative, seq_length=seq_length, test_size=test_size, epochs=epochs
        )
        
        results[country] = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_test': y_test,
            'history': history,
            'seq_length': seq_length
        }
    
    return results


# ============================================================================
# TASK 5: Comparison and Discussion
# ============================================================================

def compare_all_methods(data, seir_params=None):
    """
    Compare all prediction methods: SEIR model, Linear, Polynomial, NN, LSTM.
    
    Parameters:
    -----------
    data : dict
        Country data
    seir_params : dict
        SEIR model parameters {sigma, gamma, N}
    
    Returns:
    --------
    comparison_df : DataFrame
        Comparison of all methods
    """
    if seir_params is None:
        seir_params = {
            'sigma': 1/9.7,  # incubation period
            'gamma': 1/7.0,  # infectious period
            'N': 1e7         # population
        }
    
    results = []
    
    for country, country_data in data.items():
        days = country_data['days']
        cumulative = country_data['cumulative']
        
        # 1. SEIR Model
        beta0, lambda_fit, seir_pred = fit_seir_to_data(
            days, cumulative, 
            seir_params['sigma'], seir_params['gamma'], seir_params['N']
        )
        if seir_pred is not None:
            seir_r2 = r2_score(cumulative, seir_pred)
            seir_rmse = np.sqrt(mean_squared_error(cumulative, seir_pred))
            results.append({
                'Country': country, 'Method': 'SEIR Model',
                'R2': seir_r2, 'RMSE': seir_rmse
            })
        
        # 2. Linear Regression
        _, lin_pred, lin_metrics = linear_regression_single_country(
            days, cumulative, country
        )
        results.append({
            'Country': country, 'Method': 'Linear Regression',
            'R2': lin_metrics['r2'], 'RMSE': lin_metrics['rmse']
        })
        
        # 3. Polynomial (degree 3)
        _, _, poly_pred, poly_metrics = polynomial_fit(days, cumulative, degree=3)
        results.append({
            'Country': country, 'Method': 'Polynomial (deg=3)',
            'R2': poly_metrics['r2'], 'RMSE': poly_metrics['rmse']
        })
        
        # 4. Sigmoid
        _, sig_pred, sig_metrics = fit_sigmoid(days, cumulative)
        if sig_metrics is not None:
            results.append({
                'Country': country, 'Method': 'Sigmoid',
                'R2': sig_metrics['r2'], 'RMSE': sig_metrics['rmse']
            })
    
    return pd.DataFrame(results)


def plot_comparison(data, comparison_df):
    """
    Create comparison visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: R² comparison
    ax1 = axes[0, 0]
    pivot_r2 = comparison_df.pivot(index='Method', columns='Country', values='R2')
    pivot_r2.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('R² Score Comparison', fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.legend(title='Country')
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Good fit threshold')
    
    # Plot 2: RMSE comparison
    ax2 = axes[0, 1]
    pivot_rmse = comparison_df.pivot(index='Method', columns='Country', values='RMSE')
    pivot_rmse.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('RMSE Comparison', fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.legend(title='Country')
    
    # Plot 3: Best method per country (R²)
    ax3 = axes[1, 0]
    best_r2 = comparison_df.loc[comparison_df.groupby('Country')['R2'].idxmax()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(best_r2)))
    bars = ax3.bar(best_r2['Country'], best_r2['R2'], color=colors)
    ax3.set_title('Best R² Score per Country', fontweight='bold')
    ax3.set_ylabel('R² Score')
    for bar, method in zip(bars, best_r2['Method']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 method, ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Summary:\n\n"
    for country in comparison_df['Country'].unique():
        country_data = comparison_df[comparison_df['Country'] == country]
        best_method = country_data.loc[country_data['R2'].idxmax(), 'Method']
        best_r2 = country_data['R2'].max()
        summary_text += f"{country}:\n  Best: {best_method}\n  R²: {best_r2:.4f}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss.
    """
    if history is None:
        print("No training history available.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_comprehensive_figure(data, linear_results, nn_results, lstm_results=None):
    """
    Create a comprehensive figure showing all results.
    """
    n_countries = len(data)
    fig, axes = plt.subplots(n_countries, 3, figsize=(18, 5*n_countries))
    
    if n_countries == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (country, country_data) in enumerate(data.items()):
        days = country_data['days']
        cumulative = country_data['cumulative']
        
        # Column 1: Data + Linear
        ax1 = axes[idx, 0]
        ax1.scatter(days, cumulative, color='red', alpha=0.6, label='Actual')
        ax1.plot(days, linear_results[country]['predictions'], 'b-', 
                 linewidth=2, label='Linear')
        ax1.set_title(f'{country} - Linear Regression')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Cumulative Cases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Column 2: NN predictions
        ax2 = axes[idx, 1]
        nn_res = nn_results[country]
        ax2.scatter(nn_res['X_train'], nn_res['y_train'], color='blue', 
                    alpha=0.5, label='Train')
        ax2.scatter(nn_res['X_test'], nn_res['y_test'], color='red', 
                    alpha=0.5, label='Test (actual)')
        ax2.scatter(nn_res['X_test'], nn_res['y_pred'], color='green', 
                    marker='x', s=100, label='Test (predicted)')
        ax2.set_title(f'{country} - Neural Network\nR²={nn_res["metrics"]["r2"]:.4f}')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Cumulative Cases')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Column 3: LSTM predictions (if available)
        ax3 = axes[idx, 2]
        if lstm_results and country in lstm_results and lstm_results[country]['y_pred'] is not None:
            lstm_res = lstm_results[country]
            ax3.plot(lstm_res['y_test'], 'b-', label='Actual', linewidth=2)
            ax3.plot(lstm_res['y_pred'], 'r--', label='LSTM Predicted', linewidth=2)
            ax3.set_title(f'{country} - LSTM\nR²={lstm_res["metrics"]["r2"]:.4f}')
        else:
            ax3.text(0.5, 0.5, 'LSTM not available\nor insufficient data',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'{country} - LSTM')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Cumulative Cases')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION HELPER
# ============================================================================

def run_complete_analysis(data_path):
    """
    Run the complete analysis pipeline for Assignment 4, Topic 2.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing the .dat files
    
    Returns:
    --------
    all_results : dict
        Dictionary containing all results
    """
    print("=" * 70)
    print("ASSIGNMENT 4 - TOPIC 2: Supervised Learning")
    print("Machines versus Human Models - Who Can Save the World?")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading Ebola data...")
    data = load_all_countries(data_path)
    for country, cdata in data.items():
        print(f"    {country}: {len(cdata['days'])} data points, "
              f"total cases: {cdata['cumulative'][-1]:.0f}")
    
    # SEIR Model parameters
    seir_params = {
        'sigma': 1/9.7,
        'gamma': 1/7.0,
        'N': 1e7
    }
    
    # Task 1: Linear Regression
    print("\n[2] Task 1: Linear Regression...")
    linear_results = linear_regression_all_countries(data)
    for country, res in linear_results.items():
        print(f"    {country}: R²={res['metrics']['r2']:.4f}, "
              f"RMSE={res['metrics']['rmse']:.1f}")
    
    # Task 2: Better fitting
    print("\n[3] Task 2: Better Fitting Functions...")
    for country, cdata in data.items():
        _, _, poly_pred, poly_metrics = polynomial_fit(
            cdata['days'], cdata['cumulative'], degree=3
        )
        print(f"    {country} Polynomial(3): R²={poly_metrics['r2']:.4f}")
    
    # Task 3: Neural Network
    print("\n[4] Task 3: Neural Network...")
    nn_results = nn_predict_epidemic(data, test_size=0.2)
    for country, res in nn_results.items():
        print(f"    {country}: R²={res['metrics']['r2']:.4f}, "
              f"RMSE={res['metrics']['rmse']:.1f}")
    
    # Task 4: LSTM
    print("\n[5] Task 4: LSTM...")
    lstm_results = lstm_predict_all_countries(data, seq_length=5, test_size=0.2, epochs=50)
    for country, res in lstm_results.items():
        if res['metrics'] is not None:
            print(f"    {country}: R²={res['metrics']['r2']:.4f}, "
                  f"RMSE={res['metrics']['rmse']:.1f}")
        else:
            print(f"    {country}: LSTM not available")
    
    # Comparison
    print("\n[6] Comparing all methods...")
    comparison_df = compare_all_methods(data, seir_params)
    print(comparison_df.to_string(index=False))
    
    return {
        'data': data,
        'linear_results': linear_results,
        'nn_results': nn_results,
        'lstm_results': lstm_results,
        'comparison': comparison_df,
        'seir_params': seir_params
    }


if __name__ == "__main__":
    # Example usage
    print("This module should be imported into a Jupyter notebook.")
    print("Use: from ebola_ml_functions import *")
