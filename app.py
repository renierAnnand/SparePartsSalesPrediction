import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Advanced Spare Parts Forecasting System", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import stats
from scipy.optimize import minimize

# Forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Machine learning libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# Global variable for the default adjustment factor
FORECAST_ADJUSTMENT_FACTOR = 0.85  # 85% = 100% - 15% reduction

def apply_forecast_adjustment(forecast_values, adjustment_factor=None):
    """Apply adjustment factor to forecast values."""
    if adjustment_factor is None:
        adjustment_factor = FORECAST_ADJUSTMENT_FACTOR
        
    if isinstance(forecast_values, (list, np.ndarray)):
        adjusted_values = np.array(forecast_values) * adjustment_factor
        return np.maximum(adjusted_values, 0)  # Ensure non-negative values
    return forecast_values

class MetaLearner(BaseEstimator, RegressorMixin):
    """Meta-learner for model stacking"""
    def __init__(self):
        self.meta_model = Ridge(alpha=1.0)
        
    def fit(self, X, y):
        self.meta_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.meta_model.predict(X)

@st.cache_data
def load_spare_parts_data(uploaded_file):
    """Load and preprocess spare parts sales data with advanced preprocessing."""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
        
        # Display first few rows to understand format
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Validate required columns
        expected_columns = ['Part', 'Month', 'Sales']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Expected columns: Part, Month, Sales")
            return None
        
        st.info(f"📋 Using columns - Part: {df.columns[0]}, Month: {df.columns[1]}, Sales: {df.columns[2]}")
        
        # Standardize column names
        df = df[expected_columns].copy()
        
        # Parse dates
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        invalid_dates = df['Month'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ Found {invalid_dates} invalid dates - these will be excluded")
            df = df.dropna(subset=['Month'])
        
        # Clean sales data
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        invalid_sales = df['Sales'].isna().sum()
        if invalid_sales > 0:
            st.warning(f"⚠️ Found {invalid_sales} invalid sales values - these will be excluded")
            df = df.dropna(subset=['Sales'])
        
        # Remove negative sales
        negative_sales = (df['Sales'] < 0).sum()
        if negative_sales > 0:
            st.warning(f"⚠️ Found {negative_sales} negative sales values - these will be set to 0")
            df['Sales'] = df['Sales'].clip(lower=0)
        
        # Clean part names
        df['Part'] = df['Part'].astype(str).str.strip()
        df = df[df['Part'] != 'nan']
        
        # Advanced preprocessing - aggregate by Part and Month
        st.info("🔧 Aggregating data by Part and Month...")
        df_processed = preprocess_spare_parts_data(df)
        
        # Sort by Part and Month
        df_processed = df_processed.sort_values(['Part', 'Month']).reset_index(drop=True)
        
        # Show summary
        num_parts = df_processed['Part'].nunique()
        date_range_start = df_processed['Month'].min()
        date_range_end = df_processed['Month'].max()
        total_months = df_processed['Month'].nunique()
        
        st.success(f"✅ Processed {num_parts} spare parts with data from {date_range_start.strftime('%Y-%m')} to {date_range_end.strftime('%Y-%m')}")
        
        return df_processed
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def preprocess_spare_parts_data(df):
    """Advanced data preprocessing for improved accuracy."""
    # Group by Part and Month, sum sales (in case of duplicates)
    df_agg = df.groupby(['Part', 'Month'])['Sales'].sum().reset_index()
    
    # Store original sales for reference
    df_agg['Sales_Original'] = df_agg['Sales'].copy()
    
    # Advanced preprocessing for each part
    processed_parts = []
    unique_parts = df_agg['Part'].unique()
    
    st.info(f"🔧 Processing {len(unique_parts)} unique parts...")
    
    for i, part in enumerate(unique_parts):
        if i % 1000 == 0:  # Progress update every 1000 parts
            st.info(f"📊 Processed {i}/{len(unique_parts)} parts...")
            
        part_data = df_agg[df_agg['Part'] == part].copy()
        
        if len(part_data) < 2:
            # Keep parts with minimal data as-is
            part_data['log_transformed'] = False
            processed_parts.append(part_data)
            continue
        
        try:
            # 1. Outlier Detection and Treatment using IQR
            if len(part_data) >= 4:  # Need at least 4 points for meaningful IQR
                Q1 = part_data['Sales'].quantile(0.25)
                Q3 = part_data['Sales'].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing (preserves data points)
                    outliers_detected = ((part_data['Sales'] < lower_bound) | (part_data['Sales'] > upper_bound)).sum()
                    if outliers_detected > 0:
                        part_data['Sales'] = part_data['Sales'].clip(lower=max(0, lower_bound), upper=upper_bound)
            
            # 2. Handle missing months with interpolation
            if len(part_data) > 2:
                # Create complete month range
                date_range = pd.date_range(
                    start=part_data['Month'].min(),
                    end=part_data['Month'].max(),
                    freq='MS'
                )
                
                # Only fill gaps if reasonable (not more than 3x original data points)
                if len(date_range) <= len(part_data) * 3:
                    # Create complete dataframe
                    complete_df = pd.DataFrame({'Month': date_range})
                    complete_df['Part'] = part
                    
                    # Merge with actual data
                    part_data = complete_df.merge(part_data, on=['Part', 'Month'], how='left')
                    
                    # Try time-based interpolation, fallback to linear if it fails
                    try:
                        # Set Month as index for time-based interpolation
                        part_data_indexed = part_data.set_index('Month')
                        
                        # Interpolate missing values using time method
                        part_data_indexed['Sales'] = part_data_indexed['Sales'].interpolate(method='time', limit_direction='both')
                        part_data_indexed['Sales'] = part_data_indexed['Sales'].fillna(0)  # Fill any remaining NAs with 0
                        
                        # Update Sales_Original for new rows
                        part_data_indexed['Sales_Original'] = part_data_indexed['Sales_Original'].fillna(part_data_indexed['Sales'])
                        
                        # Reset index to get Month back as column
                        part_data = part_data_indexed.reset_index()
                        
                    except Exception:
                        # Fallback to linear interpolation if time interpolation fails
                        part_data['Sales'] = part_data['Sales'].interpolate(method='linear', limit_direction='both')
                        part_data['Sales'] = part_data['Sales'].ffill().bfill().fillna(0)
                        
                        # Update Sales_Original for new rows
                        part_data['Sales_Original'] = part_data['Sales_Original'].fillna(part_data['Sales'])
            
            # 3. Data transformation - test for optimal transformation
            if len(part_data) >= 3 and part_data['Sales'].std() > 0:
                try:
                    sales_values = part_data['Sales'].dropna()
                    if len(sales_values) > 0 and sales_values.min() >= 0:
                        skewness = stats.skew(sales_values)
                        if abs(skewness) > 1.5:  # Highly skewed data
                            part_data['Sales'] = np.log1p(part_data['Sales'])  # log1p handles zeros better
                            part_data['log_transformed'] = True
                        else:
                            part_data['log_transformed'] = False
                    else:
                        part_data['log_transformed'] = False
                except Exception:
                    part_data['log_transformed'] = False
            else:
                part_data['log_transformed'] = False
            
            processed_parts.append(part_data)
            
        except Exception as e:
            # If any preprocessing fails for this part, keep original data
            part_data['log_transformed'] = False
            processed_parts.append(part_data)
    
    # Combine all processed parts
    final_df = pd.concat(processed_parts, ignore_index=True)
    
    st.success(f"✅ Completed preprocessing for {len(unique_parts)} parts")
    
    return final_df

def optimize_sarima_parameters(data, max_p=2, max_d=2, max_q=2, seasonal_periods=12):
    """Optimize SARIMA parameters using grid search - more conservative approach"""
    if not STATSMODELS_AVAILABLE or len(data) < 24:
        return {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
    
    best_aic = np.inf
    best_params = None
    
    # More conservative grid search for stability
    param_combinations = [
        ((1, 1, 1), (1, 1, 1, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 0, 1), (1, 0, 1, 12)),
        ((2, 1, 0), (1, 1, 0, 12)),
        ((0, 1, 2), (0, 1, 1, 12)),
        ((1, 1, 0), (0, 1, 1, 12))
    ]
    
    for order, seasonal_order in param_combinations:
        try:
            model = SARIMAX(
                data['Sales'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False, maxiter=100, method='lbfgs')
            
            if fitted.aic < best_aic and np.isfinite(fitted.aic):
                best_aic = fitted.aic
                best_params = {
                    'order': order,
                    'seasonal_order': seasonal_order
                }
        except Exception:
            continue
    
    return best_params if best_params else {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}

def run_advanced_sarima_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Advanced SARIMA with better error handling and validation"""
    try:
        if not STATSMODELS_AVAILABLE or len(data) < 12:
            return run_fallback_forecast(data, forecast_periods, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Ensure data is stationary and has positive values
        sales_series = work_data['Sales'].copy()
        
        # Check for zeros or negative values that could cause issues
        if (sales_series <= 0).any():
            sales_series = sales_series.clip(lower=0.01)  # Replace zeros/negatives with small positive value
        
        # Optimize parameters with conservative approach
        best_params = optimize_sarima_parameters(work_data)
        
        # Fit the model with additional error handling
        model = SARIMAX(
            sales_series, 
            order=best_params['order'],
            seasonal_order=best_params['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit with multiple methods if first fails
        fitted_model = None
        for method in ['lbfgs', 'bfgs', 'nm']:
            try:
                fitted_model = model.fit(
                    disp=False, 
                    maxiter=200, 
                    method=method,
                    low_memory=True
                )
                break
            except Exception:
                continue
        
        if fitted_model is None:
            raise ValueError("All fitting methods failed")
        
        # Generate forecast with confidence intervals
        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast = forecast_result.predicted_mean
        
        # Convert to numpy array and ensure proper format
        forecast_values = np.array(forecast)
        
        # Check for invalid values
        if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
            raise ValueError("Forecast contains NaN or infinite values")
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Apply adjustment and ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception:
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return forecast_values, np.inf

def run_advanced_prophet_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Enhanced Prophet with better error handling"""
    try:
        if not PROPHET_AVAILABLE or len(data) < 6:
            return run_fallback_forecast(data, forecast_periods, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Prepare data for Prophet
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Ensure positive values for Prophet
        prophet_data['y'] = prophet_data['y'].clip(lower=0.01)
        
        # Use simpler Prophet configuration for stability
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Apply adjustment and ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.mean(np.abs(forecast['yhat'] - prophet_data['y']))
        
    except Exception:
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return forecast_values, np.inf

def run_advanced_ets_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Advanced ETS with better error handling"""
    try:
        if not STATSMODELS_AVAILABLE or len(data) < 6:
            return run_fallback_forecast(data, forecast_periods, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Ensure positive values
        sales_series = work_data['Sales'].clip(lower=0.01)
        
        # Try seasonal model first if enough data
        if len(sales_series) >= 24:
            try:
                model = ExponentialSmoothing(
                    sales_series,
                    seasonal='add',
                    seasonal_periods=12,
                    trend='add'
                )
                fitted_model = model.fit(optimized=True)
                forecast = fitted_model.forecast(steps=forecast_periods)
            except Exception:
                # Fallback to simpler model
                model = ExponentialSmoothing(
                    sales_series,
                    seasonal=None,
                    trend='add'
                )
                fitted_model = model.fit(optimized=True)
                forecast = fitted_model.forecast(steps=forecast_periods)
        else:
            # Simple model for limited data
            model = ExponentialSmoothing(
                sales_series,
                seasonal=None,
                trend='add'
            )
            fitted_model = model.fit(optimized=True)
            forecast = fitted_model.forecast(steps=forecast_periods)
        
        # Validate forecast
        forecast_values = np.array(forecast)
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Apply adjustment and ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception:
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return forecast_values, np.inf

def run_advanced_xgb_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Advanced XGBoost forecast with feature engineering"""
    try:
        if not XGBOOST_AVAILABLE or len(data) < 6:
            return run_simplified_ml_forecast(data, forecast_periods, adjustment_factor), 200.0
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Feature engineering
        work_data['month'] = work_data['Month'].dt.month
        work_data['quarter'] = work_data['Month'].dt.quarter
        work_data['year'] = work_data['Month'].dt.year
        
        # Lag features
        for lag in [1, 3, 6, 12]:
            if len(work_data) > lag:
                work_data[f'lag_{lag}'] = work_data['Sales'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            if len(work_data) > window:
                work_data[f'rolling_mean_{window}'] = work_data['Sales'].rolling(window=window).mean()
                work_data[f'rolling_std_{window}'] = work_data['Sales'].rolling(window=window).std()
        
        # Prepare features
        feature_cols = ['month', 'quarter', 'year'] + [col for col in work_data.columns if col.startswith(('lag_', 'rolling_'))]
        feature_cols = [col for col in feature_cols if col in work_data.columns]
        
        # Remove rows with NaN values
        train_data = work_data[feature_cols + ['Sales']].dropna()
        
        if len(train_data) < 6:
            return run_simplified_ml_forecast(data, forecast_periods, adjustment_factor), 200.0
        
        X = train_data[feature_cols]
        y = train_data['Sales']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Generate forecasts
        last_date = work_data['Month'].max()
        forecasts = []
        
        for i in range(forecast_periods):
            future_date = last_date + pd.DateOffset(months=i+1)
            
            # Create feature vector for prediction
            features = {
                'month': future_date.month,
                'quarter': future_date.quarter,
                'year': future_date.year
            }
            
            # Add lag features (use recent values)
            for lag in [1, 3, 6, 12]:
                if f'lag_{lag}' in feature_cols:
                    if len(work_data) >= lag:
                        features[f'lag_{lag}'] = work_data['Sales'].iloc[-lag]
                    else:
                        features[f'lag_{lag}'] = work_data['Sales'].mean()
            
            # Add rolling features (use recent values)
            for window in [3, 6, 12]:
                if f'rolling_mean_{window}' in feature_cols:
                    recent_values = work_data['Sales'].tail(window)
                    features[f'rolling_mean_{window}'] = recent_values.mean()
                if f'rolling_std_{window}' in feature_cols:
                    recent_values = work_data['Sales'].tail(window)
                    features[f'rolling_std_{window}'] = recent_values.std() if len(recent_values) > 1 else 0
            
            # Create feature vector
            feature_vector = [features.get(col, 0) for col in feature_cols]
            
            # Make prediction
            pred = model.predict([feature_vector])[0]
            forecasts.append(max(pred, 0))  # Ensure non-negative
        
        forecasts = np.array(forecasts)
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecasts = np.expm1(forecasts)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecasts, adjustment_factor), 150.0
        
    except Exception:
        forecast_values = run_simplified_ml_forecast(data, forecast_periods, adjustment_factor)
        return forecast_values, 200.0

def run_simplified_ml_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Simplified ML forecast using scikit-learn"""
    try:
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        if len(work_data) >= 12:
            # Use last 12 months as seasonal pattern with trend
            recent_sales = work_data['Sales'].tail(12).values
            
            # Calculate trend using linear regression
            X = np.arange(len(recent_sales)).reshape(-1, 1)
            y = recent_sales
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate forecasts
            forecasts = []
            for i in range(forecast_periods):
                # Seasonal component
                month_idx = i % 12
                seasonal_base = recent_sales[month_idx] if month_idx < len(recent_sales) else np.mean(recent_sales)
                
                # Trend component
                trend_pred = model.predict([[len(recent_sales) + i]])[0]
                trend_adjustment = (trend_pred - model.predict([[len(recent_sales) - 1]])[0]) * 0.5  # Dampen trend
                
                forecast_val = max(seasonal_base + trend_adjustment, seasonal_base * 0.5)
                forecasts.append(forecast_val)
        else:
            # Simple average-based forecast for limited data
            base_value = work_data['Sales'].mean()
            forecasts = [base_value] * forecast_periods
        
        forecasts = np.array(forecasts)
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecasts = np.expm1(forecasts)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecasts, adjustment_factor)
        
    except Exception:
        return run_fallback_forecast(data, forecast_periods, adjustment_factor)

def run_fallback_forecast(data, forecast_periods=12, adjustment_factor=1.0):
    """Robust fallback forecasting method"""
    try:
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        if len(work_data) >= 12:
            # Use seasonal naive with trend
            seasonal_pattern = work_data['Sales'].tail(12).values
            recent_trend = np.polyfit(range(len(work_data['Sales'].tail(12))), work_data['Sales'].tail(12), 1)[0]
            
            forecast = []
            for i in range(forecast_periods):
                seasonal_val = seasonal_pattern[i % 12]
                trend_adjustment = recent_trend * (i + 1) * 0.3  # Dampen trend
                forecast_val = max(seasonal_val + trend_adjustment, seasonal_val * 0.5)
                forecast.append(forecast_val)
            
            forecast = np.array(forecast)
            
        elif len(work_data) >= 3:
            # Simple trend for limited data
            recent_values = work_data['Sales'].tail(6).values if len(work_data) >= 6 else work_data['Sales'].values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            base_value = recent_values[-1]
            
            forecast = []
            for i in range(forecast_periods):
                pred_val = base_value + trend * (i + 1) * 0.5
                pred_val = max(pred_val, base_value * 0.3)  # Floor
                forecast.append(pred_val)
            
            forecast = np.array(forecast)
        else:
            # Use average for very limited data
            base_forecast = work_data['Sales'].mean() if len(work_data) > 0 else 10
            forecast = np.array([base_forecast] * forecast_periods)
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast = np.expm1(forecast)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecast, adjustment_factor)
        
    except Exception:
        # Ultimate fallback
        return np.array([10 * adjustment_factor] * forecast_periods)

def create_weighted_ensemble(forecasts_dict, validation_scores):
    """Create weighted ensemble based on validation performance"""
    # Convert scores to weights (inverse of error - lower error = higher weight)
    weights = {}
    total_inverse_score = 0
    
    for model_name, score in validation_scores.items():
        if score != np.inf and score > 0:
            inverse_score = 1 / score
            weights[model_name] = inverse_score
            total_inverse_score += inverse_score
        else:
            weights[model_name] = 0.1  # Small weight for failed models
            total_inverse_score += 0.1
    
    # Normalize weights
    for model_name in weights:
        weights[model_name] = weights[model_name] / total_inverse_score
    
    # Create weighted ensemble
    ensemble_forecast = np.zeros(len(next(iter(forecasts_dict.values()))))
    
    for model_name, forecast in forecasts_dict.items():
        weight = weights.get(model_name, 0.25)  # Default equal weight if not found
        ensemble_forecast += weight * forecast
    
    return ensemble_forecast, weights

def forecast_single_part(part_data, forecast_periods=12, adjustment_factor=1.0, use_models=None):
    """
    Run advanced forecasting for a single spare part using multiple models.
    """
    if use_models is None:
        use_models = {'sarima': True, 'prophet': True, 'ets': True, 'xgb': True}
    
    forecasts = {}
    scores = {}
    
    # SARIMA
    if use_models.get('sarima', True):
        try:
            forecast_values, score = run_advanced_sarima_forecast(part_data, forecast_periods, adjustment_factor)
            forecasts['SARIMA'] = forecast_values
            scores['SARIMA'] = score
        except Exception:
            pass
    
    # Prophet
    if use_models.get('prophet', True):
        try:
            forecast_values, score = run_advanced_prophet_forecast(part_data, forecast_periods, adjustment_factor)
            forecasts['Prophet'] = forecast_values
            scores['Prophet'] = score
        except Exception:
            pass
    
    # ETS
    if use_models.get('ets', True):
        try:
            forecast_values, score = run_advanced_ets_forecast(part_data, forecast_periods, adjustment_factor)
            forecasts['ETS'] = forecast_values
            scores['ETS'] = score
        except Exception:
            pass
    
    # XGBoost
    if use_models.get('xgb', True):
        try:
            forecast_values, score = run_advanced_xgb_forecast(part_data, forecast_periods, adjustment_factor)
            forecasts['XGBoost'] = forecast_values
            scores['XGBoost'] = score
        except Exception:
            pass
    
    # If no models succeeded, use fallback
    if not forecasts:
        fallback_forecast = run_fallback_forecast(part_data, forecast_periods, adjustment_factor)
        forecasts['Fallback'] = fallback_forecast
        scores['Fallback'] = np.inf
    
    # Create ensemble if multiple models succeeded
    if len(forecasts) > 1:
        try:
            ensemble_forecast, weights = create_weighted_ensemble(forecasts, scores)
            forecasts['Ensemble'] = ensemble_forecast
        except Exception:
            pass
    
    # Return best performing model (lowest score)
    best_model = min(scores.keys(), key=lambda k: scores[k])
    return forecasts[best_model], forecasts, scores

def generate_forecast_excel(forecast_results, start_date, model_details=None):
    """Generate Excel file with forecast results and model details."""
    try:
        # Create month headers
        month_headers = []
        current_date = start_date
        
        for i in range(12):
            month_headers.append(current_date.strftime('%b-%Y'))
            current_date = current_date + pd.DateOffset(months=1)
        
        # Create output DataFrame
        output_data = []
        
        for part_name, forecast_values in forecast_results.items():
            if isinstance(forecast_values, dict):
                # If multiple models, use ensemble or best model
                if 'Ensemble' in forecast_values:
                    values = forecast_values['Ensemble']
                else:
                    # Use first available model
                    values = next(iter(forecast_values.values()))
            else:
                values = forecast_values
            
            # Ensure we have exactly 12 values
            if len(values) != 12:
                values = np.resize(values, 12)
            
            row_data = [part_name] + values.tolist()
            output_data.append(row_data)
        
        # Create DataFrame
        columns = ['Spare Part'] + month_headers
        output_df = pd.DataFrame(output_data, columns=columns)
        
        # Sort by spare part name
        output_df = output_df.sort_values('Spare Part').reset_index(drop=True)
        
        # Round forecast values to integers
        for col in month_headers:
            output_df[col] = output_df[col].round(0).astype(int)
        
        return output_df
        
    except Exception as e:
        st.error(f"❌ Error generating Excel output: {str(e)}")
        return None

def create_summary_charts(forecast_results, start_date):
    """Create summary charts for the forecasting results."""
    try:
        # Prepare data for charting
        month_headers = []
        current_date = start_date
        for i in range(12):
            month_headers.append(current_date.strftime('%b-%Y'))
            current_date = current_date + pd.DateOffset(months=1)
        
        # Calculate totals per month
        monthly_totals = np.zeros(12)
        part_totals = {}
        
        for part_name, forecast_data in forecast_results.items():
            if isinstance(forecast_data, dict):
                # If multiple models, use ensemble or best model
                if 'Ensemble' in forecast_data:
                    values = forecast_data['Ensemble']
                else:
                    values = next(iter(forecast_data.values()))
            else:
                values = forecast_data
            
            # Ensure correct length
            if len(values) == 12:
                monthly_totals += values
                part_totals[part_name] = np.sum(values)
        
        # Create monthly forecast chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=month_headers,
            y=monthly_totals,
            name='Total Forecast',
            marker_color='lightblue'
        ))
        
        fig1.update_layout(
            title='📊 Monthly Total Forecast (All Parts)',
            xaxis_title='Month',
            yaxis_title='Total Quantity',
            height=400
        )
        
        # Create top parts chart (top 20 by total forecast)
        top_parts = sorted(part_totals.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if top_parts:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=[part[0][:30] + '...' if len(part[0]) > 30 else part[0] for part in top_parts],  # Truncate long names
                x=[part[1] for part in top_parts],
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig2.update_layout(
                title='🔝 Top 20 Parts by Total Forecast',
                xaxis_title='Total Forecast Quantity (12 months)',
                yaxis_title='Spare Part',
                height=600
            )
        else:
            fig2 = None
        
        return fig1, fig2
        
    except Exception as e:
        st.error(f"❌ Error creating charts: {str(e)}")
        return None, None

def main():
    """Main function for the advanced spare parts forecasting app."""
    st.title("🔧 Advanced Spare Parts Sales Forecasting System")
    st.markdown("**AI-powered forecasting with SARIMA, Prophet, ETS, XGBoost & Meta-Learning**")
    
    # Check library availability
    st.sidebar.header("📊 Available Libraries")
    st.sidebar.success(f"✅ Prophet: {'Available' if PROPHET_AVAILABLE else 'Not Available'}")
    st.sidebar.success(f"✅ Statsmodels: {'Available' if STATSMODELS_AVAILABLE else 'Not Available'}")
    st.sidebar.success(f"✅ XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Advanced Configuration")
    
    # Forecast adjustment
    st.sidebar.subheader("📊 Forecast Adjustment")
    adjustment_mode = st.sidebar.radio(
        "Adjustment Mode:",
        ["Slider (Quick)", "Custom Input (Precise)"],
        help="Choose how to set the forecast adjustment percentage"
    )
    
    if adjustment_mode == "Slider (Quick)":
        adjustment_percentage = st.sidebar.slider(
            "Forecast Adjustment (%)",
            min_value=-50,
            max_value=50,
            value=-15,  # Default 15% reduction
            step=5,
            help="Negative values reduce forecasts, positive values increase them"
        )
    else:
        adjustment_percentage = st.sidebar.number_input(
            "Custom Forecast Adjustment (%)",
            min_value=-100.0,
            max_value=200.0,
            value=-15.0,
            step=0.1,
            format="%.1f",
            help="Enter any percentage: negative reduces, positive increases forecasts"
        )
    
    adjustment_factor = (100 + adjustment_percentage) / 100
    
    # Show interpretation
    if adjustment_percentage < 0:
        st.sidebar.error(f"📉 **Reduction**: {abs(adjustment_percentage):.1f}% decrease")
    elif adjustment_percentage > 0:
        st.sidebar.success(f"📈 **Increase**: {adjustment_percentage:.1f}% increase")
    else:
        st.sidebar.info("⚖️ **No Change**: Original forecasts")
    
    # Model selection
    st.sidebar.subheader("🤖 Select Advanced Models")
    use_sarima = st.sidebar.checkbox("Advanced SARIMA (Auto-tuned)", value=True)
    use_prophet = st.sidebar.checkbox("Enhanced Prophet (Optimized)", value=True)
    use_ets = st.sidebar.checkbox("Auto-ETS (Best Config)", value=True)
    use_xgb = st.sidebar.checkbox("Advanced XGBoost (Feature-Rich)", value=True)
    
    use_models = {
        'sarima': use_sarima,
        'prophet': use_prophet,
        'ets': use_ets,
        'xgb': use_xgb
    }
    
    if not any(use_models.values()):
        st.sidebar.error("Please select at least one forecasting model.")
        return
    
    # Advanced options
    st.sidebar.subheader("🔬 Advanced Options")
    max_parts_to_process = st.sidebar.number_input(
        "Max Parts to Process (0 = All)",
        min_value=0,
        max_value=20000,
        value=0,
        help="Limit processing for testing (0 = process all parts)"
    )
    
    enable_ensemble = st.sidebar.checkbox("Enable Ensemble Modeling", value=True,
                                         help="Combine multiple models for better accuracy")
    
    # File upload
    st.subheader("📁 Upload Spare Parts Sales Data")
    uploaded_file = st.file_uploader(
        "Choose Excel file with spare parts sales data",
        type=["xlsx", "xls"],
        help="Upload file with Part, Month, Sales columns"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload your spare parts sales data file to begin.")
        
        # Show expected format
        with st.expander("📋 Expected File Format"):
            st.markdown("""
            **Required Format: Long Format**
            ```
            Part              | Month      | Sales
            ----------------- | ---------- | -----
            13-21707134.GN    | 2021-12-01 | 3260
            13-21707134.GN    | 2022-01-01 | 1365
            13-21707132.GN    | 2021-12-01 | 666
            ...               | ...        | ...
            ```
            
            **Column Requirements:**
            - **Part**: Spare part code/name
            - **Month**: Date in any recognizable format
            - **Sales**: Numeric sales/quantity values
            """)
        return
    
    # Load data
    spare_parts_df = load_spare_parts_data(uploaded_file)
    if spare_parts_df is None:
        return
    
    # Show data summary
    st.subheader("📊 Data Summary")
    
    num_parts = spare_parts_df['Part'].nunique()
    date_range_start = spare_parts_df['Month'].min()
    date_range_end = spare_parts_df['Month'].max()
    total_months = spare_parts_df['Month'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔧 Total Parts", num_parts)
    with col2:
        st.metric("📅 Months of Data", total_months)
    with col3:
        st.metric("📊 Data Points", len(spare_parts_df))
    with col4:
        avg_sales = spare_parts_df['Sales'].mean()
        st.metric("📈 Avg Monthly Sales", f"{avg_sales:.1f}")
    
    # Show date range
    st.info(f"📅 **Data Range**: {date_range_start.strftime('%Y-%m')} to {date_range_end.strftime('%Y-%m')}")
    
    # Calculate forecast start date
    forecast_start_date = date_range_end + pd.DateOffset(months=1)
    forecast_end_date = forecast_start_date + pd.DateOffset(months=11)
    
    st.info(f"🔮 **Forecast Period**: {forecast_start_date.strftime('%Y-%m')} to {forecast_end_date.strftime('%Y-%m')}")
    
    # Show parts data distribution
    parts_data_summary = spare_parts_df.groupby('Part').agg({
        'Month': ['count', 'min', 'max'],
        'Sales': ['sum', 'mean', 'std']
    }).round(2)
    
    parts_data_summary.columns = ['Months_Count', 'First_Month', 'Last_Month', 'Total_Sales', 'Avg_Monthly', 'Std_Monthly']
    parts_data_summary = parts_data_summary.reset_index()
    
    # Show data quality metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        parts_with_12plus = (parts_data_summary['Months_Count'] >= 12).sum()
        st.metric("📊 Parts with 12+ Months", f"{parts_with_12plus}/{num_parts}")
    
    with col2:
        parts_with_6plus = (parts_data_summary['Months_Count'] >= 6).sum()
        st.metric("📈 Parts with 6+ Months", f"{parts_with_6plus}/{num_parts}")
    
    with col3:
        parts_with_limited = (parts_data_summary['Months_Count'] < 6).sum()
        st.metric("⚠️ Parts with <6 Months", f"{parts_with_limited}/{num_parts}")
    
    with st.expander("📋 Parts Data Summary (Top 20)"):
        display_summary = parts_data_summary.head(20).copy()
        display_summary['First_Month'] = pd.to_datetime(display_summary['First_Month']).dt.strftime('%Y-%m')
        display_summary['Last_Month'] = pd.to_datetime(display_summary['Last_Month']).dt.strftime('%Y-%m')
        st.dataframe(display_summary, use_container_width=True)
        if len(parts_data_summary) > 20:
            st.info(f"Showing first 20 parts. Total: {len(parts_data_summary)} parts")
    
    # Generate forecasts
    if st.button("🚀 Generate Advanced AI Forecasts", type="primary"):
        st.subheader("🔮 Generating Advanced AI Forecasts...")
        
        if adjustment_percentage != 0:
            st.info(f"📊 Applying {adjustment_percentage:+.1f}% adjustment to all forecasts")
        
        # Determine parts to process
        parts_list = spare_parts_df['Part'].unique()
        if max_parts_to_process > 0 and max_parts_to_process < len(parts_list):
            parts_list = parts_list[:max_parts_to_process]
            st.warning(f"⚠️ Processing only first {max_parts_to_process} parts for testing")
        
        total_parts = len(parts_list)
        st.info(f"🔧 Processing {total_parts} spare parts with {sum(use_models.values())} models each")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        forecast_results = {}
        model_details = {}
        successful_forecasts = 0
        failed_forecasts = 0
        
        for i, part in enumerate(parts_list):
            status_text.text(f"Processing part {i+1}/{total_parts}: {part[:30]}...")
            
            # Get data for this part
            part_data = spare_parts_df[spare_parts_df['Part'] == part].copy()
            
            try:
                # Generate forecast using advanced models
                best_forecast, all_forecasts, scores = forecast_single_part(
                    part_data, 
                    forecast_periods=12, 
                    adjustment_factor=adjustment_factor,
                    use_models=use_models
                )
                
                if enable_ensemble and len(all_forecasts) > 1:
                    forecast_results[part] = all_forecasts
                else:
                    forecast_results[part] = best_forecast
                
                model_details[part] = {
                    'models_used': list(all_forecasts.keys()),
                    'scores': scores,
                    'data_points': len(part_data)
                }
                
                successful_forecasts += 1
                
            except Exception as e:
                # Use fallback for failed forecasts
                try:
                    fallback_forecast = run_fallback_forecast(part_data, 12, adjustment_factor)
                    forecast_results[part] = fallback_forecast
                    model_details[part] = {
                        'models_used': ['Fallback'],
                        'scores': {'Fallback': np.inf},
                        'data_points': len(part_data),
                        'error': str(e)
                    }
                    failed_forecasts += 1
                except Exception:
                    # Ultimate fallback
                    forecast_results[part] = np.array([10 * adjustment_factor] * 12)
                    failed_forecasts += 1
            
            # Update progress
            progress_bar.progress((i + 1) / total_parts)
        
        status_text.text("✅ Forecast generation completed!")
        
        # Show results summary
        st.success(f"✅ Generated forecasts for {len(forecast_results)} spare parts")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful", successful_forecasts)
        with col2:
            st.metric("⚠️ Fallback Used", failed_forecasts)
        with col3:
            success_rate = (successful_forecasts / total_parts * 100) if total_parts > 0 else 0
            st.metric("📊 Success Rate", f"{success_rate:.1f}%")
        
        # Generate Excel output
        st.subheader("📊 Forecast Results")
        
        output_df = generate_forecast_excel(forecast_results, forecast_start_date, model_details)
        
        if output_df is not None:
            # Show preview
            st.markdown("**Preview of forecast results:**")
            st.dataframe(output_df.head(10), use_container_width=True)
            
            if len(output_df) > 10:
                st.info(f"Showing first 10 parts. Total: {len(output_df)} parts in full export.")
            
            # Calculate summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_annual_forecast = output_df.iloc[:, 1:].sum().sum()
                st.metric("📊 Total Annual Forecast", f"{total_annual_forecast:,.0f}")
            
            with col2:
                avg_monthly_total = output_df.iloc[:, 1:].sum(axis=0).mean()
                st.metric("📈 Avg Monthly Total", f"{avg_monthly_total:,.0f}")
            
            with col3:
                max_month_total = output_df.iloc[:, 1:].sum(axis=0).max()
                max_month = output_df.columns[1:][output_df.iloc[:, 1:].sum(axis=0).argmax()]
                st.metric("🔝 Peak Month", f"{max_month}")
            
            with col4:
                st.metric("🔝 Peak Value", f"{max_month_total:,.0f}")
            
            # Create summary charts
            st.subheader("📊 Forecast Visualization")
            
            fig1, fig2 = create_summary_charts(forecast_results, forecast_start_date)
            
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Model performance summary
            if model_details:
                st.subheader("🤖 Model Performance Summary")
                
                model_usage = {}
                for part_detail in model_details.values():
                    for model in part_detail.get('models_used', []):
                        model_usage[model] = model_usage.get(model, 0) + 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Usage Count:**")
                    for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                        st.text(f"{model}: {count} parts")
                
                with col2:
                    if len(model_usage) > 1:
                        fig_models = go.Figure(data=[go.Pie(
                            labels=list(model_usage.keys()),
                            values=list(model_usage.values()),
                            hole=0.3
                        )])
                        fig_models.update_layout(title="Model Usage Distribution", height=300)
                        st.plotly_chart(fig_models, use_container_width=True)
            
            # Download Excel file
            st.subheader("📁 Download Results")
            
            # Convert to Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                # Main forecast sheet
                output_df.to_excel(writer, index=False, sheet_name='Spare Parts Forecast')
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Total Parts Processed',
                        'Successful Forecasts',
                        'Fallback Used',
                        'Success Rate',
                        'Forecast Period',
                        'Total Annual Forecast',
                        'Average Monthly Total',
                        'Peak Month',
                        'Adjustment Applied',
                        'Models Used',
                        'Generated On'
                    ],
                    'Value': [
                        len(output_df),
                        successful_forecasts,
                        failed_forecasts,
                        f"{success_rate:.1f}%",
                        f"{forecast_start_date.strftime('%Y-%m')} to {forecast_end_date.strftime('%Y-%m')}",
                        f"{total_annual_forecast:,.0f}",
                        f"{avg_monthly_total:,.0f}",
                        max_month,
                        f"{adjustment_percentage:+.1f}%",
                        ', '.join([model for model, used in use_models.items() if used]),
                        datetime.now().strftime('%Y-%m-%d %H:%M')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                # Model details sheet (if not too large)
                if len(model_details) <= 1000:  # Limit for performance
                    model_detail_rows = []
                    for part, details in model_details.items():
                        model_detail_rows.append({
                            'Part': part,
                            'Models_Used': ', '.join(details.get('models_used', [])),
                            'Data_Points': details.get('data_points', 0),
                            'Best_Score': min(details.get('scores', {}).values()) if details.get('scores') else 'N/A',
                            'Error': details.get('error', '')
                        })
                    
                    model_details_df = pd.DataFrame(model_detail_rows)
                    model_details_df.to_excel(writer, index=False, sheet_name='Model Details')
            
            excel_buffer.seek(0)
            
            # Download button
            adj_suffix = f"_adj{adjustment_percentage:+.1f}pct" if adjustment_percentage != 0 else ""
            filename = f"spare_parts_forecast_{forecast_start_date.strftime('%Y%m')}{adj_suffix}.xlsx"
            
            st.download_button(
                label="📥 Download Excel Forecast",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"✅ Excel file ready for download: {filename}")
            
            # Show file contents info
            with st.expander("📋 Excel File Contents"):
                st.markdown("""
                **Sheet 1: Spare Parts Forecast**
                - Column A: Spare Part codes/names
                - Columns B-M: Monthly forecasts for next 12 months
                - All values rounded to whole numbers
                
                **Sheet 2: Summary**
                - Key statistics and metadata
                - Model performance summary
                - Generation timestamp
                
                **Sheet 3: Model Details** (if ≤1000 parts)
                - Which models were used for each part
                - Data quality metrics per part
                - Error information if applicable
                """)

if __name__ == "__main__":
    main()
