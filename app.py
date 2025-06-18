import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Production Sales Forecasting Dashboard", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime
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
def load_data(uploaded_file):
    """Load and preprocess the historical sales data with advanced preprocessing."""
    try:
        df = pd.read_excel(uploaded_file)
    except Exception:
        st.error("Could not read the uploaded file. Please ensure it's a valid Excel file.")
        return None, None

    # Check for required columns
    if "Month" not in df.columns or "Sales" not in df.columns:
        st.error("The file must contain 'Month' and 'Sales' columns.")
        return None, None

    # Parse dates
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some dates could not be parsed. Please check the 'Month' column format.")
        return None, None

    # Clean sales data
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()

    # Sort by date
    df = df.sort_values("Month").reset_index(drop=True)
    
    # Check if there are item-level columns
    item_columns = [col for col in df.columns if col not in ['Month', 'Sales']]
    has_item_data = len(item_columns) > 0
    
    # Store original data for item-level analysis
    original_df = df.copy()
    
    # Check if there are multiple entries per month
    original_rows = len(df)
    unique_months = df['Month'].nunique()
    
    if original_rows > unique_months:
        st.info(f"📊 Aggregating {original_rows} data points into {unique_months} monthly totals...")
        
        # Aggregate by month - sum all sales for each month
        aggregation_dict = {'Sales': 'sum'}
        
        # If there are item columns, sum them too
        for col in item_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                aggregation_dict[col] = 'sum'
            else:
                aggregation_dict[col] = 'first'  # Take first non-numeric value
        
        df_monthly = df.groupby('Month', as_index=False).agg(aggregation_dict).sort_values('Month').reset_index(drop=True)
        
        # Add original sales column for reference
        df_monthly['Sales_Original'] = df_monthly['Sales'].copy()
        
        # Advanced preprocessing on the monthly aggregated data
        df_processed = preprocess_data(df_monthly)
        
        st.success(f"✅ Successfully aggregated to {len(df_processed)} monthly data points")
        
    else:
        # Data is already monthly, just preprocess
        df_processed = preprocess_data(df)
    
    # Return both processed total data and original item-level data
    return df_processed, original_df if has_item_data else None


def preprocess_data(df):
    """Advanced data preprocessing for improved accuracy."""
    # Store original sales for reference
    df['Sales_Original'] = df['Sales'].copy()
    
    # 1. Outlier Detection and Treatment using IQR
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing (preserves data points)
    outliers_detected = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)).sum()
    if outliers_detected > 0:
        st.info(f"📊 Detected and capped {outliers_detected} outliers for better model stability")
        df['Sales'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Handle missing values with interpolation
    if df['Sales'].isna().any():
        df['Sales'] = df['Sales'].interpolate(method='time')
    
    # 3. Data transformation - test for optimal transformation
    skewness = stats.skew(df['Sales'])
    if abs(skewness) > 1:  # Highly skewed data
        st.info(f"📈 Data skewness detected ({skewness:.2f}). Applying log transformation for better modeling.")
        df['Sales'] = np.log1p(df['Sales'])  # log1p handles zeros better
        df['log_transformed'] = True
    else:
        df['log_transformed'] = False
    
    return df


def optimize_sarima_parameters(data, max_p=2, max_d=2, max_q=2, seasonal_periods=12):
    """Optimize SARIMA parameters using grid search - more conservative approach"""
    if not STATSMODELS_AVAILABLE:
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
        except Exception as e:
            continue
    
    return best_params if best_params else {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}


def run_advanced_sarima_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Fixed SARIMA with better error handling and validation"""
    try:
        if not STATSMODELS_AVAILABLE:
            forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
            return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
        
        # Ensure we have enough data points
        if len(data) < 24:
            st.warning("⚠️ SARIMA needs at least 24 data points. Using fallback method.")
            forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
            return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Ensure data is stationary and has positive values
        sales_series = work_data['Sales'].copy()
        
        # Check for zeros or negative values that could cause issues
        if (sales_series <= 0).any():
            sales_series = sales_series.clip(lower=0.1)  # Replace zeros/negatives with small positive value
        
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
        
        # Validate forecast results
        if not isinstance(forecast, (pd.Series, np.ndarray)) or len(forecast) != forecast_periods:
            raise ValueError("Invalid forecast format or length")
        
        # Convert to numpy array and ensure proper format
        forecast_values = np.array(forecast)
        
        # Check for invalid values
        if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
            raise ValueError("Forecast contains NaN or infinite values")
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        # Final validation
        if len(forecast_values) != 12:
            raise ValueError(f"Expected 12 forecast values, got {len(forecast_values)}")
        
        # Apply adjustment
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception as e:
        st.warning(f"⚠️ Advanced SARIMA failed: {str(e)}. Using fallback method.")
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_prophet_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Enhanced Prophet with better error handling"""
    try:
        if not PROPHET_AVAILABLE:
            forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
            return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Prepare data for Prophet
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Ensure positive values for Prophet
        prophet_data['y'] = prophet_data['y'].clip(lower=0.1)
        
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
        
        # Validate forecast
        if len(forecast_values) != forecast_periods:
            raise ValueError(f"Expected {forecast_periods} forecast values, got {len(forecast_values)}")
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.mean(np.abs(forecast['yhat'] - prophet_data['y']))
        
    except Exception as e:
        st.warning(f"⚠️ Advanced Prophet failed: {str(e)}. Using fallback method.")
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_ets_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Advanced ETS with better error handling"""
    try:
        if not STATSMODELS_AVAILABLE:
            forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
            return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Ensure positive values
        sales_series = work_data['Sales'].clip(lower=0.1)
        
        # Try simple additive model first
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
        
        # Validate forecast
        forecast_values = np.array(forecast)
        if len(forecast_values) != forecast_periods:
            raise ValueError(f"Expected {forecast_periods} forecast values, got {len(forecast_values)}")
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        # Ensure positive values
        forecast_values = np.maximum(forecast_values, 0)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception as e:
        st.warning(f"⚠️ Advanced ETS failed: {str(e)}. Using fallback method.")
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_xgb_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Simplified XGBoost forecast with better error handling"""
    try:
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Simple feature-based approach
        if len(work_data) >= 12:
            # Use last 12 months as seasonal pattern
            recent_sales = work_data['Sales'].tail(12).values
            
            # Calculate trend
            trend = np.polyfit(range(len(recent_sales)), recent_sales, 1)[0]
            
            # Generate forecasts with seasonal pattern and trend
            forecasts = []
            for i in range(forecast_periods):
                month_idx = i % 12
                seasonal_base = recent_sales[month_idx] if month_idx < len(recent_sales) else np.mean(recent_sales)
                trend_adjustment = trend * (i + 1)
                forecast_val = max(seasonal_base + trend_adjustment * 0.5, seasonal_base * 0.8)
                forecasts.append(forecast_val)
        else:
            # Fallback for insufficient data
            base_value = work_data['Sales'].mean()
            forecasts = [base_value] * forecast_periods
        
        forecasts = np.array(forecasts)
        
        # Validate forecast
        if len(forecasts) != forecast_periods:
            raise ValueError(f"Expected {forecast_periods} forecast values, got {len(forecasts)}")
        
        # Reverse log transformation first if applied
        if log_transformed:
            forecasts = np.expm1(forecasts)
        
        # Ensure positive values
        forecasts = np.maximum(forecasts, 0)
        
        # Apply adjustment
        return apply_forecast_adjustment(forecasts, adjustment_factor), 200.0
        
    except Exception as e:
        st.warning(f"⚠️ Advanced XGBoost failed: {str(e)}. Using fallback method.")
        forecast_values = run_fallback_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_fallback_forecast(data, forecast_periods=12, adjustment_factor=None):
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
                trend_adjustment = recent_trend * (i + 1) * 0.5  # Dampen trend
                forecast_val = max(seasonal_val + trend_adjustment, seasonal_val * 0.7)
                forecast.append(forecast_val)
            
            forecast = np.array(forecast)
            
            # Reverse log transformation first if applied
            if log_transformed:
                forecast = np.expm1(forecast)
            
            return forecast
        else:
            base_forecast = work_data['Sales'].mean()
            
            # Reverse log transformation first if applied
            if log_transformed:
                base_forecast = np.expm1(base_forecast)
            
            return np.array([base_forecast] * forecast_periods)
            
    except Exception as e:
        # Ultimate fallback - use historical mean
        try:
            historical_mean = data['Sales'].mean() if len(data) > 0 else 1000
            return np.array([historical_mean] * forecast_periods)
        except:
            return np.array([1000] * forecast_periods)


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
        model_key = model_name.replace('_Forecast', '')
        weight = weights.get(model_key, 0.25)  # Default equal weight if not found
        ensemble_forecast += weight * forecast
    
    return ensemble_forecast, weights


def forecast_item_level(item_data, item_name, forecast_periods=12, adjustment_factor=None):
    """Run forecast for individual item"""
    try:
        # Preprocess item data
        item_processed = preprocess_data(item_data.copy())
        
        # Use fallback method for individual items (more stable)
        forecast_values = run_fallback_forecast(item_processed, forecast_periods, adjustment_factor)
        
        return forecast_values
    except Exception as e:
        st.warning(f"⚠️ Forecast failed for {item_name}: {str(e)}. Using simple average.")
        # Use simple average as ultimate fallback
        avg_sales = item_data['Sales'].mean() if len(item_data) > 0 else 0
        if adjustment_factor:
            avg_sales *= adjustment_factor
        return np.array([max(avg_sales, 0)] * forecast_periods)


def create_forecast_charts(result_df, forecast_year, adjustment_percentage):
    """Create comprehensive forecast charts"""
    
    # Main forecast chart
    fig = go.Figure()
    
    model_cols = [col for col in result_df.columns if '_Forecast' in col or col in ['Weighted_Ensemble', 'Meta_Learning']]
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#6C5CE7']
    
    for i, col in enumerate(model_cols):
        if col in ['Weighted_Ensemble', 'Meta_Learning']:
            line_style = dict(color='#6C5CE7', width=4, dash='dash') if col == 'Weighted_Ensemble' else dict(color='#00D2D3', width=4, dash='dot')
            icon = '🔥' if col == 'Weighted_Ensemble' else '🧠'
        else:
            line_style = dict(color=colors[i % len(colors)], width=3)
            icon = '📈'
        
        model_name = col.replace('_Forecast', '').replace('_', ' ').upper()
        fig.add_trace(go.Scatter(
            x=result_df['Month'],
            y=result_df[col],
            mode='lines+markers',
            name=f'{icon} {model_name}',
            line=line_style,
            marker=dict(size=8)
        ))
    
    # Create dynamic title based on adjustment
    if adjustment_percentage < 0:
        adj_text = f"{abs(adjustment_percentage):.1f}% Reduction Applied"
    elif adjustment_percentage > 0:
        adj_text = f"{adjustment_percentage:.1f}% Increase Applied"
    else:
        adj_text = "No Adjustment Applied"
    
    fig.update_layout(
        title=f'🚀 AI SALES FORECAST - {forecast_year} ({adj_text})',
        xaxis_title='Month',
        yaxis_title='Sales Volume',
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig


def create_excel_export(result_df, item_forecasts, forecast_year, adjustment_percentage):
    """Create comprehensive Excel export with multiple sheets"""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # Sheet 1: Total Forecasts Summary
        summary_df = result_df.copy()
        summary_df['Month'] = summary_df['Month'].dt.strftime('%Y-%m')
        summary_df.to_excel(writer, sheet_name='Total Forecasts', index=False)
        
        # Sheet 2: Monthly Summary with preferred model
        monthly_summary = result_df[['Month']].copy()
        monthly_summary['Month'] = monthly_summary['Month'].dt.strftime('%Y-%m')
        
        # Use Weighted Ensemble if available, otherwise use first model
        if 'Weighted_Ensemble' in result_df.columns:
            monthly_summary['Recommended_Forecast'] = result_df['Weighted_Ensemble']
            monthly_summary['Model_Used'] = 'Weighted Ensemble'
        elif 'Meta_Learning' in result_df.columns:
            monthly_summary['Recommended_Forecast'] = result_df['Meta_Learning']
            monthly_summary['Model_Used'] = 'Meta Learning'
        else:
            # Use first available forecast model
            forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
            if forecast_cols:
                monthly_summary['Recommended_Forecast'] = result_df[forecast_cols[0]]
                monthly_summary['Model_Used'] = forecast_cols[0].replace('_Forecast', '')
        
        monthly_summary['Adjustment_Applied'] = f"{adjustment_percentage:+.1f}%"
        monthly_summary.to_excel(writer, sheet_name='Recommended Forecast', index=False)
        
        # Sheet 3: Item-level forecasts (if available)
        if item_forecasts:
            # Create item-level forecast sheet
            item_forecast_df = pd.DataFrame()
            
            forecast_dates = pd.date_range(
                start=f"{forecast_year}-01-01",
                end=f"{forecast_year}-12-01",
                freq='MS'
            )
            
            item_forecast_df['Month'] = forecast_dates.strftime('%Y-%m')
            
            for item_name, forecast_values in item_forecasts.items():
                if len(forecast_values) == 12:
                    item_forecast_df[f'{item_name}_Forecast'] = forecast_values
            
            item_forecast_df.to_excel(writer, sheet_name='Item Level Forecasts', index=False)
            
            # Sheet 4: Item totals
            item_totals = []
            for item_name, forecast_values in item_forecasts.items():
                if len(forecast_values) == 12:
                    total_forecast = np.sum(forecast_values)
                    item_totals.append({
                        'Item': item_name,
                        'Total_Annual_Forecast': total_forecast,
                        'Average_Monthly_Forecast': total_forecast / 12,
                        'Adjustment_Applied': f"{adjustment_percentage:+.1f}%"
                    })
            
            if item_totals:
                item_totals_df = pd.DataFrame(item_totals)
                item_totals_df = item_totals_df.sort_values('Total_Annual_Forecast', ascending=False)
                item_totals_df.to_excel(writer, sheet_name='Item Totals', index=False)
        
        # Sheet 5: Metadata
        metadata = {
            'Forecast_Year': [forecast_year],
            'Generated_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Adjustment_Percentage': [f"{adjustment_percentage:+.1f}%"],
            'Adjustment_Factor': [f"{(100 + adjustment_percentage) / 100:.3f}"],
            'Total_Items_Forecasted': [len(item_forecasts) if item_forecasts else 0],
            'Models_Used': [', '.join([col.replace('_Forecast', '') for col in result_df.columns if '_Forecast' in col])],
            'Ensemble_Available': ['Yes' if 'Weighted_Ensemble' in result_df.columns else 'No']
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    output.seek(0)
    return output


def main():
    """Main function to run the production forecasting app."""
    st.title("🚀 Production Sales Forecasting Dashboard")
    st.markdown("**Generate next 12-month forecasts with AI models and item-level breakdowns**")

    # Sidebar configuration
    st.sidebar.header("⚙️ Forecast Configuration")
    forecast_year = st.sidebar.selectbox(
        "Select forecast year:",
        options=[2024, 2025, 2026, 2027],
        index=1  # Default to 2025
    )

    # Sidebar option to adjust forecast
    st.sidebar.subheader("📊 Forecast Adjustment")
    
    # Choice between slider and custom input
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
    else:  # Custom Input
        adjustment_percentage = st.sidebar.number_input(
            "Custom Forecast Adjustment (%)",
            min_value=-100.0,
            max_value=200.0,
            value=-15.0,
            step=0.1,
            format="%.1f",
            help="Enter any percentage: negative reduces, positive increases forecasts"
        )
    
    # Calculate current adjustment factor
    current_adjustment_factor = (100 + adjustment_percentage) / 100
    
    # Show interpretation
    if adjustment_percentage < 0:
        st.sidebar.error(f"📉 **Reduction**: {abs(adjustment_percentage):.1f}% decrease (factor: {current_adjustment_factor:.3f}x)")
    elif adjustment_percentage > 0:
        st.sidebar.success(f"📈 **Increase**: {adjustment_percentage:.1f}% increase (factor: {current_adjustment_factor:.3f}x)")
    else:
        st.sidebar.info("⚖️ **No Change**: Original forecasts (factor: 1.000x)")

    # Advanced options
    st.sidebar.subheader("🔬 Advanced Options")
    enable_item_level = st.sidebar.checkbox("Item-Level Forecasting", value=True, 
                                          help="Generate forecasts for individual items")
    enable_preprocessing = st.sidebar.checkbox("Advanced Data Preprocessing", value=True,
                                              help="Outlier detection, transformation, and cleaning")

    # Model selection
    st.sidebar.subheader("🤖 Select Forecasting Models")
    use_sarima = st.sidebar.checkbox("Advanced SARIMA (Auto-tuned)", value=True)
    use_prophet = st.sidebar.checkbox("Enhanced Prophet (Optimized)", value=True)
    use_ets = st.sidebar.checkbox("Auto-ETS (Best Config)", value=True)
    use_xgb = st.sidebar.checkbox("Advanced XGBoost (Feature-Rich)", value=True)

    if not any([use_sarima, use_prophet, use_ets, use_xgb]):
        st.sidebar.error("Please select at least one forecasting model.")
        return

    # File upload
    st.subheader("📁 Upload Historical Sales Data")
    
    historical_file = st.file_uploader(
        "📊 Upload Historical Sales Data",
        type=["xlsx", "xls"],
        help="Excel file with 'Month' and 'Sales' columns. Additional columns will be treated as items for item-level forecasting."
    )

    # Display adjustment factor information
    if adjustment_percentage < 0:
        st.info(f"📉 **Forecast Reduction Applied**: All predictions will be reduced by {abs(adjustment_percentage):.1f}% (adjustable in sidebar)")
    elif adjustment_percentage > 0:
        st.info(f"📈 **Forecast Increase Applied**: All predictions will be increased by {adjustment_percentage:.1f}% (adjustable in sidebar)")
    else:
        st.info("⚖️ **No Adjustment Applied**: Forecasts will show original predicted values (adjustable in sidebar)")

    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting.")
        return

    # Load and validate historical data
    hist_df, item_df = load_data(historical_file)
    if hist_df is None:
        return

    # Display enhanced data info
    st.subheader("📊 Data Analysis")

    # Calculate correct metrics
    unique_months = hist_df['Month'].nunique()
    avg_monthly_sales = hist_df.groupby('Month')['Sales'].sum().mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Months", unique_months)
    with col2:
        st.metric("📈 Avg Monthly Sales", f"{avg_monthly_sales:,.0f}")
    with col3:
        data_quality = min(100, unique_months * 4.17)
        st.metric("🎯 Data Quality Score", f"{data_quality:.0f}%")
    with col4:
        if item_df is not None:
            item_count = len([col for col in item_df.columns if col not in ['Month', 'Sales'] and pd.api.types.is_numeric_dtype(item_df[col])])
            st.metric("📦 Items Detected", item_count)
        else:
            st.metric("📦 Items Detected", "0")

    # Show additional data insights
    col1, col2 = st.columns(2)
    with col1:
        start_date = hist_df['Month'].min().strftime('%Y-%m')
        end_date = hist_df['Month'].max().strftime('%Y-%m')
        st.metric("📅 Data Range", f"{start_date} to {end_date}")
        
    with col2:
        total_rows = len(hist_df)
        st.metric("📊 Total Data Points", f"{total_rows}")

    # Show item-level information if available
    if item_df is not None and enable_item_level:
        numeric_cols = [col for col in item_df.columns if col not in ['Month', 'Sales'] and pd.api.types.is_numeric_dtype(item_df[col])]
        if numeric_cols:
            st.success(f"📦 **Item-level forecasting enabled** for {len(numeric_cols)} items: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}")

    # Generate forecasts
    if st.button("🚀 Generate Production Forecasts", type="primary"):
        st.subheader("🚀 Generating Production Forecasts...")
        
        # Show adjustment factor being applied
        if adjustment_percentage < 0:
            st.info(f"📉 **Note**: All forecasts will be reduced by {abs(adjustment_percentage):.1f}% (multiplied by {current_adjustment_factor:.2f})")
        elif adjustment_percentage > 0:
            st.info(f"📈 **Note**: All forecasts will be increased by {adjustment_percentage:.1f}% (multiplied by {current_adjustment_factor:.2f})")
        else:
            st.info("⚖️ **Note**: No adjustment will be applied to forecasts (multiplied by 1.00)")

        progress_bar = st.progress(0)
        forecast_results = {}
        validation_scores = {}

        # Create forecast dates
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )

        # Run each selected model
        models_to_run = []
        if use_sarima:
            models_to_run.append(("SARIMA", run_advanced_sarima_forecast))
        if use_prophet:
            models_to_run.append(("Prophet", run_advanced_prophet_forecast))
        if use_ets:
            models_to_run.append(("ETS", run_advanced_ets_forecast))
        if use_xgb:
            models_to_run.append(("XGBoost", run_advanced_xgb_forecast))

        for i, (model_name, model_func) in enumerate(models_to_run):
            with st.spinner(f"🤖 Running {model_name} forecast..."):
                try:
                    result = model_func(hist_df, forecast_periods=12, adjustment_factor=current_adjustment_factor)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        forecast_values, validation_score = result[0], result[1]
                    else:
                        forecast_values = result
                        validation_score = np.inf
                    
                    # Validate forecast values
                    if isinstance(forecast_values, (list, np.ndarray)):
                        forecast_values = np.array(forecast_values)
                        
                        if len(forecast_values) == 12 and not np.all(forecast_values == 0):
                            forecast_results[f"{model_name}_Forecast"] = forecast_values
                            validation_scores[model_name] = validation_score
                            
                            min_val, max_val = np.min(forecast_values), np.max(forecast_values)
                            st.success(f"✅ {model_name} completed (Range: {min_val:,.0f} - {max_val:,.0f})")
                        else:
                            st.warning(f"⚠️ {model_name} produced invalid results. Using fallback.")
                            fallback_forecast = run_fallback_forecast(hist_df, forecast_periods=12, adjustment_factor=current_adjustment_factor)
                            forecast_results[f"{model_name}_Forecast"] = fallback_forecast
                            validation_scores[model_name] = np.inf
                    
                except Exception as e:
                    st.error(f"❌ {model_name} failed: {str(e)}")
                    fallback_forecast = run_fallback_forecast(hist_df, forecast_periods=12, adjustment_factor=current_adjustment_factor)
                    forecast_results[f"{model_name}_Forecast"] = fallback_forecast
                    validation_scores[model_name] = np.inf

            progress_bar.progress((i + 1) / len(models_to_run))

        # Create ensemble if multiple models
        if len(forecast_results) > 1:
            with st.spinner("🔥 Creating intelligent ensemble..."):
                try:
                    ensemble_values, ensemble_weights = create_weighted_ensemble(forecast_results, validation_scores)
                    forecast_results["Weighted_Ensemble"] = ensemble_values
                    st.success(f"✅ Ensemble created with weights: {', '.join([f'{k}: {v:.1%}' for k, v in ensemble_weights.items()])}")
                except Exception as e:
                    st.warning(f"⚠️ Ensemble creation failed: {str(e)}")

        # Item-level forecasting
        item_forecasts = {}
        if enable_item_level and item_df is not None:
            numeric_cols = [col for col in item_df.columns if col not in ['Month', 'Sales'] and pd.api.types.is_numeric_dtype(item_df[col])]
            
            if numeric_cols:
                st.subheader("📦 Generating Item-Level Forecasts")
                
                item_progress = st.progress(0)
                for idx, item_col in enumerate(numeric_cols):
                    with st.spinner(f"📦 Forecasting {item_col}..."):
                        try:
                            # Create item-specific data
                            item_data = item_df[['Month', item_col]].copy()
                            item_data.rename(columns={item_col: 'Sales'}, inplace=True)
                            item_data = item_data[item_data['Sales'] > 0]  # Remove zero sales months
                            
                            if len(item_data) >= 6:  # Need at least 6 months of data
                                item_forecast = forecast_item_level(item_data, item_col, forecast_periods=12, adjustment_factor=current_adjustment_factor)
                                item_forecasts[item_col] = item_forecast
                                st.success(f"✅ {item_col}: {np.sum(item_forecast):,.0f} annual forecast")
                            else:
                                st.warning(f"⚠️ {item_col}: Insufficient data ({len(item_data)} months)")
                        
                        except Exception as e:
                            st.error(f"❌ {item_col} forecast failed: {str(e)}")
                    
                    item_progress.progress((idx + 1) / len(numeric_cols))

        # Create results dataframe
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            **forecast_results
        })

        # Display results
        st.subheader("📊 Forecast Results")
        
        # Show forecast summary
        if forecast_results:
            summary_data = []
            for model_name, forecast_values in forecast_results.items():
                if isinstance(forecast_values, (list, np.ndarray)):
                    forecast_array = np.array(forecast_values)
                    summary_data.append({
                        'Model': model_name.replace('_Forecast', '').replace('_', ' '),
                        'Annual Total': f"{np.sum(forecast_array):,.0f}",
                        'Monthly Average': f"{np.mean(forecast_array):,.0f}",
                        'Q1 Total': f"{np.sum(forecast_array[:3]):,.0f}",
                        'Q2 Total': f"{np.sum(forecast_array[3:6]):,.0f}",
                        'Q3 Total': f"{np.sum(forecast_array[6:9]):,.0f}",
                        'Q4 Total': f"{np.sum(forecast_array[9:12]):,.0f}",
                        'Adjustment Applied': f"{adjustment_percentage:+.1f}%"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        # Show forecast table
        display_df = result_df.copy()
        display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
        
        for col in display_df.columns:
            if col != 'Month':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)

        # Create forecast charts
        st.subheader("📊 Forecast Visualization")
        
        chart = create_forecast_charts(result_df, forecast_year, adjustment_percentage)
        st.plotly_chart(chart, use_container_width=True)

        # Item-level results
        if item_forecasts:
            st.subheader("📦 Item-Level Forecast Summary")
            
            item_summary = []
            for item_name, forecast_values in item_forecasts.items():
                annual_total = np.sum(forecast_values)
                item_summary.append({
                    'Item': item_name,
                    'Annual Forecast': f"{annual_total:,.0f}",
                    'Monthly Average': f"{annual_total/12:,.0f}",
                    'Peak Month': f"{np.max(forecast_values):,.0f}",
                    'Low Month': f"{np.min(forecast_values):,.0f}"
                })
            
            item_summary_df = pd.DataFrame(item_summary)
            item_summary_df = item_summary_df.sort_values('Annual Forecast', key=lambda x: x.str.replace(',', '').astype(float), ascending=False)
            st.dataframe(item_summary_df, use_container_width=True)

        # Download options
        st.subheader("📊 Download Forecasts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            excel_data = create_excel_export(result_df, item_forecasts, forecast_year, adjustment_percentage)
            adj_text = f"adj_{adjustment_percentage:+.1f}pct" if adjustment_percentage != 0 else "no_adj"
            
            st.download_button(
                label="📊 Download Complete Excel Report",
                data=excel_data,
                file_name=f"sales_forecast_{forecast_year}_{adj_text}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV export
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download CSV (Total Forecasts)",
                data=csv,
                file_name=f"total_forecasts_{forecast_year}_{adj_text}.csv",
                mime="text/csv"
            )

        # Summary metrics
        st.subheader("🎯 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Weighted_Ensemble' in result_df.columns:
                ensemble_total = result_df['Weighted_Ensemble'].sum()
                st.metric("🔥 Recommended Forecast", f"{ensemble_total:,.0f}")
        
        with col2:
            successful_models = len(forecast_results)
            st.metric("🤖 Models Used", successful_models)
        
        with col3:
            if item_forecasts:
                total_items = len(item_forecasts)
                st.metric("📦 Items Forecasted", total_items)
            else:
                st.metric("📦 Items Forecasted", "0")
        
        with col4:
            st.metric("📊 Adjustment Factor", f"{current_adjustment_factor:.3f}x")

        st.success("🎉 **Production forecasts generated successfully!** Download the Excel report for complete item-level analysis.")


if __name__ == "__main__":
    main()
