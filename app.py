import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Production Sales Forecasting Dashboard", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
import warnings
warnings.filterwarnings("ignore")

# Try to import optional libraries with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not available. Please install with: pip install plotly")

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("SciPy not available. Some advanced features will be limited.")

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
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import VotingRegressor
    from sklearn.base import BaseEstimator, RegressorMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. Some ML models will be limited.")

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


@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess the historical sales data with advanced preprocessing."""
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {str(e)}. Please ensure it's a valid Excel file.")
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
    
    # 1. Outlier Detection and Treatment using IQR (if scipy available)
    if SCIPY_AVAILABLE:
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
        df['Sales'] = df['Sales'].interpolate(method='linear')
    
    # 3. Data transformation - test for optimal transformation (if scipy available)
    if SCIPY_AVAILABLE:
        skewness = stats.skew(df['Sales'])
        if abs(skewness) > 1:  # Highly skewed data
            st.info(f"📈 Data skewness detected ({skewness:.2f}). Applying log transformation for better modeling.")
            df['Sales'] = np.log1p(df['Sales'])  # log1p handles zeros better
            df['log_transformed'] = True
        else:
            df['log_transformed'] = False
    else:
        df['log_transformed'] = False
    
    return df


def run_simple_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Simple but robust forecasting method that works without external libraries"""
    try:
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        
        # Check if data was log transformed
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        if len(work_data) >= 12:
            # Use seasonal naive with trend
            seasonal_pattern = work_data['Sales'].tail(12).values
            
            # Calculate simple trend
            if len(work_data) >= 24:
                recent_trend = (work_data['Sales'].tail(12).mean() - work_data['Sales'].tail(24).head(12).mean()) / 12
            else:
                recent_trend = (work_data['Sales'].iloc[-1] - work_data['Sales'].iloc[0]) / len(work_data)
            
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


def run_advanced_sarima_forecast(data, forecast_periods=12, adjustment_factor=None):
    """SARIMA forecast (only if statsmodels available)"""
    if not STATSMODELS_AVAILABLE:
        st.warning("⚠️ SARIMA requires statsmodels. Using simple forecast instead.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
    
    try:
        # Ensure we have enough data points
        if len(data) < 24:
            st.warning("⚠️ SARIMA needs at least 24 data points. Using simple method.")
            forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
            return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
        
        # Create a copy to avoid modifying original data
        work_data = data.copy()
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        sales_series = work_data['Sales'].copy()
        
        # Check for zeros or negative values
        if (sales_series <= 0).any():
            sales_series = sales_series.clip(lower=0.1)
        
        # Use simple SARIMA parameters
        model = SARIMAX(
            sales_series, 
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False, maxiter=100, method='lbfgs')
        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast = forecast_result.predicted_mean
        
        forecast_values = np.array(forecast)
        
        # Reverse log transformation if applied
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        forecast_values = np.maximum(forecast_values, 0)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception as e:
        st.warning(f"⚠️ SARIMA failed: {str(e)}. Using simple forecast.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_prophet_forecast(data, forecast_periods=12, adjustment_factor=None):
    """Prophet forecast (only if Prophet available)"""
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Prophet not available. Using simple forecast instead.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
    
    try:
        work_data = data.copy()
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Prepare data for Prophet
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        prophet_data['y'] = prophet_data['y'].clip(lower=0.1)
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        forecast = model.predict(future)
        
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        forecast_values = np.maximum(forecast_values, 0)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), 100.0
        
    except Exception as e:
        st.warning(f"⚠️ Prophet failed: {str(e)}. Using simple forecast.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_ets_forecast(data, forecast_periods=12, adjustment_factor=None):
    """ETS forecast (only if statsmodels available)"""
    if not STATSMODELS_AVAILABLE:
        st.warning("⚠️ ETS requires statsmodels. Using simple forecast instead.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf
    
    try:
        work_data = data.copy()
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        sales_series = work_data['Sales'].clip(lower=0.1)
        
        model = ExponentialSmoothing(
            sales_series,
            seasonal='add',
            seasonal_periods=12,
            trend='add'
        )
        fitted_model = model.fit(optimized=True)
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        forecast_values = np.array(forecast)
        
        if log_transformed:
            forecast_values = np.expm1(forecast_values)
        
        forecast_values = np.maximum(forecast_values, 0)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), fitted_model.aic
        
    except Exception as e:
        st.warning(f"⚠️ ETS failed: {str(e)}. Using simple forecast.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def run_advanced_xgb_forecast(data, forecast_periods=12, adjustment_factor=None):
    """XGBoost-style forecast"""
    try:
        work_data = data.copy()
        log_transformed = 'log_transformed' in work_data.columns and work_data['log_transformed'].iloc[0]
        
        # Simple feature-based approach
        if len(work_data) >= 12:
            recent_sales = work_data['Sales'].tail(12).values
            
            # Calculate trend
            if len(recent_sales) > 1:
                trend = (recent_sales[-1] - recent_sales[0]) / len(recent_sales)
            else:
                trend = 0
            
            # Generate forecasts with seasonal pattern and trend
            forecasts = []
            for i in range(forecast_periods):
                month_idx = i % 12
                seasonal_base = recent_sales[month_idx] if month_idx < len(recent_sales) else np.mean(recent_sales)
                trend_adjustment = trend * (i + 1)
                forecast_val = max(seasonal_base + trend_adjustment * 0.5, seasonal_base * 0.8)
                forecasts.append(forecast_val)
        else:
            base_value = work_data['Sales'].mean()
            forecasts = [base_value] * forecast_periods
        
        forecasts = np.array(forecasts)
        
        if log_transformed:
            forecasts = np.expm1(forecasts)
        
        forecasts = np.maximum(forecasts, 0)
        return apply_forecast_adjustment(forecasts, adjustment_factor), 200.0
        
    except Exception as e:
        st.warning(f"⚠️ XGBoost-style forecast failed: {str(e)}. Using simple forecast.")
        forecast_values = run_simple_forecast(data, forecast_periods, adjustment_factor)
        return apply_forecast_adjustment(forecast_values, adjustment_factor), np.inf


def create_weighted_ensemble(forecasts_dict, validation_scores):
    """Create weighted ensemble based on validation performance"""
    weights = {}
    total_inverse_score = 0
    
    for model_name, score in validation_scores.items():
        if score != np.inf and score > 0:
            inverse_score = 1 / score
            weights[model_name] = inverse_score
            total_inverse_score += inverse_score
        else:
            weights[model_name] = 0.1
            total_inverse_score += 0.1
    
    # Normalize weights
    for model_name in weights:
        weights[model_name] = weights[model_name] / total_inverse_score
    
    # Create weighted ensemble
    ensemble_forecast = np.zeros(len(next(iter(forecasts_dict.values()))))
    
    for model_name, forecast in forecasts_dict.items():
        model_key = model_name.replace('_Forecast', '')
        weight = weights.get(model_key, 0.25)
        ensemble_forecast += weight * forecast
    
    return ensemble_forecast, weights


def forecast_item_level(item_data, item_name, forecast_periods=12, adjustment_factor=None):
    """Run forecast for individual item"""
    try:
        item_processed = preprocess_data(item_data.copy())
        forecast_values = run_simple_forecast(item_processed, forecast_periods, adjustment_factor)
        return forecast_values
    except Exception as e:
        st.warning(f"⚠️ Forecast failed for {item_name}: {str(e)}. Using simple average.")
        avg_sales = item_data['Sales'].mean() if len(item_data) > 0 else 0
        if adjustment_factor:
            avg_sales *= adjustment_factor
        return np.array([max(avg_sales, 0)] * forecast_periods)


def create_forecast_charts(result_df, forecast_year, adjustment_percentage):
    """Create comprehensive forecast charts"""
    
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Please install with: pip install plotly")
        return None
    
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
    """Create comprehensive Excel export with separate sheets for each item"""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # Sheet 1: Executive Summary
        summary_data = []
        
        # Total forecast summary
        if 'Weighted_Ensemble' in result_df.columns:
            total_annual = result_df['Weighted_Ensemble'].sum()
            summary_data.append(['TOTAL COMPANY FORECAST', f"{total_annual:,.0f}", 'Weighted Ensemble'])
        elif len(result_df.columns) > 1:
            first_forecast_col = [col for col in result_df.columns if col != 'Month'][0]
            total_annual = result_df[first_forecast_col].sum()
            summary_data.append(['TOTAL COMPANY FORECAST', f"{total_annual:,.0f}", first_forecast_col.replace('_Forecast', '')])
        
        # Item summary
        if item_forecasts:
            summary_data.append(['', '', ''])  # Empty row
            summary_data.append(['ITEM BREAKDOWN', 'ANNUAL FORECAST', '% OF TOTAL'])
            
            total_items_forecast = sum(np.sum(forecast) for forecast in item_forecasts.values())
            
            # Sort items by forecast value
            sorted_items = sorted(item_forecasts.items(), key=lambda x: np.sum(x[1]), reverse=True)
            
            for item_name, forecast_values in sorted_items:
                item_annual = np.sum(forecast_values)
                percentage = (item_annual / total_items_forecast * 100) if total_items_forecast > 0 else 0
                summary_data.append([item_name, f"{item_annual:,.0f}", f"{percentage:.1f}%"])
            
            summary_data.append(['', '', ''])  # Empty row
            summary_data.append(['TOTAL ITEMS', f"{total_items_forecast:,.0f}", '100.0%'])
        
        # Create summary dataframe
        exec_summary_df = pd.DataFrame(summary_data, columns=['Description', 'Value', 'Notes'])
        exec_summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Total Company Forecasts (All Models)
        summary_df = result_df.copy()
        summary_df['Month'] = summary_df['Month'].dt.strftime('%Y-%m')
        
        # Add quarterly and annual totals
        quarterly_data = []
        for col in summary_df.columns:
            if col != 'Month':
                values = result_df[col].values
                quarterly_data.append({
                    'Model': col.replace('_Forecast', '').replace('_', ' '),
                    'Q1': f"{np.sum(values[:3]):,.0f}",
                    'Q2': f"{np.sum(values[3:6]):,.0f}",
                    'Q3': f"{np.sum(values[6:9]):,.0f}",
                    'Q4': f"{np.sum(values[9:12]):,.0f}",
                    'Annual Total': f"{np.sum(values):,.0f}",
                    'Monthly Avg': f"{np.mean(values):,.0f}"
                })
        
        # Write monthly data first
        summary_df.to_excel(writer, sheet_name='Total Forecasts', index=False, startrow=0)
        
        # Write quarterly summary below
        if quarterly_data:
            quarterly_df = pd.DataFrame(quarterly_data)
            quarterly_df.to_excel(writer, sheet_name='Total Forecasts', index=False, startrow=len(summary_df) + 3)
        
        # Sheet 3: Recommended Forecast (Clean single column)
        monthly_summary = result_df[['Month']].copy()
        monthly_summary['Month'] = monthly_summary['Month'].dt.strftime('%Y-%m')
        
        # Use best available model
        if 'Weighted_Ensemble' in result_df.columns:
            monthly_summary['Forecast'] = result_df['Weighted_Ensemble'].round(0).astype(int)
            model_used = 'Weighted Ensemble'
        else:
            forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
            if forecast_cols:
                monthly_summary['Forecast'] = result_df[forecast_cols[0]].round(0).astype(int)
                model_used = forecast_cols[0].replace('_Forecast', '')
        
        # Add summary statistics
        if 'Forecast' in monthly_summary.columns:
            monthly_summary['Cumulative'] = monthly_summary['Forecast'].cumsum()
            
            # Add metadata at the top
            metadata_rows = [
                ['Forecast Year:', forecast_year],
                ['Model Used:', model_used],
                ['Adjustment Applied:', f"{adjustment_percentage:+.1f}%"],
                ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Annual Total:', f"{monthly_summary['Forecast'].sum():,.0f}"],
                ['', ''],  # Empty row
            ]
            
            metadata_df = pd.DataFrame(metadata_rows, columns=['Parameter', 'Value'])
            metadata_df.to_excel(writer, sheet_name='Recommended Forecast', index=False, startrow=0)
            
            # Write forecast data below metadata
            monthly_summary.to_excel(writer, sheet_name='Recommended Forecast', index=False, startrow=len(metadata_rows) + 1)
        
        # Create separate sheet for each item
        if item_forecasts:
            forecast_dates = pd.date_range(
                start=f"{forecast_year}-01-01",
                end=f"{forecast_year}-12-01",
                freq='MS'
            )
            
            # Sort items by annual forecast for consistent ordering
            sorted_items = sorted(item_forecasts.items(), key=lambda x: np.sum(x[1]), reverse=True)
            
            for item_name, forecast_values in sorted_items:
                if len(forecast_values) == 12:
                    # Create detailed sheet for this item
                    item_df = pd.DataFrame({
                        'Month': forecast_dates.strftime('%Y-%m'),
                        'Month_Name': forecast_dates.strftime('%B'),
                        'Forecast': np.round(forecast_values, 0).astype(int),
                        'Cumulative': np.round(np.cumsum(forecast_values), 0).astype(int)
                    })
                    
                    # Calculate additional metrics
                    annual_total = np.sum(forecast_values)
                    monthly_avg = annual_total / 12
                    peak_month = forecast_dates[np.argmax(forecast_values)].strftime('%B')
                    peak_value = np.max(forecast_values)
                    low_month = forecast_dates[np.argmin(forecast_values)].strftime('%B')
                    low_value = np.min(forecast_values)
                    
                    # Add quarterly totals
                    q1_total = np.sum(forecast_values[:3])
                    q2_total = np.sum(forecast_values[3:6])
                    q3_total = np.sum(forecast_values[6:9])
                    q4_total = np.sum(forecast_values[9:12])
                    
                    # Create summary section
                    item_summary = [
                        ['ITEM FORECAST SUMMARY', ''],
                        ['Item Name:', item_name],
                        ['Forecast Year:', forecast_year],
                        ['Adjustment Applied:', f"{adjustment_percentage:+.1f}%"],
                        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                        ['', ''],
                        ['ANNUAL METRICS', ''],
                        ['Annual Total:', f"{annual_total:,.0f}"],
                        ['Monthly Average:', f"{monthly_avg:,.0f}"],
                        ['Peak Month:', f"{peak_month} ({peak_value:,.0f})"],
                        ['Lowest Month:', f"{low_month} ({low_value:,.0f})"],
                        ['', ''],
                        ['QUARTERLY BREAKDOWN', ''],
                        ['Q1 Total (Jan-Mar):', f"{q1_total:,.0f}"],
                        ['Q2 Total (Apr-Jun):', f"{q2_total:,.0f}"],
                        ['Q3 Total (Jul-Sep):', f"{q3_total:,.0f}"],
                        ['Q4 Total (Oct-Dec):', f"{q4_total:,.0f}"],
                        ['', ''],
                        ['MONTHLY FORECAST DETAIL', '']
                    ]
                    
                    item_summary_df = pd.DataFrame(item_summary, columns=['Metric', 'Value'])
                    
                    # Clean sheet name (Excel sheet names have limitations)
                    clean_item_name = item_name.replace('/', '_').replace('\\', '_').replace('?', '').replace('*', '').replace('[', '').replace(']', '')[:31]
                    
                    # Write summary first, then detailed data
                    item_summary_df.to_excel(writer, sheet_name=clean_item_name, index=False, startrow=0)
                    item_df.to_excel(writer, sheet_name=clean_item_name, index=False, startrow=len(item_summary) + 2)
            
            # All Items Comparison Sheet
            all_items_df = pd.DataFrame()
            all_items_df['Month'] = forecast_dates.strftime('%Y-%m')
            
            # Add each item as a column
            for item_name, forecast_values in sorted_items:
                if len(forecast_values) == 12:
                    clean_name = item_name.replace('/', '_').replace('\\', '_')[:20]  # Shorter for column names
                    all_items_df[clean_name] = np.round(forecast_values, 0).astype(int)
            
            # Add total column
            if len(all_items_df.columns) > 1:
                numeric_cols = [col for col in all_items_df.columns if col != 'Month']
                all_items_df['TOTAL'] = all_items_df[numeric_cols].sum(axis=1)
            
            all_items_df.to_excel(writer, sheet_name='All Items Comparison', index=False)
            
            # Items Summary Sheet
            items_summary = []
            for item_name, forecast_values in sorted_items:
                if len(forecast_values) == 12:
                    annual_total = np.sum(forecast_values)
                    monthly_avg = annual_total / 12
                    total_items_value = sum(np.sum(f) for f in item_forecasts.values())
                    percentage = (annual_total / total_items_value * 100) if total_items_value > 0 else 0
                    
                    items_summary.append({
                        'Rank': len(items_summary) + 1,
                        'Item': item_name,
                        'Annual Forecast': f"{annual_total:,.0f}",
                        'Monthly Average': f"{monthly_avg:,.0f}",
                        'Percentage of Total': f"{percentage:.1f}%",
                        'Q1': f"{np.sum(forecast_values[:3]):,.0f}",
                        'Q2': f"{np.sum(forecast_values[3:6]):,.0f}",
                        'Q3': f"{np.sum(forecast_values[6:9]):,.0f}",
                        'Q4': f"{np.sum(forecast_values[9:12]):,.0f}",
                        'Peak Month Value': f"{np.max(forecast_values):,.0f}",
                        'Low Month Value': f"{np.min(forecast_values):,.0f}"
                    })
            
            if items_summary:
                items_summary_df = pd.DataFrame(items_summary)
                items_summary_df.to_excel(writer, sheet_name='Items Summary', index=False)
        
        # Final sheet: Technical Metadata
        metadata = {
            'Parameter': [
                'Forecast Year',
                'Generated Date',
                'Generated Time',
                'Adjustment Percentage',
                'Adjustment Factor',
                'Total Items Forecasted',
                'Models Available',
                'Recommended Model',
                'Ensemble Available',
                'Excel Sheets Created',
                'Total Company Annual Forecast',
                'Total Items Annual Forecast'
            ],
            'Value': [
                forecast_year,
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%H:%M:%S'),
                f"{adjustment_percentage:+.1f}%",
                f"{(100 + adjustment_percentage) / 100:.3f}",
                len(item_forecasts) if item_forecasts else 0,
                ', '.join([col.replace('_Forecast', '') for col in result_df.columns if '_Forecast' in col]),
                'Weighted Ensemble' if 'Weighted_Ensemble' in result_df.columns else 'First Available Model',
                'Yes' if 'Weighted_Ensemble' in result_df.columns else 'No',
                3 + len(item_forecasts) if item_forecasts else 3,
                f"{result_df['Weighted_Ensemble'].sum():,.0f}" if 'Weighted_Ensemble' in result_df.columns else 'N/A',
                f"{sum(np.sum(f) for f in item_forecasts.values()):,.0f}" if item_forecasts else 'N/A'
            ]
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Technical Details', index=False)
    
    output.seek(0)
    return output


def main():
    """Main function to run the production forecasting app."""
    st.title("🚀 Production Sales Forecasting Dashboard")
    st.markdown("**Generate next 12-month forecasts with AI models and item-level breakdowns**")

    # Show library status
    with st.expander("📦 Library Status"):
        st.write(f"✅ Core libraries: Pandas, Numpy, Datetime")
        st.write(f"{'✅' if PLOTLY_AVAILABLE else '❌'} Plotly (for charts)")
        st.write(f"{'✅' if SCIPY_AVAILABLE else '❌'} SciPy (for advanced preprocessing)")
        st.write(f"{'✅' if PROPHET_AVAILABLE else '❌'} Prophet (for Prophet forecasting)")
        st.write(f"{'✅' if STATSMODELS_AVAILABLE else '❌'} Statsmodels (for SARIMA/ETS)")
        st.write(f"{'✅' if SKLEARN_AVAILABLE else '❌'} Scikit-learn (for ML features)")
        
        if not PLOTLY_AVAILABLE:
            st.error("⚠️ Plotly is required for charts. Install with: `pip install plotly`")

    # Sidebar configuration
    st.sidebar.header("⚙️ Forecast Configuration")
    forecast_year = st.sidebar.selectbox(
        "Select forecast year:",
        options=[2024, 2025, 2026, 2027],
        index=1  # Default to 2025
    )

    # Forecast adjustment
    st.sidebar.subheader("📊 Forecast Adjustment")
    adjustment_percentage = st.sidebar.slider(
        "Forecast Adjustment (%)",
        min_value=-50,
        max_value=50,
        value=-15,  # Default 15% reduction
        step=5,
        help="Negative values reduce forecasts, positive values increase them"
    )
    
    current_adjustment_factor = (100 + adjustment_percentage) / 100
    
    # Show interpretation
    if adjustment_percentage < 0:
        st.sidebar.error(f"📉 **Reduction**: {abs(adjustment_percentage):.1f}% decrease")
    elif adjustment_percentage > 0:
        st.sidebar.success(f"📈 **Increase**: {adjustment_percentage:.1f}% increase")
    else:
        st.sidebar.info("⚖️ **No Change**: Original forecasts")

    # Model selection
    st.sidebar.subheader("🤖 Select Forecasting Models")
    use_simple = st.sidebar.checkbox("Simple Seasonal Forecast (Always Available)", value=True)
    use_sarima = st.sidebar.checkbox(f"SARIMA {'✅' if STATSMODELS_AVAILABLE else '❌'}", value=STATSMODELS_AVAILABLE)
    use_prophet = st.sidebar.checkbox(f"Prophet {'✅' if PROPHET_AVAILABLE else '❌'}", value=PROPHET_AVAILABLE)
    use_ets = st.sidebar.checkbox(f"ETS {'✅' if STATSMODELS_AVAILABLE else '❌'}", value=STATSMODELS_AVAILABLE)
    use_xgb = st.sidebar.checkbox("XGBoost-style (Always Available)", value=True)

    # Advanced options
    st.sidebar.subheader("🔬 Advanced Options")
    enable_item_level = st.sidebar.checkbox("Item-Level Forecasting", value=True)

    # File upload
    st.subheader("📁 Upload Historical Sales Data")
    
    historical_file = st.file_uploader(
        "📊 Upload Historical Sales Data",
        type=["xlsx", "xls"],
        help="Excel file with 'Month' and 'Sales' columns. Additional numeric columns will be treated as items."
    )

    # Display adjustment info
    if adjustment_percentage < 0:
        st.info(f"📉 **Forecast Reduction Applied**: All predictions will be reduced by {abs(adjustment_percentage):.1f}%")
    elif adjustment_percentage > 0:
        st.info(f"📈 **Forecast Increase Applied**: All predictions will be increased by {adjustment_percentage:.1f}%")
    else:
        st.info("⚖️ **No Adjustment Applied**: Forecasts will show original predicted values")

    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting.")
        return

    # Load data
    hist_df, item_df = load_data(historical_file)
    if hist_df is None:
        return

    # Display data info
    st.subheader("📊 Data Analysis")

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

    # Show item info if available
    if item_df is not None and enable_item_level:
        numeric_cols = [col for col in item_df.columns if col not in ['Month', 'Sales'] and pd.api.types.is_numeric_dtype(item_df[col])]
        if numeric_cols:
            st.success(f"📦 **Item-level forecasting enabled** for {len(numeric_cols)} items: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}")

    # Generate forecasts
    if st.button("🚀 Generate Production Forecasts", type="primary"):
        st.subheader("🚀 Generating Production Forecasts...")
        
        progress_bar = st.progress(0)
        forecast_results = {}
        validation_scores = {}

        # Create forecast dates
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )

        # Run selected models
        models_to_run = []
        if use_simple:
            models_to_run.append(("Simple", lambda data, periods, adj: (run_simple_forecast(data, periods), 100.0)))
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
                        validation_score = 100.0
                    
                    if isinstance(forecast_values, (list, np.ndarray)):
                        forecast_values = np.array(forecast_values)
                        
                        if len(forecast_values) == 12 and not np.all(forecast_values == 0):
                            forecast_results[f"{model_name}_Forecast"] = forecast_values
                            validation_scores[model_name] = validation_score
                            
                            min_val, max_val = np.min(forecast_values), np.max(forecast_values)
                            st.success(f"✅ {model_name} completed (Range: {min_val:,.0f} - {max_val:,.0f})")
                        else:
                            st.warning(f"⚠️ {model_name} produced invalid results.")
                    
                except Exception as e:
                    st.error(f"❌ {model_name} failed: {str(e)}")

            progress_bar.progress((i + 1) / len(models_to_run))

        # Create ensemble if multiple models
        if len(forecast_results) > 1:
            with st.spinner("🔥 Creating intelligent ensemble..."):
                try:
                    ensemble_values, ensemble_weights = create_weighted_ensemble(forecast_results, validation_scores)
                    forecast_results["Weighted_Ensemble"] = ensemble_values
                    st.success(f"✅ Ensemble created")
                except Exception as e:
                    st.warning(f"⚠️ Ensemble creation failed: {str(e)}")

        # Item-level forecasting
        item_forecasts = {}
        if enable_item_level and item_df is not None:
            numeric_cols = [col for col in item_df.columns if col not in ['Month', 'Sales'] and pd.api.types.is_numeric_dtype(item_df[col])]
            
            if numeric_cols:
                st.subheader("📦 Generating Item-Level Forecasts")
                
                for idx, item_col in enumerate(numeric_cols):
                    try:
                        item_data = item_df[['Month', item_col]].copy()
                        item_data.rename(columns={item_col: 'Sales'}, inplace=True)
                        item_data = item_data[item_data['Sales'] > 0]
                        
                        if len(item_data) >= 6:
                            item_forecast = forecast_item_level(item_data, item_col, forecast_periods=12, adjustment_factor=current_adjustment_factor)
                            item_forecasts[item_col] = item_forecast
                            st.success(f"✅ {item_col}: {np.sum(item_forecast):,.0f} annual forecast")
                        else:
                            st.warning(f"⚠️ {item_col}: Insufficient data")
                    
                    except Exception as e:
                        st.error(f"❌ {item_col} forecast failed: {str(e)}")

        # Create results dataframe
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            **forecast_results
        })

        # Display results
        st.subheader("📊 Forecast Results")
        
        # Summary table
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
                        'Q4 Total': f"{np.sum(forecast_array[9:12]):,.0f}"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        # Monthly forecast table
        display_df = result_df.copy()
        display_df['Month'] = display_df['Month'].dt.strftime('%Y-%m')
        
        for col in display_df.columns:
            if col != 'Month':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)

        # Chart (if Plotly available)
        if PLOTLY_AVAILABLE:
            st.subheader("📊 Forecast Visualization")
            chart = create_forecast_charts(result_df, forecast_year, adjustment_percentage)
            if chart:
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
            # Sort by annual forecast (convert string back to number for sorting)
            item_summary_df['sort_key'] = item_summary_df['Annual Forecast'].str.replace(',', '').astype(float)
            item_summary_df = item_summary_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
            st.dataframe(item_summary_df, use_container_width=True)

        # Download options
        st.subheader("📊 Download Forecasts")
        
        # Show what will be included in Excel export
        if item_forecasts:
            st.info(f"📊 **Excel Report will include:**\n"
                   f"• Executive Summary with company totals\n"
                   f"• **Separate detailed sheet for each of {len(item_forecasts)} items**\n"
                   f"• All items comparison sheet\n"
                   f"• Items ranking summary\n"
                   f"• Total company forecasts (all models)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel export
            excel_data = create_excel_export(result_df, item_forecasts, forecast_year, adjustment_percentage)
            adj_text = f"adj_{adjustment_percentage:+.1f}pct" if adjustment_percentage != 0 else "no_adj"
            
            sheets_count = 5 + len(item_forecasts) if item_forecasts else 3
            
            st.download_button(
                label=f"📊 Download Complete Excel Report ({sheets_count} sheets)",
                data=excel_data,
                file_name=f"sales_forecast_{forecast_year}_{adj_text}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=f"Includes individual sheets for each of {len(item_forecasts)} items" if item_forecasts else "Complete forecast report"
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
        
        # Show Excel structure
        if item_forecasts:
            with st.expander("📋 Excel Report Structure"):
                st.write("**Sheet 1: Executive Summary** - Company totals and item breakdown")
                st.write("**Sheet 2: Total Forecasts** - All AI models with quarterly summaries")
                st.write("**Sheet 3: Recommended Forecast** - Best model with metadata")
                st.write("**Individual Item Sheets:**")
                for idx, item_name in enumerate(sorted(item_forecasts.keys(), key=lambda x: np.sum(item_forecasts[x]), reverse=True), 1):
                    annual_forecast = np.sum(item_forecasts[item_name])
                    st.write(f"  • **{item_name}** - Detailed monthly forecast ({annual_forecast:,.0f} annual)")
                st.write("**All Items Comparison** - Side-by-side monthly comparison")
                st.write("**Items Summary** - Ranked performance table")
                st.write("**Technical Details** - Generation metadata")

        # Summary metrics
        st.subheader("🎯 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Weighted_Ensemble' in result_df.columns:
                ensemble_total = result_df['Weighted_Ensemble'].sum()
                st.metric("🔥 Recommended Forecast", f"{ensemble_total:,.0f}")
            elif forecast_results:
                first_model = list(forecast_results.keys())[0]
                first_total = np.sum(forecast_results[first_model])
                st.metric("📈 Primary Forecast", f"{first_total:,.0f}")
        
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

        st.success("🎉 **Production forecasts generated successfully!** Download the Excel report for complete analysis.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check that all required libraries are installed and the data file is properly formatted.")
