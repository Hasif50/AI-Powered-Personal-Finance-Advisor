"""
Forecasting Engine for AI-Powered Personal Finance Advisor
From Hasif's Workspace

ARIMA time series forecasting for financial predictions and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available. Forecasting functionality will be limited.")

logger = logging.getLogger(__name__)

class ForecastingEngine:
    """ARIMA-based time series forecasting for financial data."""
    
    def __init__(self):
        """Initialize the forecasting engine."""
        self.model = None
        self.model_fit = None
        self.last_training_data = None
        self.is_trained = False
        
    def is_ready(self) -> bool:
        """Check if the forecasting engine is ready."""
        return STATSMODELS_AVAILABLE
    
    def generate_forecast(self, df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate financial forecast using ARIMA model."""
        try:
            if not STATSMODELS_AVAILABLE:
                return self._fallback_forecast(df, forecast_days)
            
            if df.empty:
                return self._empty_forecast_result(forecast_days)
            
            # Prepare time series data
            time_series = self._prepare_time_series(df)
            
            if time_series.empty or len(time_series) < 30:
                return self._simple_forecast(time_series, forecast_days)
            
            # Check stationarity and determine differencing order
            d_order = self._determine_differencing_order(time_series)
            
            # Determine ARIMA parameters
            p_order, q_order = self._determine_arima_parameters(time_series, d_order)
            
            # Train ARIMA model
            model_fit = self._train_arima_model(time_series, (p_order, d_order, q_order))
            
            if model_fit is None:
                return self._simple_forecast(time_series, forecast_days)
            
            # Generate forecast
            forecast_result = self._generate_arima_forecast(model_fit, forecast_days)
            
            # Add historical data for context
            historical_data = self._prepare_historical_context(time_series)
            forecast_result.update(historical_data)
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            # Fallback to simple forecast
            return self._simple_forecast(self._prepare_time_series(df), forecast_days)
    
    def _prepare_time_series(self, df: pd.DataFrame) -> pd.Series:
        """Prepare time series data for forecasting."""
        try:
            if df.empty or 'date' not in df.columns:
                return pd.Series(dtype=float)
            
            # Convert to datetime and sort
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.sort_values('date')
            
            # Create debit amount column if not exists
            if 'debit_amount' not in df_copy.columns:
                df_copy['debit_amount'] = df_copy.apply(
                    lambda row: row['amount'] if row.get('transaction_type', 'Debit') == 'Debit' else 0,
                    axis=1
                )
            
            # Aggregate by date
            daily_data = df_copy.groupby('date')['debit_amount'].sum()
            
            # Fill missing dates with 0
            daily_data = daily_data.asfreq('D', fill_value=0)
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error preparing time series: {e}")
            return pd.Series(dtype=float)
    
    def _determine_differencing_order(self, series: pd.Series, max_d: int = 2) -> int:
        """Determine the order of differencing needed for stationarity."""
        try:
            if not STATSMODELS_AVAILABLE or series.empty:
                return 1
            
            # Test original series
            if self._is_stationary(series):
                return 0
            
            # Test first difference
            diff_series = series.diff().dropna()
            if len(diff_series) > 1 and self._is_stationary(diff_series):
                return 1
            
            # Test second difference
            if max_d >= 2:
                diff2_series = series.diff().diff().dropna()
                if len(diff2_series) > 1 and self._is_stationary(diff2_series):
                    return 2
            
            # Default to 1 if tests are inconclusive
            return 1
            
        except Exception as e:
            logger.error(f"Error determining differencing order: {e}")
            return 1
    
    def _is_stationary(self, series: pd.Series, significance_level: float = 0.05) -> bool:
        """Test if a time series is stationary using ADF test."""
        try:
            if not STATSMODELS_AVAILABLE or len(series) < 3:
                return False
            
            result = adfuller(series.dropna())
            p_value = result[1]
            return p_value <= significance_level
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return False
    
    def _determine_arima_parameters(self, series: pd.Series, d: int) -> Tuple[int, int]:
        """Determine ARIMA p and q parameters using simple heuristics."""
        try:
            # For simplicity, use common parameter combinations
            # In a production system, you might use auto_arima or grid search
            
            # Default parameters based on series length
            if len(series) < 50:
                return 1, 1
            elif len(series) < 100:
                return 2, 1
            else:
                return 2, 2
                
        except Exception as e:
            logger.error(f"Error determining ARIMA parameters: {e}")
            return 1, 1
    
    def _train_arima_model(self, series: pd.Series, order: Tuple[int, int, int]):
        """Train ARIMA model with given parameters."""
        try:
            if not STATSMODELS_AVAILABLE or series.empty:
                return None
            
            # Split data for training (use 80% for training)
            split_point = int(len(series) * 0.8)
            train_series = series.iloc[:split_point] if split_point > 10 else series
            
            # Try to fit ARIMA model
            model = ARIMA(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()
            
            self.model_fit = model_fit
            self.is_trained = True
            
            return model_fit
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return None
    
    def _generate_arima_forecast(self, model_fit, forecast_days: int) -> Dict[str, Any]:
        """Generate forecast using trained ARIMA model."""
        try:
            # Generate forecast
            forecast_result = model_fit.get_forecast(steps=forecast_days)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Create forecast dates
            last_date = model_fit.data.dates[-1] if hasattr(model_fit.data, 'dates') else datetime.now().date()
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date).date()
            elif hasattr(last_date, 'date'):
                last_date = last_date.date()
            
            forecast_dates = [
                (last_date + timedelta(days=i+1)).isoformat() 
                for i in range(forecast_days)
            ]
            
            # Calculate model metrics
            model_metrics = self._calculate_model_metrics(model_fit)
            
            return {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values.tolist(),
                "confidence_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "model_metrics": model_metrics,
                "model_type": "ARIMA",
                "model_order": model_fit.model.order
            }
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise
    
    def _calculate_model_metrics(self, model_fit) -> Dict[str, float]:
        """Calculate model performance metrics."""
        try:
            metrics = {
                "aic": float(model_fit.aic),
                "bic": float(model_fit.bic),
                "log_likelihood": float(model_fit.llf)
            }
            
            # Add additional metrics if available
            if hasattr(model_fit, 'mse'):
                metrics["mse"] = float(model_fit.mse)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {e}")
            return {}
    
    def _prepare_historical_context(self, series: pd.Series, days: int = 90) -> Dict[str, Any]:
        """Prepare historical data for context in forecast visualization."""
        try:
            if series.empty:
                return {"historical_dates": [], "historical_values": []}
            
            # Get last N days of historical data
            historical_data = series.tail(days)
            
            historical_dates = [date.isoformat() for date in historical_data.index]
            historical_values = historical_data.tolist()
            
            return {
                "historical_dates": historical_dates,
                "historical_values": historical_values
            }
            
        except Exception as e:
            logger.error(f"Error preparing historical context: {e}")
            return {"historical_dates": [], "historical_values": []}
    
    def _simple_forecast(self, series: pd.Series, forecast_days: int) -> Dict[str, Any]:
        """Generate simple forecast using moving average when ARIMA fails."""
        try:
            if series.empty:
                return self._empty_forecast_result(forecast_days)
            
            # Use moving average for simple forecast
            window_size = min(30, len(series))
            if window_size < 1:
                avg_value = 0
            else:
                avg_value = series.tail(window_size).mean()
            
            # Add some trend if detectable
            if len(series) >= 7:
                recent_trend = series.tail(7).mean() - series.tail(14).head(7).mean()
                trend_per_day = recent_trend / 7
            else:
                trend_per_day = 0
            
            # Generate forecast
            forecast_values = []
            for i in range(forecast_days):
                forecast_value = max(0, avg_value + (trend_per_day * i))
                forecast_values.append(forecast_value)
            
            # Create forecast dates
            last_date = series.index[-1] if not series.empty else datetime.now().date()
            if hasattr(last_date, 'date'):
                last_date = last_date.date()
            
            forecast_dates = [
                (last_date + timedelta(days=i+1)).isoformat() 
                for i in range(forecast_days)
            ]
            
            # Simple confidence intervals (±20% of forecast value)
            conf_lower = [max(0, val * 0.8) for val in forecast_values]
            conf_upper = [val * 1.2 for val in forecast_values]
            
            # Historical context
            historical_context = self._prepare_historical_context(series)
            
            return {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
                "confidence_intervals": {
                    "lower": conf_lower,
                    "upper": conf_upper
                },
                "model_metrics": {
                    "method": "moving_average",
                    "window_size": window_size,
                    "trend_per_day": trend_per_day
                },
                "model_type": "Simple Moving Average",
                **historical_context
            }
            
        except Exception as e:
            logger.error(f"Error in simple forecast: {e}")
            return self._empty_forecast_result(forecast_days)
    
    def _fallback_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Fallback forecast when statsmodels is not available."""
        try:
            if df.empty:
                return self._empty_forecast_result(forecast_days)
            
            # Calculate simple average
            debit_transactions = df[df.get('transaction_type', 'Debit') == 'Debit']
            if debit_transactions.empty:
                avg_daily_spending = 0
            else:
                total_spending = debit_transactions['amount'].sum()
                date_range = (pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days
                avg_daily_spending = total_spending / max(date_range, 1)
            
            # Generate simple forecast
            forecast_values = [avg_daily_spending] * forecast_days
            
            # Create forecast dates
            last_date = pd.to_datetime(df['date']).max().date()
            forecast_dates = [
                (last_date + timedelta(days=i+1)).isoformat() 
                for i in range(forecast_days)
            ]
            
            return {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
                "confidence_intervals": {
                    "lower": [val * 0.7 for val in forecast_values],
                    "upper": [val * 1.3 for val in forecast_values]
                },
                "model_metrics": {
                    "method": "simple_average",
                    "avg_daily_spending": avg_daily_spending
                },
                "model_type": "Simple Average (Fallback)",
                "historical_dates": [],
                "historical_values": []
            }
            
        except Exception as e:
            logger.error(f"Error in fallback forecast: {e}")
            return self._empty_forecast_result(forecast_days)
    
    def _empty_forecast_result(self, forecast_days: int) -> Dict[str, Any]:
        """Return empty forecast result."""
        today = datetime.now().date()
        forecast_dates = [
            (today + timedelta(days=i+1)).isoformat() 
            for i in range(forecast_days)
        ]
        
        return {
            "forecast_dates": forecast_dates,
            "forecast_values": [0.0] * forecast_days,
            "confidence_intervals": {
                "lower": [0.0] * forecast_days,
                "upper": [0.0] * forecast_days
            },
            "model_metrics": {},
            "model_type": "No Data",
            "historical_dates": [],
            "historical_values": []
        }
