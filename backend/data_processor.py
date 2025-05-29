"""
Data Processing Module for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Handles data preprocessing, feature engineering, and data transformation tasks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import io
import logging
from faker import Faker
import random

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations for financial analysis."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.fake = Faker()
        self.categories = [
            'Groceries', 'Utilities', 'Rent/Mortgage', 'Transportation', 
            'Entertainment', 'Healthcare', 'Dining Out', 'Shopping', 
            'Income', 'Other'
        ]
        
    def is_ready(self) -> bool:
        """Check if the data processor is ready."""
        return True
    
    def transactions_to_dataframe(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of transaction dictionaries to pandas DataFrame."""
        try:
            if not transactions:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Ensure required columns exist
            required_columns = ['date', 'description', 'amount', 'category']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'category':
                        df[col] = 'Other'
                    elif col == 'description':
                        df[col] = 'Transaction'
                    else:
                        raise ValueError(f"Required column '{col}' missing from transaction data")
            
            # Data type conversions
            df['date'] = pd.to_datetime(df['date'])
            df['amount'] = pd.to_numeric(df['amount'])
            
            # Add transaction_type if not present
            if 'transaction_type' not in df.columns:
                df['transaction_type'] = df.apply(
                    lambda row: 'Credit' if row['category'] == 'Income' else 'Debit', 
                    axis=1
                )
            
            # Feature engineering
            df = self._add_time_features(df)
            df = self._add_spending_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {e}")
            raise
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the DataFrame."""
        try:
            if 'date' not in df.columns:
                return df
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add time-based features
            df['day_of_week'] = df['date'].dt.day_name()
            df['day_of_week_num'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month_name()
            df['month_num'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.to_period('Q').astype(str)
            df['year'] = df['date'].dt.year
            df['day_of_month'] = df['date'].dt.day
            df['is_weekend'] = df['day_of_week_num'].isin([5, 6])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spending-related features to the DataFrame."""
        try:
            if df.empty or 'amount' not in df.columns:
                return df
            
            # Sort by date for rolling calculations
            df = df.sort_values('date')
            
            # Create debit amount column
            df['debit_amount'] = df.apply(
                lambda row: row['amount'] if row.get('transaction_type', 'Debit') == 'Debit' else 0, 
                axis=1
            )
            
            # Rolling features (using row-based windows for simplicity)
            df['rolling_mean_7_debit'] = df['debit_amount'].rolling(window=7, min_periods=1).mean()
            df['rolling_mean_30_debit'] = df['debit_amount'].rolling(window=30, min_periods=1).mean()
            
            # Lagged features
            df['lag_1_debit'] = df['debit_amount'].shift(1).fillna(0)
            df['lag_7_debit'] = df['debit_amount'].shift(7).fillna(0)
            
            # Amount statistics
            df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding spending features: {e}")
            return df
    
    def process_uploaded_file(self, file_content: bytes) -> pd.DataFrame:
        """Process uploaded CSV file content."""
        try:
            # Read CSV from bytes
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Standardize column names (lowercase, replace spaces with underscores)
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Map common column name variations
            column_mapping = {
                'transaction_date': 'date',
                'trans_date': 'date',
                'desc': 'description',
                'transaction_description': 'description',
                'amt': 'amount',
                'transaction_amount': 'amount',
                'cat': 'category',
                'transaction_category': 'category',
                'type': 'transaction_type',
                'trans_type': 'transaction_type'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns
            if 'date' not in df.columns:
                raise ValueError("Date column not found in uploaded file")
            if 'amount' not in df.columns:
                raise ValueError("Amount column not found in uploaded file")
            
            # Fill missing values
            if 'description' not in df.columns:
                df['description'] = 'Transaction'
            if 'category' not in df.columns:
                df['category'] = 'Other'
            
            # Convert to standard format
            transactions = df.to_dict('records')
            return self.transactions_to_dataframe(transactions)
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise
    
    def generate_synthetic_data(self, num_transactions: int = 1000) -> pd.DataFrame:
        """Generate synthetic transaction data for testing."""
        try:
            transaction_data = []
            
            # Generate date range (last 2 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2 * 365)
            
            for _ in range(num_transactions):
                # Generate random date
                transaction_date = self.fake.date_between(start_date=start_date, end_date=end_date)
                
                # Generate description
                description_type = random.choice(['company', 'product', 'service', 'generic_store'])
                if description_type == 'company':
                    description = f"{self.fake.company()} {random.choice(['Services', 'Payment', 'Purchase'])}"
                elif description_type == 'product':
                    description = f"Purchase of {self.fake.word()} {self.fake.word()}"
                elif description_type == 'service':
                    description = f"{self.fake.catch_phrase()} service"
                else:
                    description = f"{random.choice(['Shopping at ', 'Payment to '])}{self.fake.company_suffix()} {self.fake.company()}"
                
                # Generate amount
                amount = round(random.uniform(5.00, 1000.00), 2)
                
                # Choose category
                category = random.choice(self.categories)
                
                # Determine transaction type
                if category == 'Income':
                    transaction_type = 'Credit'
                else:
                    transaction_type = random.choices(['Debit', 'Credit'], weights=[0.9, 0.1], k=1)[0]
                
                transaction_data.append({
                    'date': transaction_date.isoformat(),
                    'description': description,
                    'amount': amount,
                    'category': category,
                    'transaction_type': transaction_type
                })
            
            # Convert to DataFrame and add features
            return self.transactions_to_dataframe(transaction_data)
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data for forecasting (placeholder implementation)."""
        try:
            # For now, generate synthetic historical data
            # In a real implementation, this would load from a database
            return self.generate_synthetic_data(1000)
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Select base features
            features = ['amount']
            
            # One-hot encode categorical features
            df_features = df.copy()
            
            # One-hot encode category and day of week
            if 'category' in df_features.columns:
                category_dummies = pd.get_dummies(df_features['category'], prefix='cat')
                df_features = pd.concat([df_features, category_dummies], axis=1)
                features.extend(category_dummies.columns.tolist())
            
            if 'day_of_week' in df_features.columns:
                dow_dummies = pd.get_dummies(df_features['day_of_week'], prefix='day')
                df_features = pd.concat([df_features, dow_dummies], axis=1)
                features.extend(dow_dummies.columns.tolist())
            
            # Return only the feature columns
            return df_features[features].fillna(0)
            
        except Exception as e:
            logger.error(f"Error preparing clustering features: {e}")
            return pd.DataFrame()
    
    def prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Select features for anomaly detection
            features = ['amount']
            
            if 'day_of_week_num' in df.columns:
                features.append('day_of_week_num')
            
            return df[features].dropna()
            
        except Exception as e:
            logger.error(f"Error preparing anomaly features: {e}")
            return pd.DataFrame()
    
    def prepare_time_series_data(self, df: pd.DataFrame, target_column: str = 'debit_amount') -> pd.Series:
        """Prepare time series data for forecasting."""
        try:
            if df.empty or 'date' not in df.columns:
                return pd.Series(dtype=float)
            
            # Ensure date is datetime and sort
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Aggregate by date
            if target_column not in df.columns:
                target_column = 'amount'
            
            daily_data = df.groupby('date')[target_column].sum()
            
            # Fill missing dates with 0
            daily_data = daily_data.asfreq('D', fill_value=0)
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.Series(dtype=float)
