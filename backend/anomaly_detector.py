"""
Anomaly Detection Module for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Isolation Forest-based anomaly detection for identifying unusual financial transactions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Isolation Forest-based anomaly detection for financial transactions."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the anomaly detector."""
        self.model_dir = model_dir
        self.isolation_forest = None
        self.scaler = None
        self.feature_columns = []
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing models
        self._load_models()
    
    def is_ready(self) -> bool:
        """Check if the anomaly detector is ready."""
        return True  # Can work with or without pre-trained models
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            isolation_forest_path = os.path.join(self.model_dir, 'isolation_forest_model.joblib')
            scaler_path = os.path.join(self.model_dir, 'scaler_anomaly.joblib')
            
            if os.path.exists(isolation_forest_path):
                self.isolation_forest = joblib.load(isolation_forest_path)
                self.is_trained = True
                logger.info("Loaded pre-trained Isolation Forest model")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_columns = list(self.scaler.feature_names_in_)
                logger.info("Loaded pre-trained anomaly detection scaler")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained anomaly detection models: {e}")
            self.is_trained = False
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.isolation_forest is not None:
                isolation_forest_path = os.path.join(self.model_dir, 'isolation_forest_model.joblib')
                joblib.dump(self.isolation_forest, isolation_forest_path)
                logger.info(f"Saved Isolation Forest model to {isolation_forest_path}")
            
            if self.scaler is not None:
                scaler_path = os.path.join(self.model_dir, 'scaler_anomaly.joblib')
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Saved anomaly detection scaler to {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error saving anomaly detection models: {e}")
    
    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 'auto') -> Dict[str, Any]:
        """Detect anomalous transactions using Isolation Forest."""
        try:
            if df.empty:
                return self._empty_anomaly_result()
            
            # Prepare features for anomaly detection
            X, feature_names = self._prepare_features(df)
            
            if X.empty:
                return self._empty_anomaly_result()
            
            # Train or use existing model
            if not self.is_trained or self.feature_columns != feature_names:
                self._train_anomaly_model(X, feature_names, contamination)
            
            # Detect anomalies
            anomaly_scores, anomaly_flags = self._predict_anomalies(X, feature_names)
            
            # Analyze anomalies
            anomaly_analysis = self._analyze_anomalies(df, anomaly_scores, anomaly_flags)
            
            # Generate risk assessment
            risk_assessment = self._assess_risk(anomaly_analysis)
            
            return {
                "anomalous_transactions": anomaly_analysis["anomalous_transactions"],
                "summary": anomaly_analysis["summary"],
                "risk_assessment": risk_assessment,
                "total_transactions": len(df),
                "anomaly_count": anomaly_analysis["summary"]["anomaly_count"]
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for anomaly detection."""
        try:
            if df.empty:
                return pd.DataFrame(), []
            
            # Start with amount as primary feature
            features = ['amount']
            df_features = df.copy()
            
            # Add day of week as numerical feature
            if 'day_of_week_num' in df_features.columns:
                features.append('day_of_week_num')
            elif 'day_of_week' in df_features.columns:
                # Convert day names to numbers
                day_mapping = {
                    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6
                }
                df_features['day_of_week_num'] = df_features['day_of_week'].map(day_mapping).fillna(0)
                features.append('day_of_week_num')
            
            # Add time-based features if available
            if 'is_weekend' in df_features.columns:
                df_features['is_weekend_int'] = df_features['is_weekend'].astype(int)
                features.append('is_weekend_int')
            
            # Add hour of day if available
            if 'hour' in df_features.columns:
                features.append('hour')
            
            # Add rolling statistics if available
            if 'rolling_mean_7_debit' in df_features.columns:
                features.append('rolling_mean_7_debit')
            
            # Select features and handle missing values
            X = df_features[features].fillna(0)
            
            return X, features
            
        except Exception as e:
            logger.error(f"Error preparing anomaly detection features: {e}")
            return pd.DataFrame(), []
    
    def _train_anomaly_model(self, X: pd.DataFrame, feature_names: List[str], contamination: float):
        """Train the Isolation Forest model."""
        try:
            # Initialize models
            self.scaler = StandardScaler()
            
            # Determine contamination parameter
            if contamination == 'auto':
                # Use a reasonable default based on data size
                if len(X) < 100:
                    contamination = 0.1  # 10% for small datasets
                else:
                    contamination = 0.05  # 5% for larger datasets
            
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.isolation_forest.fit(X_scaled)
            
            # Store feature columns
            self.feature_columns = feature_names
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            logger.info(f"Trained Isolation Forest model with contamination={contamination}")
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            raise
    
    def _predict_anomalies(self, X: pd.DataFrame, feature_names: List[str]) -> tuple:
        """Predict anomalies for the given features."""
        try:
            if not self.is_trained:
                raise ValueError("Anomaly detection model not trained")
            
            # Ensure feature alignment
            if self.feature_columns != feature_names:
                # Reindex to match training features
                X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
            else:
                X_aligned = X
            
            # Scale features
            X_scaled = self.scaler.transform(X_aligned)
            
            # Predict anomalies
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            anomaly_flags = self.isolation_forest.predict(X_scaled)
            
            return anomaly_scores, anomaly_flags
            
        except Exception as e:
            logger.error(f"Error predicting anomalies: {e}")
            raise
    
    def _analyze_anomalies(self, df: pd.DataFrame, anomaly_scores: np.ndarray, 
                          anomaly_flags: np.ndarray) -> Dict[str, Any]:
        """Analyze detected anomalies."""
        try:
            # Add anomaly information to dataframe
            df_with_anomalies = df.copy()
            df_with_anomalies['anomaly_score'] = anomaly_scores
            df_with_anomalies['is_anomaly'] = (anomaly_flags == -1)
            
            # Get anomalous transactions
            anomalous_transactions = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()
            
            # Convert to list of dictionaries for JSON serialization
            anomaly_list = []
            for _, row in anomalous_transactions.iterrows():
                anomaly_dict = {
                    'date': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                    'description': row.get('description', 'N/A'),
                    'amount': float(row['amount']),
                    'category': row.get('category', 'N/A'),
                    'anomaly_score': float(row['anomaly_score']),
                    'transaction_type': row.get('transaction_type', 'N/A')
                }
                anomaly_list.append(anomaly_dict)
            
            # Sort by anomaly score (most anomalous first)
            anomaly_list.sort(key=lambda x: x['anomaly_score'])
            
            # Summary statistics
            summary = {
                'anomaly_count': len(anomalous_transactions),
                'anomaly_percentage': (len(anomalous_transactions) / len(df)) * 100 if len(df) > 0 else 0,
                'avg_anomaly_score': float(anomaly_scores.mean()),
                'min_anomaly_score': float(anomaly_scores.min()),
                'max_anomaly_score': float(anomaly_scores.max())
            }
            
            # Category analysis of anomalies
            if not anomalous_transactions.empty and 'category' in anomalous_transactions.columns:
                category_counts = anomalous_transactions['category'].value_counts()
                summary['anomaly_categories'] = category_counts.to_dict()
            
            # Amount analysis of anomalies
            if not anomalous_transactions.empty:
                summary['anomaly_amount_stats'] = {
                    'total_anomalous_amount': float(anomalous_transactions['amount'].sum()),
                    'avg_anomalous_amount': float(anomalous_transactions['amount'].mean()),
                    'max_anomalous_amount': float(anomalous_transactions['amount'].max()),
                    'min_anomalous_amount': float(anomalous_transactions['amount'].min())
                }
            
            return {
                'anomalous_transactions': anomaly_list,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {e}")
            return {
                'anomalous_transactions': [],
                'summary': {'anomaly_count': 0, 'anomaly_percentage': 0}
            }
    
    def _assess_risk(self, anomaly_analysis: Dict[str, Any]) -> str:
        """Assess overall risk based on anomaly analysis."""
        try:
            summary = anomaly_analysis.get('summary', {})
            anomaly_count = summary.get('anomaly_count', 0)
            anomaly_percentage = summary.get('anomaly_percentage', 0)
            
            # Risk assessment logic
            if anomaly_count == 0:
                return "LOW - No anomalous transactions detected."
            elif anomaly_percentage < 2:
                return "LOW - Very few anomalous transactions detected (< 2%)."
            elif anomaly_percentage < 5:
                return "MEDIUM - Some anomalous transactions detected (2-5%). Review recommended."
            elif anomaly_percentage < 10:
                return "HIGH - Significant number of anomalous transactions detected (5-10%). Immediate review recommended."
            else:
                return "CRITICAL - High percentage of anomalous transactions detected (> 10%). Urgent review required."
                
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return "UNKNOWN - Error in risk assessment."
    
    def _empty_anomaly_result(self) -> Dict[str, Any]:
        """Return empty anomaly detection result."""
        return {
            "anomalous_transactions": [],
            "summary": {
                "anomaly_count": 0,
                "anomaly_percentage": 0,
                "avg_anomaly_score": 0,
                "min_anomaly_score": 0,
                "max_anomaly_score": 0
            },
            "risk_assessment": "NO DATA - No transaction data available for anomaly detection.",
            "total_transactions": 0,
            "anomaly_count": 0
        }
    
    def get_anomaly_threshold(self) -> float:
        """Get the current anomaly threshold."""
        try:
            if self.is_trained and hasattr(self.isolation_forest, 'offset_'):
                return float(self.isolation_forest.offset_)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting anomaly threshold: {e}")
            return 0.0
    
    def set_contamination(self, contamination: float):
        """Set contamination parameter and retrain if needed."""
        try:
            if self.isolation_forest is not None:
                self.isolation_forest.contamination = contamination
                logger.info(f"Updated contamination parameter to {contamination}")
        except Exception as e:
            logger.error(f"Error setting contamination: {e}")
