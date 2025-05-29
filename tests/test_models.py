"""
ML Models Tests for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Test suite for machine learning models and algorithms.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.spending_segmentation import SpendingSegmentation
from backend.anomaly_detector import AnomalyDetector
from backend.forecasting_engine import ForecastingEngine
from backend.recommendation_engine import RecommendationEngine

class TestSpendingSegmentation:
    """Test cases for SpendingSegmentation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.segmentation = SpendingSegmentation()
    
    def test_segment_transactions(self):
        """Test transaction segmentation."""
        # Create sample data
        df = pd.DataFrame({
            'amount': [100, 200, 150, 300, 250, 180, 220, 190, 160, 210],
            'category': ['Groceries'] * 5 + ['Transportation'] * 5,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] * 2
        })
        
        result = self.segmentation.segment_transactions(df)
        assert "segments" in result
        assert "cluster_analysis" in result
        assert "insights" in result
        assert len(result["segments"]) == len(df)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = self.segmentation.segment_transactions(df)
        assert result["segments"] == []
        assert result["cluster_analysis"] == {}
    
    def test_determine_optimal_clusters(self):
        """Test optimal cluster determination."""
        # Create sample feature matrix
        X = pd.DataFrame({
            'amount': np.random.normal(100, 50, 100),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        optimal_k = self.segmentation._determine_optimal_clusters(X)
        assert isinstance(optimal_k, int)
        assert optimal_k >= 2
        assert optimal_k <= 8

class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Create sample data with obvious anomaly
        df = pd.DataFrame({
            'amount': [100, 110, 95, 105, 5000, 98, 102],  # 5000 is anomalous
            'day_of_week_num': [0, 1, 2, 3, 4, 5, 6],
            'date': pd.date_range('2024-01-01', periods=7),
            'description': ['Normal'] * 6 + ['Large Purchase'],
            'category': ['Groceries'] * 7,
            'transaction_type': ['Debit'] * 7
        })
        
        result = self.detector.detect_anomalies(df)
        assert "anomalous_transactions" in result
        assert "summary" in result
        assert "risk_assessment" in result
        assert result["total_transactions"] == 7
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = self.detector.detect_anomalies(df)
        assert result["anomalous_transactions"] == []
        assert result["anomaly_count"] == 0
    
    def test_prepare_features(self):
        """Test feature preparation for anomaly detection."""
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday'],
            'is_weekend': [False, False, False]
        })
        
        X, features = self.detector._prepare_features(df)
        assert not X.empty
        assert 'amount' in features
        assert len(X) == 3

class TestForecastingEngine:
    """Test cases for ForecastingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.forecasting = ForecastingEngine()
    
    def test_generate_forecast(self):
        """Test forecast generation."""
        # Create sample time series data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'amount': np.random.normal(100, 20, 100),
            'transaction_type': ['Debit'] * 100
        })
        
        result = self.forecasting.generate_forecast(df, forecast_days=7)
        assert "forecast_dates" in result
        assert "forecast_values" in result
        assert "confidence_intervals" in result
        assert len(result["forecast_dates"]) == 7
        assert len(result["forecast_values"]) == 7
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = self.forecasting.generate_forecast(df)
        assert "forecast_dates" in result
        assert "forecast_values" in result
        assert len(result["forecast_values"]) == 30  # default forecast days
    
    def test_prepare_time_series(self):
        """Test time series preparation."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'amount': [100] * 10,
            'transaction_type': ['Debit'] * 10
        })
        
        series = self.forecasting._prepare_time_series(df)
        assert isinstance(series, pd.Series)
        assert len(series) >= 10  # May have filled missing dates

class TestRecommendationEngine:
    """Test cases for RecommendationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RecommendationEngine()
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Sample analysis results
        spending_analysis = {
            "total_spending": 1000.0,
            "category_breakdown": {"Groceries": 400, "Transportation": 300, "Entertainment": 300},
            "spending_trends": {"daily_average": 33.33}
        }
        
        segmentation_result = {
            "cluster_analysis": {
                "cluster_0": {"size": 10, "avg_amount": 100, "percentage": 50}
            }
        }
        
        anomaly_result = {
            "summary": {"anomaly_count": 2, "anomaly_percentage": 5},
            "anomalous_transactions": []
        }
        
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'transaction_type': ['Debit', 'Debit', 'Credit']
        })
        
        result = self.engine.generate_recommendations(
            spending_analysis, segmentation_result, anomaly_result, df
        )
        
        assert "recommendations" in result
        assert "priority_actions" in result
        assert "goal_progress" in result
        assert isinstance(result["recommendations"], list)
    
    def test_prioritize_recommendations(self):
        """Test recommendation prioritization."""
        recommendations = [
            {"priority": "high", "action_items": ["Action 1", "Action 2"]},
            {"priority": "low", "action_items": ["Action 3"]},
            {"priority": "critical", "action_items": ["Action 4", "Action 5"]}
        ]
        
        priority_actions = self.engine._prioritize_recommendations(recommendations)
        assert isinstance(priority_actions, list)
        assert len(priority_actions) > 0
    
    def test_calculate_goal_progress(self):
        """Test goal progress calculation."""
        spending_analysis = {"total_spending": 1000.0}
        df = pd.DataFrame({
            'amount': [1000, 800],
            'transaction_type': ['Credit', 'Debit']
        })
        
        progress = self.engine._calculate_goal_progress(spending_analysis, df)
        assert isinstance(progress, dict)

class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate sample data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'amount': np.random.normal(100, 30, 50),
            'category': np.random.choice(['Groceries', 'Transportation', 'Entertainment'], 50),
            'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 50),
            'transaction_type': ['Debit'] * 50
        })
        
        # Initialize all components
        analyzer = FinancialAnalyzer()
        segmentation = SpendingSegmentation()
        detector = AnomalyDetector()
        engine = RecommendationEngine()
        
        # Run analysis pipeline
        spending_analysis = analyzer.analyze_spending(df)
        segmentation_result = segmentation.segment_transactions(df)
        anomaly_result = detector.detect_anomalies(df)
        recommendations = engine.generate_recommendations(
            spending_analysis, segmentation_result, anomaly_result, df
        )
        
        # Verify all components produced results
        assert spending_analysis["total_spending"] > 0
        assert len(segmentation_result["segments"]) == len(df)
        assert "anomalous_transactions" in anomaly_result
        assert "recommendations" in recommendations

if __name__ == "__main__":
    pytest.main([__file__])
