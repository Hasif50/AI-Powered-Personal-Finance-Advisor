"""
Backend API Tests for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Test suite for FastAPI backend endpoints and functionality.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.main import app
from backend.data_processor import DataProcessor
from backend.financial_analyzer import FinancialAnalyzer

client = TestClient(app)

class TestAPI:
    """Test cases for API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_status_endpoint(self):
        """Test status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "server_status" in data
        assert "components" in data
        assert data["server_status"] == "Backend is running"
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        response = client.post("/api/data/generate", json={"num_transactions": 100})
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "data_summary" in data
        assert data["data_summary"]["total_transactions"] == 100
    
    def test_spending_analysis(self):
        """Test spending analysis endpoint."""
        # Sample transaction data
        transactions = [
            {
                "date": "2024-01-15",
                "description": "Grocery Store",
                "amount": 85.50,
                "category": "Groceries",
                "transaction_type": "Debit"
            },
            {
                "date": "2024-01-16", 
                "description": "Gas Station",
                "amount": 45.00,
                "category": "Transportation",
                "transaction_type": "Debit"
            }
        ]
        
        response = client.post("/api/analyze/spending", json={"transactions": transactions})
        assert response.status_code == 200
        data = response.json()
        assert "total_spending" in data
        assert "category_breakdown" in data
        assert "insights" in data
        assert data["total_spending"] == 130.50
    
    def test_spending_segmentation(self):
        """Test spending segmentation endpoint."""
        transactions = [
            {
                "date": "2024-01-15",
                "description": "Grocery Store",
                "amount": 85.50,
                "category": "Groceries",
                "transaction_type": "Debit"
            }
        ]
        
        response = client.post("/api/analyze/segments", json={"transactions": transactions})
        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert "cluster_analysis" in data
    
    def test_anomaly_detection(self):
        """Test anomaly detection endpoint."""
        transactions = [
            {
                "date": "2024-01-15",
                "description": "Normal Purchase",
                "amount": 50.00,
                "category": "Groceries",
                "transaction_type": "Debit"
            },
            {
                "date": "2024-01-16",
                "description": "Large Purchase",
                "amount": 5000.00,
                "category": "Shopping",
                "transaction_type": "Debit"
            }
        ]
        
        response = client.post("/api/detect/anomalies", json={"transactions": transactions})
        assert response.status_code == 200
        data = response.json()
        assert "anomalous_transactions" in data
        assert "summary" in data
        assert "risk_assessment" in data
    
    def test_recommendations(self):
        """Test recommendations endpoint."""
        transactions = [
            {
                "date": "2024-01-15",
                "description": "Grocery Store",
                "amount": 85.50,
                "category": "Groceries",
                "transaction_type": "Debit"
            }
        ]
        
        response = client.post("/api/recommendations", json={"transactions": transactions})
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "priority_actions" in data
        assert "goal_progress" in data

class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
    
    def test_transactions_to_dataframe(self):
        """Test transaction conversion to DataFrame."""
        transactions = [
            {
                "date": "2024-01-15",
                "description": "Test Transaction",
                "amount": 100.0,
                "category": "Test"
            }
        ]
        
        df = self.processor.transactions_to_dataframe(transactions)
        assert not df.empty
        assert len(df) == 1
        assert "date" in df.columns
        assert "amount" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = self.processor.generate_synthetic_data(100)
        assert not df.empty
        assert len(df) == 100
        assert "date" in df.columns
        assert "amount" in df.columns
        assert "category" in df.columns
    
    def test_prepare_clustering_features(self):
        """Test clustering feature preparation."""
        # Create sample data
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'category': ['A', 'B', 'A'],
            'day_of_week': ['Monday', 'Tuesday', 'Monday']
        })
        
        features_df = self.processor.prepare_clustering_features(df)
        assert not features_df.empty
        assert 'amount' in features_df.columns

class TestFinancialAnalyzer:
    """Test cases for FinancialAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FinancialAnalyzer()
    
    def test_analyze_spending(self):
        """Test spending analysis."""
        # Create sample data
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'amount': [100, 200, 150, 300, 250, 180, 220, 190, 160, 210],
            'category': ['Groceries'] * 5 + ['Transportation'] * 5,
            'transaction_type': ['Debit'] * 10
        })
        
        result = self.analyzer.analyze_spending(df)
        assert "total_spending" in result
        assert "category_breakdown" in result
        assert "insights" in result
        assert result["total_spending"] == 1960.0
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.analyze_spending(df)
        assert result["total_spending"] == 0.0
        assert result["category_breakdown"] == {}
    
    def test_cash_flow_analysis(self):
        """Test cash flow analysis."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'amount': [1000, 200, 150, 300],
            'transaction_type': ['Credit', 'Debit', 'Debit', 'Debit']
        })
        
        result = self.analyzer.analyze_cash_flow(df)
        assert "total_income" in result
        assert "total_expenses" in result
        assert "net_cash_flow" in result
        assert result["total_income"] == 1000.0
        assert result["total_expenses"] == 650.0

if __name__ == "__main__":
    pytest.main([__file__])
