"""
AI-Powered Personal Finance Advisor - FastAPI Backend
From Hasif's Workspace

Main FastAPI application providing financial analysis endpoints.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from .financial_analyzer import FinancialAnalyzer
from .spending_segmentation import SpendingSegmentation
from .forecasting_engine import ForecastingEngine
from .anomaly_detector import AnomalyDetector
from .recommendation_engine import RecommendationEngine
from .data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Personal Finance Advisor API",
    description="Comprehensive financial analysis and recommendation system from Hasif's Workspace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
financial_analyzer = FinancialAnalyzer()
spending_segmentation = SpendingSegmentation()
forecasting_engine = ForecastingEngine()
anomaly_detector = AnomalyDetector()
recommendation_engine = RecommendationEngine()
data_processor = DataProcessor()

# Pydantic models for request/response
class Transaction(BaseModel):
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    description: str = Field(..., description="Transaction description")
    amount: float = Field(..., description="Transaction amount")
    category: str = Field(..., description="Transaction category")
    transaction_type: str = Field(default="Debit", description="Transaction type (Debit/Credit)")

class TransactionList(BaseModel):
    transactions: List[Transaction]

class AnalysisRequest(BaseModel):
    transactions: List[Transaction]
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")

class ForecastRequest(BaseModel):
    forecast_days: int = Field(default=30, description="Number of days to forecast")
    data_source: str = Field(default="uploaded", description="Source of historical data")

class SpendingAnalysisResponse(BaseModel):
    total_spending: float
    category_breakdown: Dict[str, float]
    spending_trends: Dict[str, Any]
    insights: List[str]

class ForecastResponse(BaseModel):
    forecast_dates: List[str]
    forecast_values: List[float]
    confidence_intervals: Dict[str, List[float]]
    model_metrics: Dict[str, float]

class AnomalyResponse(BaseModel):
    anomalous_transactions: List[Dict[str, Any]]
    anomaly_summary: Dict[str, Any]
    risk_assessment: str

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    priority_actions: List[str]
    goal_progress: Dict[str, Any]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "AI-Powered Personal Finance Advisor"
    }

# Status endpoint
@app.get("/api/status")
async def get_status():
    """Get the status of all backend components."""
    try:
        status_info = {
            "server_status": "Backend is running",
            "components": {
                "financial_analyzer": financial_analyzer.is_ready(),
                "spending_segmentation": spending_segmentation.is_ready(),
                "forecasting_engine": forecasting_engine.is_ready(),
                "anomaly_detector": anomaly_detector.is_ready(),
                "recommendation_engine": recommendation_engine.is_ready(),
                "data_processor": data_processor.is_ready()
            },
            "timestamp": datetime.now().isoformat()
        }
        return status_info
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system status")

# Spending analysis endpoint
@app.post("/api/analyze/spending", response_model=SpendingAnalysisResponse)
async def analyze_spending(request: AnalysisRequest):
    """Analyze spending patterns and provide insights."""
    try:
        # Convert transactions to DataFrame
        df = data_processor.transactions_to_dataframe(request.transactions)
        
        # Perform spending analysis
        analysis_result = financial_analyzer.analyze_spending(df)
        
        return SpendingAnalysisResponse(
            total_spending=analysis_result["total_spending"],
            category_breakdown=analysis_result["category_breakdown"],
            spending_trends=analysis_result["spending_trends"],
            insights=analysis_result["insights"]
        )
    except Exception as e:
        logger.error(f"Error in spending analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing spending: {str(e)}")

# Spending segmentation endpoint
@app.post("/api/analyze/segments")
async def analyze_spending_segments(request: AnalysisRequest):
    """Perform spending behavior segmentation using K-means clustering."""
    try:
        # Convert transactions to DataFrame
        df = data_processor.transactions_to_dataframe(request.transactions)
        
        # Perform segmentation
        segmentation_result = spending_segmentation.segment_transactions(df)
        
        return {
            "segments": segmentation_result["segments"],
            "cluster_analysis": segmentation_result["cluster_analysis"],
            "segment_insights": segmentation_result["insights"]
        }
    except Exception as e:
        logger.error(f"Error in spending segmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing segmentation: {str(e)}")

# Financial forecasting endpoint
@app.post("/api/forecast/spending", response_model=ForecastResponse)
async def forecast_spending(request: ForecastRequest):
    """Generate financial forecasts using ARIMA time series models."""
    try:
        # Load historical data (this would typically come from a database)
        # For now, we'll use the most recent data or generate synthetic data
        historical_data = data_processor.load_historical_data()
        
        # Generate forecast
        forecast_result = forecasting_engine.generate_forecast(
            historical_data, 
            forecast_days=request.forecast_days
        )
        
        return ForecastResponse(
            forecast_dates=forecast_result["forecast_dates"],
            forecast_values=forecast_result["forecast_values"],
            confidence_intervals=forecast_result["confidence_intervals"],
            model_metrics=forecast_result["model_metrics"]
        )
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

# Anomaly detection endpoint
@app.post("/api/detect/anomalies", response_model=AnomalyResponse)
async def detect_anomalies(request: AnalysisRequest):
    """Detect anomalous transactions using Isolation Forest."""
    try:
        # Convert transactions to DataFrame
        df = data_processor.transactions_to_dataframe(request.transactions)
        
        # Detect anomalies
        anomaly_result = anomaly_detector.detect_anomalies(df)
        
        return AnomalyResponse(
            anomalous_transactions=anomaly_result["anomalous_transactions"],
            anomaly_summary=anomaly_result["summary"],
            risk_assessment=anomaly_result["risk_assessment"]
        )
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

# Recommendations endpoint
@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: AnalysisRequest):
    """Generate personalized financial recommendations."""
    try:
        # Convert transactions to DataFrame
        df = data_processor.transactions_to_dataframe(request.transactions)
        
        # Generate comprehensive analysis for recommendations
        spending_analysis = financial_analyzer.analyze_spending(df)
        segmentation_result = spending_segmentation.segment_transactions(df)
        anomaly_result = anomaly_detector.detect_anomalies(df)
        
        # Generate recommendations
        recommendations_result = recommendation_engine.generate_recommendations(
            spending_analysis=spending_analysis,
            segmentation_result=segmentation_result,
            anomaly_result=anomaly_result,
            transaction_data=df
        )
        
        return RecommendationResponse(
            recommendations=recommendations_result["recommendations"],
            priority_actions=recommendations_result["priority_actions"],
            goal_progress=recommendations_result["goal_progress"]
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# File upload endpoint for transaction data
@app.post("/api/upload/transactions")
async def upload_transactions(file: UploadFile = File(...)):
    """Upload transaction data from CSV file."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and process the uploaded file
        content = await file.read()
        df = data_processor.process_uploaded_file(content)
        
        # Basic validation and summary
        summary = {
            "total_transactions": len(df),
            "date_range": {
                "start": df['date'].min().isoformat() if not df.empty else None,
                "end": df['date'].max().isoformat() if not df.empty else None
            },
            "total_amount": float(df['amount'].sum()) if not df.empty else 0,
            "categories": df['category'].unique().tolist() if not df.empty else []
        }
        
        return {
            "message": "File uploaded successfully",
            "summary": summary,
            "data_preview": df.head(10).to_dict('records') if not df.empty else []
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")

# Generate synthetic data endpoint
@app.post("/api/data/generate")
async def generate_synthetic_data(num_transactions: int = 1000):
    """Generate synthetic transaction data for testing and demonstration."""
    try:
        synthetic_data = data_processor.generate_synthetic_data(num_transactions)
        
        return {
            "message": f"Generated {num_transactions} synthetic transactions",
            "data_summary": {
                "total_transactions": len(synthetic_data),
                "date_range": {
                    "start": synthetic_data['date'].min().isoformat(),
                    "end": synthetic_data['date'].max().isoformat()
                },
                "categories": synthetic_data['category'].unique().tolist()
            },
            "sample_data": synthetic_data.head(10).to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating synthetic data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
