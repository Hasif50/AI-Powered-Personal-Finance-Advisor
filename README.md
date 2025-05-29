# AI-Powered Personal Finance Advisor

A comprehensive AI-driven financial analysis tool that provides personalized insights, spending behavior analysis, and financial forecasting. This project leverages machine learning techniques to help users understand their financial patterns and make informed decisions.

This project demonstrates expertise in Python, machine learning implementation, financial data analysis, and solving personal finance challenges through intelligent automation.

## ✨ Features

- **Automated Transaction Analysis:** Processes and categorizes financial transaction data with intelligent pattern recognition.
- **Spending Behavior Segmentation:** Uses K-means clustering to identify distinct spending patterns and user behaviors.
- **Financial Forecasting:** Implements ARIMA time series models to predict future spending trends and cash flow.
- **Anomaly Detection:** Employs Isolation Forest to identify unusual transactions and potential fraud.
- **Personalized Recommendations:** Generates actionable financial advice based on spending patterns and user goals.
- **Interactive Dashboard:** Streamlit-powered frontend for intuitive data visualization and user interaction.
- **FastAPI Backend:** Robust REST API for financial analysis and machine learning model serving.
- **Synthetic Data Generation:** Creates realistic financial datasets for testing and demonstration.
- **Dockerized Deployment:** Complete containerization for easy deployment and scalability.

## 🛠️ Tech Stack

### **Backend:**
- Python 3.10+
- FastAPI: For building the REST API
- Uvicorn: ASGI server for FastAPI
- Pandas & NumPy: For data manipulation and analysis
- Scikit-learn: For machine learning models (K-means, Isolation Forest)
- Statsmodels: For ARIMA time series forecasting
- NLTK: For natural language processing of transaction descriptions
- Joblib: For model serialization and persistence

### **Frontend:**
- Streamlit: For creating the interactive web application
- Plotly: For advanced data visualizations
- Requests: For API communication

### **Data & ML:**
- Faker: For synthetic data generation
- TensorFlow: For advanced ML capabilities
- PyTorch: For deep learning models (future enhancements)

### **Containerization:**
- Docker & Docker Compose

## 📂 Project Structure

```
├── backend/                # FastAPI application and ML modules
│   ├── __init__.py
│   ├── main.py             # FastAPI app definition and endpoints
│   ├── financial_analyzer.py # Core financial analysis logic
│   ├── spending_segmentation.py # K-means clustering for spending patterns
│   ├── forecasting_engine.py # ARIMA time series forecasting
│   ├── anomaly_detector.py # Isolation Forest anomaly detection
│   ├── recommendation_engine.py # Personalized recommendation logic
│   ├── data_processor.py   # Data preprocessing and feature engineering
│   ├── requirements.txt    # Backend Python dependencies
│   └── Dockerfile          # Dockerfile for backend
├── frontend/               # Streamlit application
│   ├── app.py              # Main Streamlit application
│   ├── components/         # Reusable UI components
│   ├── requirements.txt    # Frontend Python dependencies
│   └── Dockerfile          # Dockerfile for frontend
├── data/                   # Data storage and samples
│   └── .gitkeep           # Keep directory in git
├── models/                 # Trained ML models storage
│   └── .gitkeep           # Keep directory in git
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_backend.py     # Backend API tests
│   └── test_models.py      # ML model tests
├── scripts/                # Utility scripts
│   └── generate_synthetic_data.py # Data generation script
├── notebooks/              # Jupyter notebooks for development
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Model_Development.ipynb
├── .gitignore              # Git ignore file
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Root level dependencies
├── Dockerfile              # Main Dockerfile
└── README.md               # This file
```

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.10 or later
- `pip` (Python package installer)
- Docker and Docker Compose (recommended for easiest setup)

### Option 1: Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd AI-Powered-Personal-Finance-Advisor
   ```

2. Build and start the services with Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Option 2: Local Development

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd AI-Powered-Personal-Finance-Advisor
   ```

2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   # On Windows:
   # venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

3. Set up the frontend:
   ```bash
   cd frontend
   python -m venv venv
   # On Windows:
   # venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

4. Generate synthetic data:
   ```bash
   python scripts/generate_synthetic_data.py
   ```

5. Start the backend server:
   ```bash
   # From the project root
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. In a new terminal, start the frontend:
   ```bash
   # From the project root
   streamlit run frontend/app.py
   ```

7. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📖 User Guide

### Financial Analysis Dashboard

1. Access the application at http://localhost:8501
2. Upload your transaction data (CSV format) or use the synthetic data generator
3. Explore different analysis modules:
   - **Spending Overview:** View transaction summaries and category breakdowns
   - **Behavior Analysis:** Discover your spending patterns through clustering
   - **Forecasting:** See predictions for future spending trends
   - **Anomaly Detection:** Identify unusual transactions
   - **Recommendations:** Get personalized financial advice

### API Usage

The backend provides comprehensive REST API endpoints for financial analysis:

#### Endpoint 1: `/api/analyze/spending` (Spending Analysis)
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/analyze/spending`
- **Request Body (JSON):**
  ```json
  {
    "transactions": [
      {
        "date": "2024-01-15",
        "description": "Grocery Store Purchase",
        "amount": 85.50,
        "category": "Groceries"
      }
    ]
  }
  ```

#### Endpoint 2: `/api/forecast/spending` (Financial Forecasting)
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/forecast/spending`
- **Request Body (JSON):**
  ```json
  {
    "historical_data": "path/to/data.csv",
    "forecast_days": 30
  }
  ```

#### Endpoint 3: `/api/detect/anomalies` (Anomaly Detection)
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/detect/anomalies`

## 🔌 Key Features Explained

### Spending Behavior Segmentation
Uses K-means clustering to identify distinct spending patterns, helping users understand their financial behavior and compare with similar user segments.

### Financial Forecasting
Implements ARIMA time series models to predict future spending trends, enabling proactive financial planning and budget management.

### Anomaly Detection
Employs Isolation Forest algorithm to identify unusual transactions that may indicate fraud, errors, or significant changes in spending behavior.

### Personalized Recommendations
Generates actionable financial advice based on spending patterns, forecasts, and user-defined financial goals.

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_backend.py
python -m pytest tests/test_models.py
```

## 📊 Data Privacy and Security

This application prioritizes data privacy and security:
- All financial data processing is performed locally
- No sensitive information is transmitted to external services
- Synthetic data generation for safe testing and demonstration
- Comprehensive data anonymization techniques
- Secure API endpoints with proper validation

## 🚀 Future Enhancements

- Integration with bank APIs for real-time data
- Advanced deep learning models for pattern recognition
- Mobile application development
- Multi-currency support
- Investment portfolio analysis
- Goal-based financial planning tools

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Developed from Hasif's Workspace - Demonstrating expertise in AI, machine learning, and financial technology solutions.*
