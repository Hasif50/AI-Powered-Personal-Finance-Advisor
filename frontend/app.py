"""
AI-Powered Personal Finance Advisor - Streamlit Frontend
From Hasif's Workspace

Main Streamlit application for financial analysis and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import io

# Page configuration
st.set_page_config(
    page_title="AI-Powered Personal Finance Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .alert-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuration
API_BASE_URL = "http://localhost:8000"


class FinanceAdvisorApp:
    """Main application class for the Finance Advisor."""

    def __init__(self):
        """Initialize the application."""
        self.api_base_url = API_BASE_URL

    def run(self):
        """Run the main application."""
        # Header
        st.markdown(
            '<h1 class="main-header">💰 AI-Powered Personal Finance Advisor</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "**From Hasif's Workspace** - Your intelligent financial analysis companion"
        )

        # Sidebar
        self._render_sidebar()

        # Main content
        page = st.session_state.get("current_page", "Dashboard")

        if page == "Dashboard":
            self._render_dashboard()
        elif page == "Upload Data":
            self._render_upload_page()
        elif page == "Spending Analysis":
            self._render_spending_analysis()
        elif page == "Behavior Segmentation":
            self._render_segmentation_analysis()
        elif page == "Forecasting":
            self._render_forecasting()
        elif page == "Anomaly Detection":
            self._render_anomaly_detection()
        elif page == "Recommendations":
            self._render_recommendations()
        elif page == "Generate Sample Data":
            self._render_sample_data_generator()

    def _render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("Navigation")

        pages = [
            "Dashboard",
            "Upload Data",
            "Generate Sample Data",
            "Spending Analysis",
            "Behavior Segmentation",
            "Forecasting",
            "Anomaly Detection",
            "Recommendations",
        ]

        current_page = st.sidebar.radio("Select Page", pages)
        st.session_state["current_page"] = current_page

        # API Status
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")

        try:
            response = requests.get(f"{self.api_base_url}/api/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                st.sidebar.success("✅ Backend Connected")

                # Component status
                components = status_data.get("components", {})
                for component, status in components.items():
                    icon = "✅" if status else "❌"
                    st.sidebar.text(f"{icon} {component.replace('_', ' ').title()}")
            else:
                st.sidebar.error("❌ Backend Error")
        except:
            st.sidebar.error("❌ Backend Offline")

    def _render_dashboard(self):
        """Render the main dashboard."""
        st.header("📊 Financial Dashboard")

        # Check if we have data
        if "transaction_data" not in st.session_state:
            st.info(
                "👆 Please upload transaction data or generate sample data to get started!"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📁 Upload Your Data", use_container_width=True):
                    st.session_state["current_page"] = "Upload Data"
                    st.rerun()

            with col2:
                if st.button("🎲 Generate Sample Data", use_container_width=True):
                    st.session_state["current_page"] = "Generate Sample Data"
                    st.rerun()
            return

        # Display dashboard with data
        df = st.session_state["transaction_data"]

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_transactions = len(df)
            st.metric("Total Transactions", total_transactions)

        with col2:
            total_spending = df[df.get("transaction_type", "Debit") == "Debit"][
                "amount"
            ].sum()
            st.metric("Total Spending", f"${total_spending:,.2f}")

        with col3:
            if "transaction_type" in df.columns:
                total_income = df[df["transaction_type"] == "Credit"]["amount"].sum()
                st.metric("Total Income", f"${total_income:,.2f}")
            else:
                st.metric("Total Income", "N/A")

        with col4:
            avg_transaction = df["amount"].mean()
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Spending by category
            if "category" in df.columns:
                category_spending = (
                    df[df.get("transaction_type", "Debit") == "Debit"]
                    .groupby("category")["amount"]
                    .sum()
                )
                fig = px.pie(
                    values=category_spending.values,
                    names=category_spending.index,
                    title="Spending by Category",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Spending over time
            if "date" in df.columns:
                df_copy = df.copy()
                df_copy["date"] = pd.to_datetime(df_copy["date"])
                daily_spending = (
                    df_copy[df_copy.get("transaction_type", "Debit") == "Debit"]
                    .groupby("date")["amount"]
                    .sum()
                )

                fig = px.line(
                    x=daily_spending.index,
                    y=daily_spending.values,
                    title="Daily Spending Trend",
                )
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="Amount ($)")
                st.plotly_chart(fig, use_container_width=True)

    def _render_upload_page(self):
        """Render the data upload page."""
        st.header("📁 Upload Transaction Data")

        st.markdown("""
        Upload your transaction data in CSV format. The file should contain columns for:
        - **Date**: Transaction date
        - **Description**: Transaction description  
        - **Amount**: Transaction amount
        - **Category**: Transaction category (optional)
        - **Transaction Type**: 'Debit' or 'Credit' (optional)
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)

                st.success(
                    f"✅ File uploaded successfully! Found {len(df)} transactions."
                )

                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                # Process with backend
                if st.button("Process Data", type="primary"):
                    with st.spinner("Processing data..."):
                        # Send to backend for processing
                        files = {"file": uploaded_file.getvalue()}
                        response = requests.post(
                            f"{self.api_base_url}/api/upload/transactions",
                            files={"file": uploaded_file},
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Data processed successfully!")

                            # Store in session state
                            st.session_state["transaction_data"] = df
                            st.session_state["upload_summary"] = result.get(
                                "summary", {}
                            )

                            # Show summary
                            summary = result.get("summary", {})
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Total Transactions",
                                    summary.get("total_transactions", 0),
                                )
                            with col2:
                                st.metric(
                                    "Total Amount",
                                    f"${summary.get('total_amount', 0):,.2f}",
                                )
                            with col3:
                                categories = summary.get("categories", [])
                                st.metric("Categories", len(categories))

                            st.info("👈 Navigate to other pages to analyze your data!")
                        else:
                            st.error(
                                "❌ Error processing data. Please check your file format."
                            )

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    def _render_sample_data_generator(self):
        """Render the sample data generator page."""
        st.header("🎲 Generate Sample Data")

        st.markdown("""
        Generate synthetic transaction data for testing and demonstration purposes.
        This creates realistic financial transactions with various categories and patterns.
        """)

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            num_transactions = st.slider("Number of Transactions", 100, 2000, 1000, 100)

        with col2:
            st.info(
                f"This will generate {num_transactions} synthetic transactions over the past 2 years."
            )

        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/data/generate",
                        json={"num_transactions": num_transactions},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Sample data generated successfully!")

                        # Create DataFrame from sample data
                        sample_data = result.get("sample_data", [])
                        if sample_data:
                            df = pd.DataFrame(sample_data)

                            # Store in session state
                            st.session_state["transaction_data"] = df

                            # Show summary
                            summary = result.get("data_summary", {})
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Generated Transactions",
                                    summary.get("total_transactions", 0),
                                )
                            with col2:
                                date_range = summary.get("date_range", {})
                                if date_range:
                                    st.metric(
                                        "Date Range",
                                        f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}",
                                    )
                            with col3:
                                categories = summary.get("categories", [])
                                st.metric("Categories", len(categories))

                            # Show preview
                            st.subheader("Sample Data Preview")
                            st.dataframe(df.head(10))

                            st.info(
                                "👈 Navigate to other pages to analyze the generated data!"
                            )
                    else:
                        st.error("❌ Error generating sample data.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    def _render_spending_analysis(self):
        """Render the spending analysis page."""
        st.header("💳 Spending Analysis")

        if "transaction_data" not in st.session_state:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state["transaction_data"]

        # Convert DataFrame to API format
        transactions = df.to_dict("records")

        # Call API for analysis
        with st.spinner("Analyzing spending patterns..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/api/analyze/spending",
                    json={"transactions": transactions},
                )

                if response.status_code == 200:
                    analysis = response.json()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Spending", f"${analysis['total_spending']:,.2f}"
                        )

                    with col2:
                        category_breakdown = analysis.get("category_breakdown", {})
                        top_category = (
                            max(category_breakdown, key=category_breakdown.get)
                            if category_breakdown
                            else "N/A"
                        )
                        st.metric("Top Category", top_category)

                    with col3:
                        spending_trends = analysis.get("spending_trends", {})
                        daily_avg = spending_trends.get("daily_average", 0)
                        st.metric("Daily Average", f"${daily_avg:.2f}")

                    # Category breakdown chart
                    if category_breakdown:
                        st.subheader("Spending by Category")
                        fig = px.bar(
                            x=list(category_breakdown.keys()),
                            y=list(category_breakdown.values()),
                            title="Category Breakdown",
                        )
                        fig.update_xaxes(title="Category")
                        fig.update_yaxes(title="Amount ($)")
                        st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    insights = analysis.get("insights", [])
                    if insights:
                        st.subheader("💡 Insights")
                        for insight in insights:
                            st.info(insight)

                else:
                    st.error("❌ Error analyzing spending data.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    def _render_segmentation_analysis(self):
        """Render the behavior segmentation page."""
        st.header("🎯 Spending Behavior Segmentation")

        if "transaction_data" not in st.session_state:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state["transaction_data"]
        transactions = df.to_dict("records")

        with st.spinner("Analyzing spending behavior patterns..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/api/analyze/segments",
                    json={"transactions": transactions},
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display cluster analysis
                    cluster_analysis = result.get("cluster_analysis", {})

                    if cluster_analysis:
                        st.subheader("📊 Spending Segments")

                        # Create metrics for each cluster
                        clusters = list(cluster_analysis.keys())
                        cols = st.columns(len(clusters))

                        for i, (cluster_id, analysis) in enumerate(
                            cluster_analysis.items()
                        ):
                            with cols[i]:
                                st.markdown(
                                    f"**{cluster_id.replace('_', ' ').title()}**"
                                )
                                st.metric("Transactions", analysis["size"])
                                st.metric(
                                    "Avg Amount", f"${analysis['avg_amount']:.2f}"
                                )
                                st.metric(
                                    "Percentage", f"{analysis['percentage']:.1f}%"
                                )

                    # Insights
                    insights = result.get("insights", [])
                    if insights:
                        st.subheader("💡 Segmentation Insights")
                        for insight in insights:
                            st.info(insight)

                else:
                    st.error("❌ Error analyzing spending segments.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    def _render_forecasting(self):
        """Render the forecasting page."""
        st.header("🔮 Financial Forecasting")

        if "transaction_data" not in st.session_state:
            st.warning("⚠️ Please upload data first!")
            return

        # Forecast parameters
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)

        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating financial forecast..."):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/api/forecast/spending",
                        json={"forecast_days": forecast_days},
                    )

                    if response.status_code == 200:
                        forecast = response.json()

                        # Display forecast chart
                        st.subheader("📈 Spending Forecast")

                        # Prepare data for plotting
                        historical_dates = forecast.get("historical_dates", [])
                        historical_values = forecast.get("historical_values", [])
                        forecast_dates = forecast.get("forecast_dates", [])
                        forecast_values = forecast.get("forecast_values", [])

                        # Create plot
                        fig = go.Figure()

                        # Historical data
                        if historical_dates and historical_values:
                            fig.add_trace(
                                go.Scatter(
                                    x=historical_dates,
                                    y=historical_values,
                                    mode="lines",
                                    name="Historical",
                                    line=dict(color="blue"),
                                )
                            )

                        # Forecast data
                        if forecast_dates and forecast_values:
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=forecast_values,
                                    mode="lines",
                                    name="Forecast",
                                    line=dict(color="red", dash="dash"),
                                )
                            )

                            # Confidence intervals
                            conf_int = forecast.get("confidence_intervals", {})
                            if conf_int.get("upper") and conf_int.get("lower"):
                                fig.add_trace(
                                    go.Scatter(
                                        x=forecast_dates + forecast_dates[::-1],
                                        y=conf_int["upper"] + conf_int["lower"][::-1],
                                        fill="toself",
                                        fillcolor="rgba(255,0,0,0.1)",
                                        line=dict(color="rgba(255,255,255,0)"),
                                        name="Confidence Interval",
                                    )
                                )

                        fig.update_layout(
                            title="Daily Spending Forecast",
                            xaxis_title="Date",
                            yaxis_title="Amount ($)",
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Model metrics
                        model_metrics = forecast.get("model_metrics", {})
                        if model_metrics:
                            st.subheader("📊 Model Information")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric(
                                    "Model Type", forecast.get("model_type", "N/A")
                                )

                            with col2:
                                if "aic" in model_metrics:
                                    st.metric(
                                        "AIC Score", f"{model_metrics['aic']:.2f}"
                                    )

                    else:
                        st.error("❌ Error generating forecast.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    def _render_anomaly_detection(self):
        """Render the anomaly detection page."""
        st.header("🚨 Anomaly Detection")

        if "transaction_data" not in st.session_state:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state["transaction_data"]
        transactions = df.to_dict("records")

        with st.spinner("Detecting anomalous transactions..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/api/detect/anomalies",
                    json={"transactions": transactions},
                )

                if response.status_code == 200:
                    result = response.json()

                    # Risk assessment
                    risk_assessment = result.get("risk_assessment", "")
                    if "CRITICAL" in risk_assessment or "HIGH" in risk_assessment:
                        st.error(f"🚨 {risk_assessment}")
                    elif "MEDIUM" in risk_assessment:
                        st.warning(f"⚠️ {risk_assessment}")
                    else:
                        st.success(f"✅ {risk_assessment}")

                    # Summary metrics
                    summary = result.get("summary", {})
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Transactions", result.get("total_transactions", 0)
                        )
                    with col2:
                        st.metric("Anomalies Found", summary.get("anomaly_count", 0))
                    with col3:
                        st.metric(
                            "Anomaly Rate",
                            f"{summary.get('anomaly_percentage', 0):.1f}%",
                        )

                    # Anomalous transactions
                    anomalous_transactions = result.get("anomalous_transactions", [])
                    if anomalous_transactions:
                        st.subheader("🔍 Anomalous Transactions")

                        # Convert to DataFrame for display
                        anomaly_df = pd.DataFrame(anomalous_transactions)

                        # Sort by anomaly score (most anomalous first)
                        anomaly_df = anomaly_df.sort_values("anomaly_score")

                        # Display table
                        st.dataframe(
                            anomaly_df[
                                [
                                    "date",
                                    "description",
                                    "amount",
                                    "category",
                                    "anomaly_score",
                                ]
                            ],
                            use_container_width=True,
                        )

                        # Visualization
                        if len(anomaly_df) > 0:
                            fig = px.scatter(
                                anomaly_df,
                                x="date",
                                y="amount",
                                color="anomaly_score",
                                title="Anomalous Transactions Over Time",
                                hover_data=["description", "category"],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("✅ No anomalous transactions detected!")

                else:
                    st.error("❌ Error detecting anomalies.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    def _render_recommendations(self):
        """Render the recommendations page."""
        st.header("💡 Personalized Recommendations")

        if "transaction_data" not in st.session_state:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state["transaction_data"]
        transactions = df.to_dict("records")

        with st.spinner("Generating personalized recommendations..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/api/recommendations",
                    json={"transactions": transactions},
                )

                if response.status_code == 200:
                    result = response.json()

                    # Priority actions
                    priority_actions = result.get("priority_actions", [])
                    if priority_actions:
                        st.subheader("🎯 Priority Actions")
                        for i, action in enumerate(priority_actions[:5], 1):
                            st.markdown(f"**{i}.** {action}")

                    # Detailed recommendations
                    recommendations = result.get("recommendations", [])
                    if recommendations:
                        st.subheader("📋 Detailed Recommendations")

                        # Group by priority
                        priority_groups = {}
                        for rec in recommendations:
                            priority = rec.get("priority", "medium")
                            if priority not in priority_groups:
                                priority_groups[priority] = []
                            priority_groups[priority].append(rec)

                        # Display by priority
                        priority_order = ["critical", "high", "medium", "low"]
                        for priority in priority_order:
                            if priority in priority_groups:
                                st.markdown(f"### {priority.title()} Priority")

                                for rec in priority_groups[priority]:
                                    # Choose card style based on priority
                                    if priority == "critical":
                                        card_class = "alert-card"
                                    elif priority == "high":
                                        card_class = "recommendation-card"
                                    else:
                                        card_class = "success-card"

                                    with st.container():
                                        st.markdown(
                                            f'<div class="{card_class}">',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(
                                            f"**{rec.get('title', 'Recommendation')}**"
                                        )
                                        st.markdown(rec.get("description", ""))

                                        # Action items
                                        action_items = rec.get("action_items", [])
                                        if action_items:
                                            st.markdown("**Action Items:**")
                                            for item in action_items:
                                                st.markdown(f"• {item}")

                                        # Potential savings
                                        potential_savings = rec.get(
                                            "potential_savings", 0
                                        )
                                        if potential_savings > 0:
                                            st.markdown(
                                                f"**Potential Savings:** ${potential_savings:.2f}"
                                            )

                                        st.markdown("</div>", unsafe_allow_html=True)
                                        st.markdown("")

                    # Goal progress
                    goal_progress = result.get("goal_progress", {})
                    if goal_progress:
                        st.subheader("🎯 Goal Progress")

                        cols = st.columns(len(goal_progress))
                        for i, (goal_name, progress) in enumerate(
                            goal_progress.items()
                        ):
                            with cols[i]:
                                st.markdown(
                                    f"**{goal_name.replace('_', ' ').title()}**"
                                )

                                progress_pct = progress.get("progress_percentage", 0)
                                current = progress.get("current", 0)
                                target = progress.get("target", 0)

                                # Progress bar
                                st.progress(progress_pct / 100)
                                st.metric("Progress", f"{progress_pct:.1f}%")
                                st.metric("Current", f"${current:,.2f}")
                                st.metric("Target", f"${target:,.2f}")

                else:
                    st.error("❌ Error generating recommendations.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    app = FinanceAdvisorApp()
    app.run()


if __name__ == "__main__":
    main()
