"""
Financial Analysis Module for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Core financial analysis functionality including spending analysis and insights generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Core financial analysis engine for spending patterns and insights."""
    
    def __init__(self):
        """Initialize the financial analyzer."""
        self.insights_cache = {}
        
    def is_ready(self) -> bool:
        """Check if the financial analyzer is ready."""
        return True
    
    def analyze_spending(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive spending analysis."""
        try:
            if df.empty:
                return self._empty_analysis_result()
            
            # Basic spending metrics
            total_spending = self._calculate_total_spending(df)
            category_breakdown = self._analyze_category_breakdown(df)
            spending_trends = self._analyze_spending_trends(df)
            insights = self._generate_spending_insights(df, total_spending, category_breakdown)
            
            return {
                "total_spending": total_spending,
                "category_breakdown": category_breakdown,
                "spending_trends": spending_trends,
                "insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in spending analysis: {e}")
            raise
    
    def _calculate_total_spending(self, df: pd.DataFrame) -> float:
        """Calculate total spending (debit transactions only)."""
        try:
            debit_transactions = df[df.get('transaction_type', 'Debit') == 'Debit']
            return float(debit_transactions['amount'].sum()) if not debit_transactions.empty else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating total spending: {e}")
            return 0.0
    
    def _analyze_category_breakdown(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze spending breakdown by category."""
        try:
            if 'category' not in df.columns:
                return {}
            
            # Filter debit transactions only
            debit_transactions = df[df.get('transaction_type', 'Debit') == 'Debit']
            
            if debit_transactions.empty:
                return {}
            
            category_spending = debit_transactions.groupby('category')['amount'].sum()
            return category_spending.to_dict()
            
        except Exception as e:
            logger.error(f"Error analyzing category breakdown: {e}")
            return {}
    
    def _analyze_spending_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spending trends over time."""
        try:
            if df.empty or 'date' not in df.columns:
                return {}
            
            # Ensure date is datetime
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter debit transactions
            debit_df = df[df.get('transaction_type', 'Debit') == 'Debit'].copy()
            
            if debit_df.empty:
                return {}
            
            trends = {}
            
            # Daily spending trend
            daily_spending = debit_df.groupby('date')['amount'].sum()
            trends['daily_average'] = float(daily_spending.mean())
            trends['daily_std'] = float(daily_spending.std())
            
            # Weekly spending pattern
            debit_df['day_of_week'] = debit_df['date'].dt.day_name()
            weekly_pattern = debit_df.groupby('day_of_week')['amount'].sum()
            trends['weekly_pattern'] = weekly_pattern.to_dict()
            
            # Monthly spending trend
            debit_df['month'] = debit_df['date'].dt.to_period('M')
            monthly_spending = debit_df.groupby('month')['amount'].sum()
            trends['monthly_trend'] = {str(k): float(v) for k, v in monthly_spending.items()}
            
            # Recent vs historical comparison (if enough data)
            if len(daily_spending) >= 60:  # At least 2 months of data
                recent_30_days = daily_spending.tail(30).mean()
                previous_30_days = daily_spending.iloc[-60:-30].mean()
                trends['recent_vs_previous'] = {
                    'recent_30_day_avg': float(recent_30_days),
                    'previous_30_day_avg': float(previous_30_days),
                    'change_percentage': float(((recent_30_days - previous_30_days) / previous_30_days) * 100) if previous_30_days > 0 else 0
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing spending trends: {e}")
            return {}
    
    def _generate_spending_insights(self, df: pd.DataFrame, total_spending: float, 
                                  category_breakdown: Dict[str, float]) -> List[str]:
        """Generate actionable insights based on spending analysis."""
        try:
            insights = []
            
            if df.empty:
                return ["No transaction data available for analysis."]
            
            # Insight 1: Highest spending category
            if category_breakdown:
                top_category = max(category_breakdown, key=category_breakdown.get)
                top_amount = category_breakdown[top_category]
                percentage = (top_amount / total_spending) * 100 if total_spending > 0 else 0
                insights.append(f"Your highest spending category is '{top_category}' at ${top_amount:.2f} ({percentage:.1f}% of total spending).")
            
            # Insight 2: Transaction frequency analysis
            if 'date' in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                date_range = (df_copy['date'].max() - df_copy['date'].min()).days
                if date_range > 0:
                    avg_transactions_per_day = len(df) / date_range
                    insights.append(f"You average {avg_transactions_per_day:.1f} transactions per day over the analyzed period.")
            
            # Insight 3: Large transaction analysis
            if 'amount' in df.columns and not df.empty:
                amount_mean = df['amount'].mean()
                amount_std = df['amount'].std()
                large_transactions = df[df['amount'] > (amount_mean + 2 * amount_std)]
                if not large_transactions.empty:
                    insights.append(f"You have {len(large_transactions)} transactions that are significantly larger than your average spending pattern.")
            
            # Insight 4: Weekend vs weekday spending
            if 'is_weekend' in df.columns:
                debit_df = df[df.get('transaction_type', 'Debit') == 'Debit']
                if not debit_df.empty:
                    weekend_spending = debit_df[debit_df['is_weekend']]['amount'].sum()
                    weekday_spending = debit_df[~debit_df['is_weekend']]['amount'].sum()
                    weekend_days = debit_df['is_weekend'].sum()
                    weekday_days = (~debit_df['is_weekend']).sum()
                    
                    if weekend_days > 0 and weekday_days > 0:
                        weekend_avg = weekend_spending / weekend_days
                        weekday_avg = weekday_spending / weekday_days
                        if weekend_avg > weekday_avg * 1.2:
                            insights.append(f"You tend to spend more on weekends (${weekend_avg:.2f} avg) compared to weekdays (${weekday_avg:.2f} avg).")
                        elif weekday_avg > weekend_avg * 1.2:
                            insights.append(f"You spend more on weekdays (${weekday_avg:.2f} avg) compared to weekends (${weekend_avg:.2f} avg).")
            
            # Insight 5: Income vs spending analysis
            if 'transaction_type' in df.columns:
                income_transactions = df[df['transaction_type'] == 'Credit']
                debit_transactions = df[df['transaction_type'] == 'Debit']
                
                if not income_transactions.empty and not debit_transactions.empty:
                    total_income = income_transactions['amount'].sum()
                    total_debits = debit_transactions['amount'].sum()
                    
                    if total_income > 0:
                        spending_ratio = (total_debits / total_income) * 100
                        if spending_ratio > 90:
                            insights.append(f"You're spending {spending_ratio:.1f}% of your income. Consider reviewing your budget to increase savings.")
                        elif spending_ratio < 70:
                            insights.append(f"Great job! You're only spending {spending_ratio:.1f}% of your income, leaving room for savings and investments.")
            
            # Insight 6: Category diversity
            if category_breakdown and len(category_breakdown) > 1:
                category_count = len(category_breakdown)
                if category_count >= 5:
                    insights.append(f"Your spending is well-diversified across {category_count} different categories.")
                else:
                    insights.append(f"Your spending is concentrated in {category_count} categories. Consider if this aligns with your financial goals.")
            
            # Default insight if no specific insights generated
            if not insights:
                insights.append("Your financial data has been analyzed. Consider setting specific financial goals to receive more targeted insights.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Error generating insights from your financial data."]
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result structure."""
        return {
            "total_spending": 0.0,
            "category_breakdown": {},
            "spending_trends": {},
            "insights": ["No transaction data available for analysis."],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def analyze_cash_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cash flow patterns (income vs expenses)."""
        try:
            if df.empty or 'transaction_type' not in df.columns:
                return {}
            
            # Separate income and expenses
            income_df = df[df['transaction_type'] == 'Credit']
            expense_df = df[df['transaction_type'] == 'Debit']
            
            total_income = float(income_df['amount'].sum()) if not income_df.empty else 0.0
            total_expenses = float(expense_df['amount'].sum()) if not expense_df.empty else 0.0
            net_cash_flow = total_income - total_expenses
            
            # Monthly cash flow analysis
            monthly_cash_flow = {}
            if 'date' in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                df_copy['month'] = df_copy['date'].dt.to_period('M')
                
                for month in df_copy['month'].unique():
                    month_data = df_copy[df_copy['month'] == month]
                    month_income = month_data[month_data['transaction_type'] == 'Credit']['amount'].sum()
                    month_expenses = month_data[month_data['transaction_type'] == 'Debit']['amount'].sum()
                    monthly_cash_flow[str(month)] = {
                        'income': float(month_income),
                        'expenses': float(month_expenses),
                        'net': float(month_income - month_expenses)
                    }
            
            return {
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_cash_flow': net_cash_flow,
                'monthly_cash_flow': monthly_cash_flow,
                'savings_rate': (net_cash_flow / total_income * 100) if total_income > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cash flow: {e}")
            return {}
    
    def get_spending_summary(self, df: pd.DataFrame, period: str = 'monthly') -> Dict[str, Any]:
        """Get spending summary for a specific period."""
        try:
            if df.empty or 'date' not in df.columns:
                return {}
            
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # Filter debit transactions
            debit_df = df_copy[df_copy.get('transaction_type', 'Debit') == 'Debit']
            
            if debit_df.empty:
                return {}
            
            if period == 'daily':
                grouped = debit_df.groupby(debit_df['date'].dt.date)
            elif period == 'weekly':
                grouped = debit_df.groupby(debit_df['date'].dt.to_period('W'))
            elif period == 'monthly':
                grouped = debit_df.groupby(debit_df['date'].dt.to_period('M'))
            else:
                grouped = debit_df.groupby(debit_df['date'].dt.to_period('Y'))
            
            summary = {}
            for period_key, group in grouped:
                summary[str(period_key)] = {
                    'total_amount': float(group['amount'].sum()),
                    'transaction_count': len(group),
                    'average_transaction': float(group['amount'].mean()),
                    'categories': group['category'].value_counts().to_dict() if 'category' in group.columns else {}
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting spending summary: {e}")
            return {}
