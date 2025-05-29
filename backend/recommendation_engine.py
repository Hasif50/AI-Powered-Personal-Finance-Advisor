"""
Recommendation Engine for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Rule-based recommendation system for personalized financial advice.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Rule-based recommendation engine for personalized financial advice."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommendation_rules = self._initialize_rules()
        
    def is_ready(self) -> bool:
        """Check if the recommendation engine is ready."""
        return True
    
    def generate_recommendations(self, spending_analysis: Dict[str, Any], 
                               segmentation_result: Dict[str, Any],
                               anomaly_result: Dict[str, Any],
                               transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate personalized financial recommendations."""
        try:
            recommendations = []
            priority_actions = []
            goal_progress = {}
            
            # Generate spending-based recommendations
            spending_recs = self._generate_spending_recommendations(spending_analysis, transaction_data)
            recommendations.extend(spending_recs)
            
            # Generate segmentation-based recommendations
            segment_recs = self._generate_segmentation_recommendations(segmentation_result)
            recommendations.extend(segment_recs)
            
            # Generate anomaly-based recommendations
            anomaly_recs = self._generate_anomaly_recommendations(anomaly_result)
            recommendations.extend(anomaly_recs)
            
            # Generate general financial health recommendations
            health_recs = self._generate_financial_health_recommendations(transaction_data)
            recommendations.extend(health_recs)
            
            # Prioritize recommendations
            priority_actions = self._prioritize_recommendations(recommendations)
            
            # Calculate goal progress (simulated for demo)
            goal_progress = self._calculate_goal_progress(spending_analysis, transaction_data)
            
            return {
                "recommendations": recommendations,
                "priority_actions": priority_actions,
                "goal_progress": goal_progress,
                "total_recommendations": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._empty_recommendation_result()
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize recommendation rules."""
        return {
            "spending_thresholds": {
                "high_category_percentage": 40,  # If a category is >40% of spending
                "low_savings_rate": 10,  # If savings rate is <10%
                "high_weekend_spending": 30  # If weekend spending is >30% higher
            },
            "anomaly_thresholds": {
                "high_anomaly_percentage": 5,  # If >5% transactions are anomalous
                "large_transaction_threshold": 500  # Transactions >$500
            },
            "recommendation_priorities": {
                "critical": ["fraud_alert", "budget_exceeded", "negative_cash_flow"],
                "high": ["high_spending_category", "low_savings", "unusual_pattern"],
                "medium": ["optimization", "goal_tracking", "trend_analysis"],
                "low": ["general_tips", "educational"]
            }
        }
    
    def _generate_spending_recommendations(self, spending_analysis: Dict[str, Any], 
                                         transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate recommendations based on spending analysis."""
        recommendations = []
        
        try:
            category_breakdown = spending_analysis.get("category_breakdown", {})
            total_spending = spending_analysis.get("total_spending", 0)
            spending_trends = spending_analysis.get("spending_trends", {})
            
            # High spending category recommendation
            if category_breakdown and total_spending > 0:
                top_category = max(category_breakdown, key=category_breakdown.get)
                top_percentage = (category_breakdown[top_category] / total_spending) * 100
                
                if top_percentage > self.recommendation_rules["spending_thresholds"]["high_category_percentage"]:
                    recommendations.append({
                        "type": "spending_optimization",
                        "priority": "high",
                        "category": "budget_management",
                        "title": f"High Spending in {top_category}",
                        "description": f"Your '{top_category}' spending represents {top_percentage:.1f}% of your total expenses. Consider setting a specific budget for this category.",
                        "action_items": [
                            f"Set a monthly budget limit for {top_category}",
                            f"Track {top_category} expenses more closely",
                            f"Look for alternatives to reduce {top_category} costs"
                        ],
                        "potential_savings": category_breakdown[top_category] * 0.1  # 10% potential savings
                    })
            
            # Weekend vs weekday spending analysis
            weekly_pattern = spending_trends.get("weekly_pattern", {})
            if weekly_pattern:
                weekend_spending = weekly_pattern.get("Saturday", 0) + weekly_pattern.get("Sunday", 0)
                weekday_spending = sum(v for k, v in weekly_pattern.items() if k not in ["Saturday", "Sunday"])
                
                if weekend_spending > 0 and weekday_spending > 0:
                    weekend_avg = weekend_spending / 2
                    weekday_avg = weekday_spending / 5
                    
                    if weekend_avg > weekday_avg * 1.3:
                        recommendations.append({
                            "type": "spending_pattern",
                            "priority": "medium",
                            "category": "lifestyle",
                            "title": "High Weekend Spending",
                            "description": f"You spend {((weekend_avg/weekday_avg - 1) * 100):.1f}% more on weekends compared to weekdays.",
                            "action_items": [
                                "Plan weekend activities with a budget in mind",
                                "Look for free or low-cost weekend activities",
                                "Consider meal prepping to reduce weekend dining costs"
                            ],
                            "potential_savings": (weekend_avg - weekday_avg) * 8  # Monthly savings
                        })
            
            # Recent spending trend analysis
            recent_vs_previous = spending_trends.get("recent_vs_previous", {})
            if recent_vs_previous:
                change_percentage = recent_vs_previous.get("change_percentage", 0)
                if change_percentage > 20:
                    recommendations.append({
                        "type": "trend_alert",
                        "priority": "high",
                        "category": "budget_management",
                        "title": "Increasing Spending Trend",
                        "description": f"Your spending has increased by {change_percentage:.1f}% compared to the previous month.",
                        "action_items": [
                            "Review recent purchases for unnecessary expenses",
                            "Identify what caused the spending increase",
                            "Set stricter budget limits for the next month"
                        ],
                        "potential_savings": recent_vs_previous.get("recent_30_day_avg", 0) * 0.05
                    })
                elif change_percentage < -10:
                    recommendations.append({
                        "type": "positive_feedback",
                        "priority": "low",
                        "category": "achievement",
                        "title": "Great Job Reducing Spending!",
                        "description": f"You've reduced your spending by {abs(change_percentage):.1f}% compared to last month.",
                        "action_items": [
                            "Continue the good spending habits",
                            "Consider investing the saved money",
                            "Set new savings goals"
                        ],
                        "potential_savings": 0
                    })
            
        except Exception as e:
            logger.error(f"Error generating spending recommendations: {e}")
        
        return recommendations
    
    def _generate_segmentation_recommendations(self, segmentation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on spending segmentation."""
        recommendations = []
        
        try:
            cluster_analysis = segmentation_result.get("cluster_analysis", {})
            
            if not cluster_analysis:
                return recommendations
            
            # Find the user's dominant spending pattern
            largest_cluster = max(cluster_analysis.keys(), 
                                key=lambda k: cluster_analysis[k]["size"])
            cluster_info = cluster_analysis[largest_cluster]
            
            # High-value transaction cluster
            if cluster_info.get("avg_amount", 0) > 200:
                recommendations.append({
                    "type": "spending_behavior",
                    "priority": "medium",
                    "category": "pattern_analysis",
                    "title": "High-Value Transaction Pattern",
                    "description": f"Your largest spending group has an average transaction of ${cluster_info['avg_amount']:.2f}.",
                    "action_items": [
                        "Review high-value purchases for necessity",
                        "Consider if these purchases align with your financial goals",
                        "Look for opportunities to negotiate better prices"
                    ],
                    "potential_savings": cluster_info["avg_amount"] * 0.05 * cluster_info["size"]
                })
            
            # Category concentration analysis
            top_categories = cluster_info.get("top_categories", {})
            if top_categories:
                dominant_category = max(top_categories, key=top_categories.get)
                if top_categories[dominant_category] > 0.6:  # >60% concentration
                    recommendations.append({
                        "type": "diversification",
                        "priority": "medium",
                        "category": "spending_diversity",
                        "title": f"Concentrated Spending in {dominant_category}",
                        "description": f"One of your spending patterns is heavily concentrated in {dominant_category} ({top_categories[dominant_category]*100:.1f}%).",
                        "action_items": [
                            f"Review if {dominant_category} spending is necessary",
                            "Look for ways to optimize costs in this category",
                            "Consider if this concentration aligns with your priorities"
                        ],
                        "potential_savings": cluster_info["total_amount"] * 0.1
                    })
            
        except Exception as e:
            logger.error(f"Error generating segmentation recommendations: {e}")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on anomaly detection."""
        recommendations = []
        
        try:
            summary = anomaly_result.get("summary", {})
            anomalous_transactions = anomaly_result.get("anomalous_transactions", [])
            anomaly_percentage = summary.get("anomaly_percentage", 0)
            
            # High anomaly rate alert
            if anomaly_percentage > self.recommendation_rules["anomaly_thresholds"]["high_anomaly_percentage"]:
                recommendations.append({
                    "type": "security_alert",
                    "priority": "critical",
                    "category": "fraud_prevention",
                    "title": "High Number of Unusual Transactions",
                    "description": f"{anomaly_percentage:.1f}% of your transactions appear unusual. Please review for potential fraud or errors.",
                    "action_items": [
                        "Review all flagged transactions immediately",
                        "Check bank statements for unauthorized charges",
                        "Contact your bank if you find suspicious activity",
                        "Consider changing your payment card if necessary"
                    ],
                    "potential_savings": 0
                })
            
            # Large transaction alerts
            large_transactions = [t for t in anomalous_transactions 
                                if t.get("amount", 0) > self.recommendation_rules["anomaly_thresholds"]["large_transaction_threshold"]]
            
            if large_transactions:
                recommendations.append({
                    "type": "transaction_review",
                    "priority": "high",
                    "category": "expense_review",
                    "title": "Large Unusual Transactions Detected",
                    "description": f"Found {len(large_transactions)} large transactions that appear unusual.",
                    "action_items": [
                        "Verify all large transactions are legitimate",
                        "Ensure large purchases align with your budget",
                        "Consider if these were planned expenses"
                    ],
                    "potential_savings": 0,
                    "transactions": large_transactions[:3]  # Show top 3
                })
            
            # Category-specific anomalies
            if summary.get("anomaly_categories"):
                for category, count in summary["anomaly_categories"].items():
                    if count >= 3:  # Multiple anomalies in same category
                        recommendations.append({
                            "type": "category_alert",
                            "priority": "medium",
                            "category": "pattern_analysis",
                            "title": f"Multiple Unusual {category} Transactions",
                            "description": f"Detected {count} unusual transactions in {category}. This might indicate a change in spending pattern.",
                            "action_items": [
                                f"Review your {category} spending pattern",
                                f"Check if {category} prices have increased",
                                f"Consider if your {category} needs have changed"
                            ],
                            "potential_savings": 0
                        })
            
        except Exception as e:
            logger.error(f"Error generating anomaly recommendations: {e}")
        
        return recommendations
    
    def _generate_financial_health_recommendations(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate general financial health recommendations."""
        recommendations = []
        
        try:
            if transaction_data.empty:
                return recommendations
            
            # Calculate basic financial metrics
            if 'transaction_type' in transaction_data.columns:
                income = transaction_data[transaction_data['transaction_type'] == 'Credit']['amount'].sum()
                expenses = transaction_data[transaction_data['transaction_type'] == 'Debit']['amount'].sum()
                
                if income > 0:
                    savings_rate = ((income - expenses) / income) * 100
                    
                    if savings_rate < 10:
                        recommendations.append({
                            "type": "savings_improvement",
                            "priority": "high",
                            "category": "financial_health",
                            "title": "Low Savings Rate",
                            "description": f"Your current savings rate is {savings_rate:.1f}%. Financial experts recommend saving at least 20% of income.",
                            "action_items": [
                                "Create a detailed budget to identify areas to cut expenses",
                                "Look for ways to increase your income",
                                "Start with a goal of saving 1% more each month",
                                "Consider automating your savings"
                            ],
                            "potential_savings": income * 0.1  # Target 10% savings
                        })
                    elif savings_rate > 30:
                        recommendations.append({
                            "type": "investment_opportunity",
                            "priority": "medium",
                            "category": "wealth_building",
                            "title": "Excellent Savings Rate - Consider Investing",
                            "description": f"Your savings rate of {savings_rate:.1f}% is excellent! Consider investing your surplus for long-term growth.",
                            "action_items": [
                                "Research investment options like index funds or ETFs",
                                "Consider opening a retirement account (401k, IRA)",
                                "Diversify your investments across different asset classes",
                                "Consult with a financial advisor"
                            ],
                            "potential_savings": 0
                        })
            
            # Transaction frequency analysis
            if 'date' in transaction_data.columns:
                transaction_data_copy = transaction_data.copy()
                transaction_data_copy['date'] = pd.to_datetime(transaction_data_copy['date'])
                date_range = (transaction_data_copy['date'].max() - transaction_data_copy['date'].min()).days
                
                if date_range > 0:
                    avg_transactions_per_day = len(transaction_data) / date_range
                    
                    if avg_transactions_per_day > 10:
                        recommendations.append({
                            "type": "spending_frequency",
                            "priority": "medium",
                            "category": "behavioral_change",
                            "title": "High Transaction Frequency",
                            "description": f"You average {avg_transactions_per_day:.1f} transactions per day. Consider consolidating purchases.",
                            "action_items": [
                                "Plan purchases in advance to reduce impulse buying",
                                "Make shopping lists and stick to them",
                                "Consider bulk purchases for frequently used items",
                                "Use the 24-hour rule for non-essential purchases"
                            ],
                            "potential_savings": transaction_data['amount'].sum() * 0.05
                        })
            
        except Exception as e:
            logger.error(f"Error generating financial health recommendations: {e}")
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Prioritize recommendations and return top priority actions."""
        try:
            priority_order = ["critical", "high", "medium", "low"]
            priority_actions = []
            
            # Sort recommendations by priority
            sorted_recs = sorted(recommendations, 
                               key=lambda x: priority_order.index(x.get("priority", "low")))
            
            # Extract top priority actions
            for rec in sorted_recs[:5]:  # Top 5 recommendations
                if rec.get("action_items"):
                    priority_actions.extend(rec["action_items"][:2])  # Top 2 actions per recommendation
            
            return priority_actions[:10]  # Return top 10 priority actions
            
        except Exception as e:
            logger.error(f"Error prioritizing recommendations: {e}")
            return []
    
    def _calculate_goal_progress(self, spending_analysis: Dict[str, Any], 
                               transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate progress towards financial goals (simulated for demo)."""
        try:
            # Simulated goals for demonstration
            goals = {
                "emergency_fund": {
                    "target": 5000,
                    "current": 2500,  # Simulated
                    "progress_percentage": 50,
                    "status": "in_progress"
                },
                "monthly_savings": {
                    "target": 500,
                    "current": 300,  # Based on analysis
                    "progress_percentage": 60,
                    "status": "in_progress"
                },
                "debt_reduction": {
                    "target": 10000,
                    "current": 7500,  # Simulated
                    "progress_percentage": 75,
                    "status": "in_progress"
                }
            }
            
            # Update based on actual data if available
            if 'transaction_type' in transaction_data.columns:
                income = transaction_data[transaction_data['transaction_type'] == 'Credit']['amount'].sum()
                expenses = transaction_data[transaction_data['transaction_type'] == 'Debit']['amount'].sum()
                actual_savings = income - expenses
                
                if actual_savings > 0:
                    goals["monthly_savings"]["current"] = min(actual_savings, goals["monthly_savings"]["target"])
                    goals["monthly_savings"]["progress_percentage"] = (goals["monthly_savings"]["current"] / goals["monthly_savings"]["target"]) * 100
            
            return goals
            
        except Exception as e:
            logger.error(f"Error calculating goal progress: {e}")
            return {}
    
    def _empty_recommendation_result(self) -> Dict[str, Any]:
        """Return empty recommendation result."""
        return {
            "recommendations": [],
            "priority_actions": [],
            "goal_progress": {},
            "total_recommendations": 0
        }
