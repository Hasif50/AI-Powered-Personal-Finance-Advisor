"""
Spending Segmentation Module for AI-Powered Personal Finance Advisor
From Hasif's Workspace

K-means clustering for spending behavior analysis and user segmentation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os

logger = logging.getLogger(__name__)

class SpendingSegmentation:
    """K-means clustering for spending behavior segmentation."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the spending segmentation module."""
        self.model_dir = model_dir
        self.kmeans_model = None
        self.scaler = None
        self.feature_columns = []
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing models
        self._load_models()
    
    def is_ready(self) -> bool:
        """Check if the segmentation module is ready."""
        return True  # Can work with or without pre-trained models
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            kmeans_path = os.path.join(self.model_dir, 'kmeans_model.joblib')
            scaler_path = os.path.join(self.model_dir, 'scaler_clustering.joblib')
            
            if os.path.exists(kmeans_path) and os.path.exists(scaler_path):
                self.kmeans_model = joblib.load(kmeans_path)
                self.scaler = joblib.load(scaler_path)
                
                # Get feature columns from scaler if available
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_columns = list(self.scaler.feature_names_in_)
                
                self.is_trained = True
                logger.info("Loaded pre-trained clustering models")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            self.is_trained = False
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.kmeans_model is not None:
                kmeans_path = os.path.join(self.model_dir, 'kmeans_model.joblib')
                joblib.dump(self.kmeans_model, kmeans_path)
                logger.info(f"Saved K-means model to {kmeans_path}")
            
            if self.scaler is not None:
                scaler_path = os.path.join(self.model_dir, 'scaler_clustering.joblib')
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def segment_transactions(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Perform spending behavior segmentation on transactions."""
        try:
            if df.empty:
                return self._empty_segmentation_result()
            
            # Prepare features for clustering
            X, feature_names = self._prepare_features(df)
            
            if X.empty:
                return self._empty_segmentation_result()
            
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                n_clusters = self._determine_optimal_clusters(X)
            
            # Train or use existing model
            if not self.is_trained or self.feature_columns != feature_names:
                self._train_clustering_model(X, feature_names, n_clusters)
            
            # Apply clustering
            cluster_labels = self._predict_clusters(X, feature_names)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, X, cluster_labels, feature_names)
            
            # Generate insights
            insights = self._generate_segmentation_insights(cluster_analysis)
            
            return {
                "segments": cluster_labels.tolist(),
                "cluster_analysis": cluster_analysis,
                "insights": insights,
                "n_clusters": n_clusters,
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in spending segmentation: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for clustering."""
        try:
            if df.empty:
                return pd.DataFrame(), []
            
            # Start with amount as base feature
            features = ['amount']
            df_features = df.copy()
            
            # One-hot encode category
            if 'category' in df_features.columns:
                category_dummies = pd.get_dummies(df_features['category'], prefix='cat')
                df_features = pd.concat([df_features, category_dummies], axis=1)
                features.extend(category_dummies.columns.tolist())
            
            # One-hot encode day of week
            if 'day_of_week' in df_features.columns:
                dow_dummies = pd.get_dummies(df_features['day_of_week'], prefix='day')
                df_features = pd.concat([df_features, dow_dummies], axis=1)
                features.extend(dow_dummies.columns.tolist())
            
            # Add time-based features if available
            if 'is_weekend' in df_features.columns:
                df_features['is_weekend_int'] = df_features['is_weekend'].astype(int)
                features.append('is_weekend_int')
            
            # Select and clean features
            X = df_features[features].fillna(0)
            
            # Convert boolean columns to int
            for col in X.columns:
                if X[col].dtype == 'bool':
                    X[col] = X[col].astype(int)
            
            return X, features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), []
    
    def _determine_optimal_clusters(self, X: pd.DataFrame, max_clusters: int = 8) -> int:
        """Determine optimal number of clusters using elbow method."""
        try:
            if X.empty or len(X) < 2:
                return 2
            
            # Limit max clusters based on data size
            max_clusters = min(max_clusters, len(X) - 1, 8)
            
            if max_clusters < 2:
                return 2
            
            # Scale features for clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate WCSS for different cluster numbers
            wcss = []
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
                
                # Calculate silhouette score
                if len(X_scaled) > k:
                    sil_score = silhouette_score(X_scaled, kmeans.labels_)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
            
            # Find elbow point (simple method)
            if len(wcss) >= 2:
                # Calculate rate of change
                rate_of_change = []
                for i in range(1, len(wcss)):
                    rate_of_change.append(wcss[i-1] - wcss[i])
                
                # Find the point where rate of change starts to decrease significantly
                if len(rate_of_change) >= 2:
                    for i in range(1, len(rate_of_change)):
                        if rate_of_change[i] < rate_of_change[i-1] * 0.5:
                            return k_range[i]
            
            # Fallback: use silhouette score
            if silhouette_scores:
                best_k_idx = np.argmax(silhouette_scores)
                return k_range[best_k_idx]
            
            # Default fallback
            return min(4, max_clusters)
            
        except Exception as e:
            logger.error(f"Error determining optimal clusters: {e}")
            return 3
    
    def _train_clustering_model(self, X: pd.DataFrame, feature_names: List[str], n_clusters: int):
        """Train the K-means clustering model."""
        try:
            # Initialize scaler and model
            self.scaler = StandardScaler()
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train K-means model
            self.kmeans_model.fit(X_scaled)
            
            # Store feature columns
            self.feature_columns = feature_names
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            logger.info(f"Trained K-means model with {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
            raise
    
    def _predict_clusters(self, X: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """Predict cluster labels for the given features."""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            # Ensure feature alignment
            if self.feature_columns != feature_names:
                # Reindex to match training features
                X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
            else:
                X_aligned = X
            
            # Scale features
            X_scaled = self.scaler.transform(X_aligned)
            
            # Predict clusters
            cluster_labels = self.kmeans_model.predict(X_scaled)
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error predicting clusters: {e}")
            raise
    
    def _analyze_clusters(self, df: pd.DataFrame, X: pd.DataFrame, 
                         cluster_labels: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze the characteristics of each cluster."""
        try:
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = cluster_labels
            
            cluster_analysis = {}
            
            for cluster_id in np.unique(cluster_labels):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                cluster_features = X[cluster_labels == cluster_id]
                
                analysis = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(df)) * 100,
                    'avg_amount': float(cluster_data['amount'].mean()),
                    'total_amount': float(cluster_data['amount'].sum()),
                    'std_amount': float(cluster_data['amount'].std()),
                    'transaction_count': len(cluster_data)
                }
                
                # Category distribution
                if 'category' in cluster_data.columns:
                    category_dist = cluster_data['category'].value_counts(normalize=True)
                    analysis['top_categories'] = category_dist.head(3).to_dict()
                
                # Day of week distribution
                if 'day_of_week' in cluster_data.columns:
                    dow_dist = cluster_data['day_of_week'].value_counts(normalize=True)
                    analysis['day_of_week_pattern'] = dow_dist.to_dict()
                
                # Feature means for this cluster
                if not cluster_features.empty:
                    analysis['feature_means'] = cluster_features.mean().to_dict()
                
                cluster_analysis[f'cluster_{cluster_id}'] = analysis
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}")
            return {}
    
    def _generate_segmentation_insights(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from cluster analysis."""
        try:
            insights = []
            
            if not cluster_analysis:
                return ["No cluster analysis available."]
            
            # Find largest cluster
            largest_cluster = max(cluster_analysis.keys(), 
                                key=lambda k: cluster_analysis[k]['size'])
            largest_size = cluster_analysis[largest_cluster]['percentage']
            insights.append(f"Your largest spending pattern group represents {largest_size:.1f}% of your transactions.")
            
            # Find highest spending cluster
            highest_spending_cluster = max(cluster_analysis.keys(), 
                                         key=lambda k: cluster_analysis[k]['avg_amount'])
            highest_avg = cluster_analysis[highest_spending_cluster]['avg_amount']
            insights.append(f"Your highest-value transaction group has an average amount of ${highest_avg:.2f}.")
            
            # Analyze category patterns
            for cluster_id, analysis in cluster_analysis.items():
                if 'top_categories' in analysis and analysis['top_categories']:
                    top_category = max(analysis['top_categories'], 
                                     key=analysis['top_categories'].get)
                    percentage = analysis['top_categories'][top_category] * 100
                    if percentage > 50:
                        insights.append(f"Cluster {cluster_id.split('_')[1]} is dominated by '{top_category}' transactions ({percentage:.1f}%).")
            
            # Weekend vs weekday patterns
            for cluster_id, analysis in cluster_analysis.items():
                if 'day_of_week_pattern' in analysis:
                    weekend_days = ['Saturday', 'Sunday']
                    weekend_percentage = sum(analysis['day_of_week_pattern'].get(day, 0) 
                                           for day in weekend_days) * 100
                    if weekend_percentage > 40:
                        insights.append(f"Cluster {cluster_id.split('_')[1]} shows high weekend activity ({weekend_percentage:.1f}%).")
            
            # Spending diversity
            cluster_count = len(cluster_analysis)
            if cluster_count >= 4:
                insights.append(f"Your spending shows good diversity with {cluster_count} distinct patterns identified.")
            elif cluster_count <= 2:
                insights.append(f"Your spending patterns are quite concentrated in {cluster_count} main groups.")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating segmentation insights: {e}")
            return ["Error generating insights from segmentation analysis."]
    
    def _empty_segmentation_result(self) -> Dict[str, Any]:
        """Return empty segmentation result."""
        return {
            "segments": [],
            "cluster_analysis": {},
            "insights": ["No transaction data available for segmentation."],
            "n_clusters": 0,
            "feature_names": []
        }
