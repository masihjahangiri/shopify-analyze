import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter, defaultdict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.schema import App, ShopifyDataset

logger = logging.getLogger(__name__)

def get_popular_apps(df: pd.DataFrame, 
                   by: str = 'reviews_count', 
                   n: int = 10, 
                   ascending: bool = False) -> pd.DataFrame:
    """
    Get the most popular apps based on a specific metric.
    
    Args:
        df: DataFrame containing app data
        by: Column to sort by (reviews_count, rating, etc.)
        n: Number of results to return
        ascending: Whether to sort in ascending order
        
    Returns:
        DataFrame with top n apps
    """
    if by not in df.columns:
        logger.warning(f"Column '{by}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    sorted_df = df.sort_values(by=by, ascending=ascending)
    return sorted_df.head(n)

def get_category_distribution(apps_df: pd.DataFrame, 
                           apps_categories_df: pd.DataFrame, 
                           categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the distribution of apps across categories.
    
    Args:
        apps_df: DataFrame with app data
        apps_categories_df: DataFrame with app-category relationships
        categories_df: DataFrame with category data
        
    Returns:
        DataFrame with category counts and percentages
    """
    # Merge categories with apps_categories to get category names
    categories_mapping = pd.merge(
        apps_categories_df,
        categories_df,
        left_on='category_id',
        right_on='id',
        how='left'
    )
    
    # Count apps in each category
    category_counts = categories_mapping.groupby('title').size().reset_index(name='count')
    
    # Calculate percentage
    total_apps = len(apps_df)
    category_counts['percentage'] = (category_counts['count'] / total_apps) * 100
    
    # Sort by count descending
    return category_counts.sort_values('count', ascending=False)

def get_rating_distribution(df: pd.DataFrame, 
                         rating_col: str = 'rating_value',
                         bins: List[float] = None) -> pd.DataFrame:
    """
    Get the distribution of app ratings.
    
    Args:
        df: DataFrame with app data
        rating_col: Column containing rating values
        bins: Rating bins (defaults to 0.5 increments)
        
    Returns:
        DataFrame with rating distribution
    """
    if rating_col not in df.columns:
        logger.warning(f"Column '{rating_col}' not found in DataFrame")
        return pd.DataFrame()
    
    if bins is None:
        bins = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    # Create rating bins
    df_copy = df.copy()
    df_copy['rating_bin'] = pd.cut(df_copy[rating_col], bins=bins)
    
    # Count apps in each bin
    rating_dist = df_copy.groupby('rating_bin').size().reset_index(name='count')
    
    # Calculate percentage
    total_apps = len(df)
    rating_dist['percentage'] = (rating_dist['count'] / total_apps) * 100
    
    return rating_dist

def get_developer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics about app developers.
    
    Args:
        df: DataFrame with app data
        
    Returns:
        DataFrame with developer statistics
    """
    if 'developer' not in df.columns:
        logger.warning("Column 'developer' not found in DataFrame")
        return pd.DataFrame()
    
    # Group by developer and count apps
    dev_stats = df.groupby('developer').agg(
        app_count=('id', 'count')
    ).reset_index()
    
    # Add average rating if available
    rating_col = 'rating_value' if 'rating_value' in df.columns else 'rating'
    if rating_col in df.columns:
        avg_ratings = df.groupby('developer')[rating_col].mean().reset_index()
        dev_stats = pd.merge(dev_stats, avg_ratings, on='developer', how='left')
    
    # Add total reviews if available
    if 'reviews_count' in df.columns:
        total_reviews = df.groupby('developer')['reviews_count'].sum().reset_index()
        dev_stats = pd.merge(dev_stats, total_reviews, on='developer', how='left')
    
    # Sort by app count descending
    return dev_stats.sort_values('app_count', ascending=False)

def analyze_reviews_sentiment(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment in app reviews based on ratings.
    
    Args:
        reviews_df: DataFrame with review data
        
    Returns:
        DataFrame with sentiment analysis results
    """
    if reviews_df.empty or 'rating' not in reviews_df.columns:
        logger.warning("Reviews data is empty or missing rating column")
        return pd.DataFrame()
    
    # Define sentiment based on rating
    def get_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating <= 3:
            return 'neutral'
        else:
            return 'positive'
    
    reviews_df = reviews_df.copy()
    reviews_df['sentiment'] = reviews_df['rating'].apply(get_sentiment)
    
    # Group by app_id and sentiment
    sentiment_counts = reviews_df.groupby(['app_id', 'sentiment']).size().reset_index(name='count')
    
    # Pivot to get sentiment counts as columns
    sentiment_pivot = sentiment_counts.pivot_table(
        index='app_id',
        columns='sentiment',
        values='count',
        fill_value=0
    ).reset_index()
    
    # Calculate total reviews and percentages
    sentiment_pivot['total_reviews'] = sentiment_pivot.sum(axis=1)
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in sentiment_pivot.columns:
            sentiment_pivot[f'{sentiment}_pct'] = (sentiment_pivot[sentiment] / sentiment_pivot['total_reviews']) * 100
    
    return sentiment_pivot

def analyze_pricing_trends(pricing_plans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze trends in app pricing plans.
    
    Args:
        pricing_plans_df: DataFrame with pricing plan data
        
    Returns:
        DataFrame with pricing plan trends
    """
    if pricing_plans_df.empty or 'title' not in pricing_plans_df.columns:
        logger.warning("Pricing plans data is empty or missing title column")
        return pd.DataFrame()
    
    # Categorize plans
    def categorize_plan(title):
        if pd.isna(title):
            return 'Unknown'
        
        title_lower = str(title).lower()
        
        if 'free' in title_lower:
            return 'Free'
        elif 'trial' in title_lower:
            return 'Trial'
        elif any(term in title_lower for term in ['basic', 'starter', 'start']):
            return 'Basic'
        elif any(term in title_lower for term in ['standard', 'pro', 'plus', 'premium']):
            return 'Premium'
        elif any(term in title_lower for term in ['enterprise', 'business', 'advanced', 'ultimate']):
            return 'Enterprise'
        else:
            return 'Other'
    
    pricing_plans_df = pricing_plans_df.copy()
    pricing_plans_df['plan_type'] = pricing_plans_df['title'].apply(categorize_plan)
    
    # Count plans by type
    plan_counts = pricing_plans_df['plan_type'].value_counts().reset_index()
    plan_counts.columns = ['plan_type', 'count']
    
    # Calculate percentages
    total_plans = len(pricing_plans_df)
    plan_counts['percentage'] = (plan_counts['count'] / total_plans) * 100
    
    return plan_counts

def find_similar_apps(df: pd.DataFrame, 
                    app_id: int, 
                    n: int = 5, 
                    text_col: str = 'description') -> pd.DataFrame:
    """
    Find apps similar to a given app based on description similarity.
    
    Args:
        df: DataFrame with app data
        app_id: ID of the app to find similar apps for
        n: Number of similar apps to return
        text_col: Column to use for text similarity
        
    Returns:
        DataFrame with similar apps
    """
    if text_col not in df.columns:
        logger.warning(f"Column '{text_col}' not found in DataFrame")
        return pd.DataFrame()
    
    if app_id not in df['id'].values:
        logger.warning(f"App ID {app_id} not found in DataFrame")
        return pd.DataFrame()
    
    # Get app descriptions
    descriptions = df[text_col].fillna('').tolist()
    app_indices = {id: idx for idx, id in enumerate(df['id'])}
    
    # Vectorize descriptions
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Calculate similarity scores
    app_idx = app_indices[app_id]
    app_vector = tfidf_matrix[app_idx:app_idx+1]
    similarity_scores = cosine_similarity(app_vector, tfidf_matrix).flatten()
    
    # Get top similar apps (excluding the input app)
    similar_indices = similarity_scores.argsort()[::-1][1:n+1]
    similar_app_ids = [df['id'].iloc[idx] for idx in similar_indices]
    
    # Return similar apps
    similar_apps = df[df['id'].isin(similar_app_ids)].copy()
    
    # Add similarity score
    similarity_dict = {df['id'].iloc[idx]: similarity_scores[idx] for idx in similar_indices}
    similar_apps['similarity_score'] = similar_apps['id'].map(similarity_dict)
    
    return similar_apps.sort_values('similarity_score', ascending=False)

def get_time_series_data(reviews_df: pd.DataFrame, 
                        freq: str = 'M',
                        metrics: List[str] = None) -> pd.DataFrame:
    """
    Generate time series data from reviews.
    
    Args:
        reviews_df: DataFrame with review data
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Y' for yearly)
        metrics: List of metrics to calculate (defaults to ['count', 'avg_rating'])
        
    Returns:
        DataFrame with time series data
    """
    if reviews_df.empty or 'posted_at' not in reviews_df.columns:
        logger.warning("Reviews data is empty or missing posted_at column")
        return pd.DataFrame()
    
    if metrics is None:
        metrics = ['count', 'avg_rating']
    
    # Convert posted_at to datetime if needed
    reviews_df = reviews_df.copy()
    if not pd.api.types.is_datetime64_dtype(reviews_df['posted_at']):
        reviews_df['posted_at'] = pd.to_datetime(reviews_df['posted_at'], errors='coerce')
    
    # Set index to posted_at
    reviews_df = reviews_df.set_index('posted_at')
    
    # Create time series for different metrics
    result = pd.DataFrame()
    
    if 'count' in metrics:
        # Count reviews per time period
        counts = reviews_df.resample(freq).size()
        result['review_count'] = counts
    
    if 'avg_rating' in metrics and 'rating' in reviews_df.columns:
        # Average rating per time period
        avg_ratings = reviews_df['rating'].resample(freq).mean()
        result['avg_rating'] = avg_ratings
    
    if 'sentiment' in metrics and 'sentiment' in reviews_df.columns:
        # Sentiment counts per time period
        for sentiment in ['negative', 'neutral', 'positive']:
            sentiment_count = reviews_df[reviews_df['sentiment'] == sentiment].resample(freq).size()
            result[f'{sentiment}_count'] = sentiment_count
    
    # Reset index to make date a column
    result = result.reset_index()
    result = result.rename(columns={'posted_at': 'date'})
    
    return result 