import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
from bs4 import BeautifulSoup
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def clean_html(html_text: str) -> str:
    """Clean HTML content from description fields."""
    if pd.isna(html_text) or not isinstance(html_text, str):
        return ''
    
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def extract_rating_from_link(rating_link: str) -> float:
    """Extract rating value from rating icon link."""
    if pd.isna(rating_link) or not isinstance(rating_link, str):
        return np.nan
    
    # Example pattern: looking for rating value in URL or alt text
    pattern = r'(\d+\.\d+)'
    matches = re.search(pattern, rating_link)
    if matches:
        return float(matches.group(1))
    
    return np.nan

def preprocess_apps(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the apps dataset."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Clean HTML in descriptions if needed
    if 'description_raw' in df.columns and 'description' not in df.columns:
        logger.info("Cleaning HTML from app descriptions")
        df['description'] = df['description_raw'].apply(clean_html)
    
    # Extract rating from rating link if needed
    if 'rating' in df.columns:
        try:
            if isinstance(df['rating'].iloc[0], str) and ('http' in df['rating'].iloc[0] or 'svg' in df['rating'].iloc[0]):
                logger.info("Extracting numeric ratings from rating links")
                df['rating_value'] = df['rating'].apply(extract_rating_from_link)
        except (IndexError, TypeError):
            pass  # Handle empty dataframes or non-string types
    
    # Convert reviews_count to numeric
    if 'reviews_count' in df.columns:
        df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce')
    
    return df

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the reviews dataset."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['posted_at', 'developer_reply_posted_at']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract year and month for time-based analysis
    if 'posted_at' in df.columns:
        df['posted_year'] = df['posted_at'].dt.year
        df['posted_month'] = df['posted_at'].dt.month
    
    # Convert rating to numeric
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Create a has_developer_reply flag
    if 'developer_reply' in df.columns:
        df['has_developer_reply'] = df['developer_reply'].notna()
    
    # Convert helpful_count to numeric
    if 'helpful_count' in df.columns:
        df['helpful_count'] = pd.to_numeric(df['helpful_count'], errors='coerce')
    
    return df

def get_apps_with_categories(apps_df: pd.DataFrame, 
                          apps_categories_df: pd.DataFrame, 
                          categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join apps with their categories.
    
    Returns:
        DataFrame with apps and their categories as a list in a new 'categories' column
    """
    if apps_df.empty or apps_categories_df.empty or categories_df.empty:
        return apps_df
    
    # Merge categories with apps_categories to get category names
    categories_mapping = pd.merge(
        apps_categories_df,
        categories_df,
        left_on='category_id',
        right_on='id',
        how='left'
    )
    
    # Group by app_id and aggregate category titles into lists
    app_categories = categories_mapping.groupby('app_id')['title'].apply(list).reset_index()
    app_categories.rename(columns={'title': 'categories'}, inplace=True)
    
    # Merge with apps dataframe
    result = pd.merge(
        apps_df,
        app_categories,
        left_on='id',
        right_on='app_id',
        how='left',
        suffixes=(None, '_cat')
    )
    
    # Drop the duplicate app_id column
    if 'app_id' in result.columns:
        result = result.drop('app_id', axis=1)
    
    # Handle apps with no categories
    def fill_empty_categories(x):
        if isinstance(x, list):
            return x
        return []
    
    result['categories'] = result['categories'].apply(fill_empty_categories)
    
    return result

def get_app_reviews_summary(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for app reviews.
    
    Returns:
        DataFrame with app_id and review statistics (count, avg rating, etc.)
    """
    if reviews_df.empty:
        return pd.DataFrame()
    
    # Group by app_id and calculate summary statistics
    summary = reviews_df.groupby('app_id').agg(
        review_count=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        min_rating=('rating', 'min'),
        max_rating=('rating', 'max'),
        std_rating=('rating', 'std'),
        developer_reply_count=('has_developer_reply', 'sum'),
        reply_rate=('has_developer_reply', 'mean'),
        earliest_review=('posted_at', 'min'),
        latest_review=('posted_at', 'max')
    ).reset_index()
    
    return summary

def categorize_pricing_plans(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize pricing plans into common categories."""
    if df.empty or 'title' not in df.columns:
        return df
    
    df = df.copy()
    
    # Create a plan_type column based on the title
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
    
    df['plan_type'] = df['title'].apply(categorize_plan)
    
    return df

def create_master_dataset(
    apps_df: pd.DataFrame,
    apps_categories_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    reviews_summary_df: Optional[pd.DataFrame] = None,
    key_benefits_df: Optional[pd.DataFrame] = None,
    pricing_plans_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create a master dataset joining apps with their categories, reviews, and other relevant data.
    
    Args:
        apps_df: Apps dataset
        apps_categories_df: App-category relationships
        categories_df: Categories dataset
        reviews_summary_df: Optional summary of reviews by app
        key_benefits_df: Optional key benefits dataset
        pricing_plans_df: Optional pricing plans dataset
        
    Returns:
        Consolidated master dataset
    """
    # Start with apps and their categories
    master_df = get_apps_with_categories(apps_df, apps_categories_df, categories_df)
    
    # Add review statistics if available
    if reviews_summary_df is not None and not reviews_summary_df.empty:
        master_df = pd.merge(
            master_df,
            reviews_summary_df,
            left_on='id',
            right_on='app_id',
            how='left',
            suffixes=(None, '_review')
        )
        # Drop the duplicate app_id column
        if 'app_id' in master_df.columns:
            master_df = master_df.drop('app_id', axis=1)
    
    # Add key benefits count if available
    if key_benefits_df is not None and not key_benefits_df.empty:
        benefits_count = key_benefits_df.groupby('app_id').size().reset_index(name='benefits_count')
        master_df = pd.merge(
            master_df,
            benefits_count,
            left_on='id',
            right_on='app_id',
            how='left',
            suffixes=(None, '_benefit')
        )
        # Drop the duplicate app_id column
        if 'app_id' in master_df.columns:
            master_df = master_df.drop('app_id', axis=1)
        master_df['benefits_count'] = master_df['benefits_count'].fillna(0)
    
    # Add pricing plan counts if available
    if pricing_plans_df is not None and not pricing_plans_df.empty:
        # Categorize pricing plans first
        pricing_plans_df = categorize_pricing_plans(pricing_plans_df)
        
        # Get counts by plan type
        plan_counts = pricing_plans_df.groupby(['app_id', 'plan_type']).size().unstack(fill_value=0).reset_index()
        
        # Add a total_plans column
        if not plan_counts.empty:
            plan_types = [col for col in plan_counts.columns if col != 'app_id']
            plan_counts['total_plans'] = plan_counts[plan_types].sum(axis=1)
            
            # Merge with master dataset
            master_df = pd.merge(
                master_df,
                plan_counts,
                left_on='id',
                right_on='app_id',
                how='left',
                suffixes=(None, '_plan')
            )
            
            # Drop the duplicate app_id column
            if 'app_id' in master_df.columns:
                master_df = master_df.drop('app_id', axis=1)
                
            # Fill missing plan counts with 0
            for col in plan_counts.columns:
                if col != 'app_id' and col in master_df.columns:
                    master_df[col] = master_df[col].fillna(0)
    
    return master_df 