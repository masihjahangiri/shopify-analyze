import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

def extract_common_terms(texts: List[str], 
                       n_terms: int = 20, 
                       ngram_range: Tuple[int, int] = (1, 2),
                       stop_words: str = 'english') -> Dict[str, int]:
    """
    Extract most common terms from a list of texts.
    
    Args:
        texts: List of text strings
        n_terms: Number of top terms to extract
        ngram_range: Range of n-grams to consider
        stop_words: Stop words to exclude
        
    Returns:
        Dictionary of term -> frequency
    """
    if not texts or all(pd.isna(text) for text in texts):
        logger.warning("No valid texts provided for term extraction")
        return {}
    
    # Filter out empty or non-string texts
    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    
    if not valid_texts:
        return {}
    
    # Create the vectorizer
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=n_terms
    )
    
    # Fit and transform the texts
    try:
        counts = vectorizer.fit_transform(valid_texts)
        features = vectorizer.get_feature_names_out()
        
        # Sum counts across all documents for each term
        term_counts = counts.sum(axis=0).A1
        
        # Create dictionary of term -> count
        term_dict = {features[i]: int(term_counts[i]) for i in range(len(features))}
        
        # Sort by count descending
        sorted_terms = {k: v for k, v in sorted(term_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_terms
    except Exception as e:
        logger.error(f"Error extracting common terms: {e}")
        return {}

def identify_trending_categories(apps_categories_df: pd.DataFrame,
                               categories_df: pd.DataFrame,
                               reviews_df: pd.DataFrame,
                               period_col: str = 'posted_year',
                               min_reviews: int = 10) -> pd.DataFrame:
    """
    Identify trending categories based on review growth.
    
    Args:
        apps_categories_df: DataFrame with app-category relationships
        categories_df: DataFrame with category data
        reviews_df: DataFrame with review data
        period_col: Column to use for time periods
        min_reviews: Minimum number of reviews for a category to be considered
        
    Returns:
        DataFrame with trending categories
    """
    if apps_categories_df.empty or categories_df.empty or reviews_df.empty:
        logger.warning("One or more required DataFrames are empty")
        return pd.DataFrame()
    
    if period_col not in reviews_df.columns:
        logger.warning(f"Column '{period_col}' not found in reviews DataFrame")
        return pd.DataFrame()
    
    # Join app categories with reviews
    app_categories = pd.merge(
        apps_categories_df,
        categories_df,
        left_on='category_id',
        right_on='id',
        how='left'
    )
    
    reviews_with_categories = pd.merge(
        reviews_df,
        app_categories,
        on='app_id',
        how='inner'
    )
    
    # Count reviews by category and period
    category_period_counts = reviews_with_categories.groupby(
        ['title_y', period_col]).size().reset_index(name='review_count')
    
    # Pivot to get periods as columns
    category_trends = category_period_counts.pivot(
        index='title_y',
        columns=period_col,
        values='review_count'
    ).fillna(0)
    
    # Filter categories with minimum reviews
    total_reviews = category_trends.sum(axis=1)
    category_trends = category_trends[total_reviews >= min_reviews]
    
    # Calculate growth rate between periods
    growth_rates = pd.DataFrame(index=category_trends.index)
    periods = sorted(category_trends.columns)
    
    for i in range(1, len(periods)):
        prev_period = periods[i-1]
        curr_period = periods[i]
        
        # Calculate growth rate (percentage change)
        growth_rate = ((category_trends[curr_period] - category_trends[prev_period]) / 
                        category_trends[prev_period].replace(0, 1)) * 100
        
        growth_rates[f'{prev_period}_to_{curr_period}_growth'] = growth_rate
    
    # Add total reviews column
    growth_rates['total_reviews'] = total_reviews[growth_rates.index]
    
    # Add average growth rate column
    if len(periods) > 1:
        growth_cols = [col for col in growth_rates.columns if 'growth' in col]
        growth_rates['avg_growth_rate'] = growth_rates[growth_cols].mean(axis=1)
    
    # Sort by average growth rate
    growth_rates = growth_rates.sort_values('avg_growth_rate', ascending=False)
    
    # Reset index to make category a column
    growth_rates = growth_rates.reset_index().rename(columns={'title_y': 'category'})
    
    return growth_rates

def extract_key_features_by_rating(pricing_plan_features_df: pd.DataFrame,
                                pricing_plans_df: pd.DataFrame,
                                apps_df: pd.DataFrame,
                                rating_threshold: float = 4.0) -> Dict[str, Dict[str, int]]:
    """
    Extract key features that distinguish high-rated vs. low-rated apps.
    
    Args:
        pricing_plan_features_df: DataFrame with pricing plan features
        pricing_plans_df: DataFrame with pricing plans
        apps_df: DataFrame with app data
        rating_threshold: Threshold for high/low rating
        
    Returns:
        Dictionary with high-rated and low-rated features
    """
    if (pricing_plan_features_df.empty or pricing_plans_df.empty or 
        apps_df.empty or 'rating_value' not in apps_df.columns):
        logger.warning("Missing required data for feature extraction")
        return {'high_rated': {}, 'low_rated': {}}
    
    # Create high/low rating groups
    apps_df = apps_df.copy()
    apps_df['rating_group'] = np.where(apps_df['rating_value'] >= rating_threshold, 'high', 'low')
    
    # Join plans with apps to get rating groups
    plans_with_ratings = pd.merge(
        pricing_plans_df,
        apps_df[['id', 'rating_group']],
        left_on='app_id',
        right_on='id',
        how='inner'
    )
    
    # Join features with plans
    features_with_ratings = pd.merge(
        pricing_plan_features_df,
        plans_with_ratings[['id', 'rating_group']],
        left_on='pricing_plan_id',
        right_on='id',
        how='inner'
    )
    
    # Extract features by rating group
    high_rated_features = features_with_ratings[features_with_ratings['rating_group'] == 'high']['feature']
    low_rated_features = features_with_ratings[features_with_ratings['rating_group'] == 'low']['feature']
    
    # Count frequency of each feature
    high_rated_counts = Counter(high_rated_features)
    low_rated_counts = Counter(low_rated_features)
    
    # Convert to dictionaries sorted by frequency
    high_rated_dict = {k: v for k, v in sorted(high_rated_counts.items(), key=lambda item: item[1], reverse=True)}
    low_rated_dict = {k: v for k, v in sorted(low_rated_counts.items(), key=lambda item: item[1], reverse=True)}
    
    return {
        'high_rated': high_rated_dict,
        'low_rated': low_rated_dict
    }

def analyze_review_text(reviews_df: pd.DataFrame, 
                      sentiment: Optional[str] = None,
                      min_rating: Optional[float] = None,
                      max_rating: Optional[float] = None,
                      n_terms: int = 30) -> Dict[str, Dict[str, int]]:
    """
    Analyze review text to extract common terms by sentiment or rating.
    
    Args:
        reviews_df: DataFrame with review data
        sentiment: Optional sentiment to filter by ('positive', 'neutral', 'negative')
        min_rating: Optional minimum rating to include
        max_rating: Optional maximum rating to include
        n_terms: Number of terms to extract
        
    Returns:
        Dictionary with extracted terms and frequencies
    """
    if reviews_df.empty or 'body' not in reviews_df.columns:
        logger.warning("Reviews data is empty or missing body column")
        return {}
    
    # Filter reviews by sentiment if specified
    filtered_reviews = reviews_df.copy()
    
    if sentiment is not None and 'sentiment' in filtered_reviews.columns:
        filtered_reviews = filtered_reviews[filtered_reviews['sentiment'] == sentiment]
    
    # Filter by rating if specified
    if min_rating is not None and 'rating' in filtered_reviews.columns:
        filtered_reviews = filtered_reviews[filtered_reviews['rating'] >= min_rating]
    
    if max_rating is not None and 'rating' in filtered_reviews.columns:
        filtered_reviews = filtered_reviews[filtered_reviews['rating'] <= max_rating]
    
    # Extract review text
    review_texts = filtered_reviews['body'].dropna().tolist()
    
    # Extract common terms
    term_counts = extract_common_terms(review_texts, n_terms=n_terms)
    
    # Create word cloud data
    if term_counts:
        word_cloud_data = term_counts
    else:
        word_cloud_data = {}
    
    return {
        'term_counts': term_counts,
        'word_cloud_data': word_cloud_data
    }

def generate_wordcloud_image(word_freq: Dict[str, int], 
                           width: int = 800, 
                           height: int = 400, 
                           max_words: int = 100,
                           background_color: str = 'white') -> Optional[str]:
    """
    Generate a word cloud image from word frequencies.
    
    Args:
        word_freq: Dictionary of word -> frequency
        width: Width of the word cloud image
        height: Height of the word cloud image
        max_words: Maximum number of words to include
        background_color: Background color of the word cloud
        
    Returns:
        Base64-encoded PNG image of the word cloud
    """
    if not word_freq:
        logger.warning("No word frequencies provided for word cloud generation")
        return None
    
    try:
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            relative_scaling=0.5
        ).generate_from_frequencies(word_freq)
        
        # Convert to image
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        
        # Encode to base64
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_data}"
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        return None

def identify_market_gaps(apps_df: pd.DataFrame,
                        apps_categories_df: pd.DataFrame,
                        categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential market gaps based on category saturation and ratings.
    
    Args:
        apps_df: DataFrame with app data
        apps_categories_df: DataFrame with app-category relationships
        categories_df: DataFrame with category data
        
    Returns:
        DataFrame with market gap analysis
    """
    if apps_df.empty or apps_categories_df.empty or categories_df.empty:
        logger.warning("One or more required DataFrames are empty")
        return pd.DataFrame()
    
    # Join categories with apps
    categories_mapping = pd.merge(
        apps_categories_df,
        categories_df,
        left_on='category_id',
        right_on='id',
        how='left'
    )
    
    # Get app counts by category
    category_counts = categories_mapping.groupby('title_y').size().reset_index(name='app_count')
    
    # Get average ratings by category
    if 'rating_value' in apps_df.columns:
        app_ratings = pd.merge(
            categories_mapping,
            apps_df[['id', 'rating_value']],
            left_on='app_id',
            right_on='id',
            how='inner'
        )
        
        category_ratings = app_ratings.groupby('title_y')['rating_value'].agg(
            avg_rating='mean',
            max_rating='max',
            min_rating='min',
            rating_std='std'
        ).reset_index()
        
        # Merge counts and ratings
        category_analysis = pd.merge(
            category_counts,
            category_ratings,
            on='title_y',
            how='left'
        )
    else:
        category_analysis = category_counts
        category_analysis['avg_rating'] = np.nan
        category_analysis['max_rating'] = np.nan
        category_analysis['min_rating'] = np.nan
        category_analysis['rating_std'] = np.nan
    
    # Calculate market gap score
    # Lower app count and higher avg rating might indicate potential market gaps
    if 'avg_rating' in category_analysis.columns and not category_analysis['avg_rating'].isna().all():
        # Normalize app count (inverted, fewer apps = higher score)
        max_count = category_analysis['app_count'].max()
        min_count = category_analysis['app_count'].min()
        category_analysis['app_count_score'] = 1 - ((category_analysis['app_count'] - min_count) / 
                                                 (max_count - min_count + 1e-10))
        
        # Normalize average rating
        max_rating = category_analysis['avg_rating'].max()
        min_rating = category_analysis['avg_rating'].min() 
        category_analysis['rating_score'] = ((category_analysis['avg_rating'] - min_rating) / 
                                         (max_rating - min_rating + 1e-10))
        
        # Calculate opportunity score (weighted average of app count and rating scores)
        category_analysis['opportunity_score'] = (0.7 * category_analysis['app_count_score'] + 
                                               0.3 * category_analysis['rating_score'])
        
        # Sort by opportunity score
        category_analysis = category_analysis.sort_values('opportunity_score', ascending=False)
    
    # Rename column for clarity
    category_analysis = category_analysis.rename(columns={'title_y': 'category'})
    
    return category_analysis 