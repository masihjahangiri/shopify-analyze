import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

# Set default styling for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Default figure size
DEFAULT_FIG_SIZE = (12, 8)
DEFAULT_DPI = 100

def plot_category_distribution(category_counts: pd.DataFrame, 
                             top_n: int = 20,
                             figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot distribution of apps across categories.
    
    Args:
        category_counts: DataFrame with category counts
        top_n: Number of top categories to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if category_counts.empty or 'count' not in category_counts.columns:
        logger.warning("Invalid category counts data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Get top N categories
    if len(category_counts) > top_n:
        plot_data = category_counts.head(top_n).copy()
        plot_data = plot_data.sort_values('count')  # Sort for horizontal bar chart
    else:
        plot_data = category_counts.copy().sort_values('count')
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(plot_data.index, plot_data['count'], color=sns.color_palette('viridis', len(plot_data)))
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = plot_data['percentage'].iloc[i] if 'percentage' in plot_data.columns else (width / category_counts['count'].sum() * 100)
        ax.text(width + (width * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{width:,.0f} ({percentage:.1f}%)', 
                va='center')
    
    # Set labels and title
    ax.set_xlabel('Number of Apps')
    ax.set_ylabel('Category')
    ax.set_title(f'Top {top_n} App Categories')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_rating_distribution(rating_dist: pd.DataFrame,
                           figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot distribution of app ratings.
    
    Args:
        rating_dist: DataFrame with rating distribution
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if rating_dist.empty or 'count' not in rating_dist.columns:
        logger.warning("Invalid rating distribution data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    bars = ax.bar(rating_dist.index.astype(str), rating_dist['count'], 
                 color=sns.color_palette('viridis', len(rating_dist)))
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (height * 0.01), 
                f'{height:,.0f}', 
                ha='center')
    
    # Set labels and title
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Apps')
    ax.set_title('Distribution of App Ratings')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_time_series(time_series_df: pd.DataFrame,
                    y_cols: List[str] = None,
                    title: str = 'Time Series Analysis',
                    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        time_series_df: DataFrame with time series data
        y_cols: List of columns to plot (if None, all numeric columns except date)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if time_series_df.empty or 'date' not in time_series_df.columns:
        logger.warning("Invalid time series data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Determine columns to plot
    if y_cols is None:
        y_cols = [col for col in time_series_df.columns 
                 if col != 'date' and pd.api.types.is_numeric_dtype(time_series_df[col])]
    
    if not y_cols:
        logger.warning("No numeric columns found for time series plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No numeric data available", ha='center', va='center')
        return fig
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each column
    for col in y_cols:
        if col in time_series_df.columns:
            ax.plot(time_series_df['date'], time_series_df[col], marker='o', label=col)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_developer_stats(dev_stats: pd.DataFrame,
                        top_n: int = 20,
                        figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot statistics about app developers.
    
    Args:
        dev_stats: DataFrame with developer statistics
        top_n: Number of top developers to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if dev_stats.empty or 'app_count' not in dev_stats.columns:
        logger.warning("Invalid developer statistics data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Get top N developers by app count
    if len(dev_stats) > top_n:
        plot_data = dev_stats.head(top_n).copy()
        plot_data = plot_data.sort_values('app_count')  # Sort for horizontal bar chart
    else:
        plot_data = dev_stats.copy().sort_values('app_count')
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(plot_data['developer'], plot_data['app_count'], 
                  color=sns.color_palette('viridis', len(plot_data)))
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{width:,.0f}', 
                va='center')
    
    # Set labels and title
    ax.set_xlabel('Number of Apps')
    ax.set_ylabel('Developer')
    ax.set_title(f'Top {top_n} App Developers by Number of Apps')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_sentiment_distribution(sentiment_data: pd.DataFrame,
                              figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot distribution of review sentiments.
    
    Args:
        sentiment_data: DataFrame with sentiment counts
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_data.empty:
        logger.warning("Invalid sentiment data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Check if we have percentage columns
    has_percentages = all(f'{sentiment}_pct' in sentiment_data.columns 
                         for sentiment in ['positive', 'neutral', 'negative'])
    
    # Aggregate data
    if has_percentages:
        # Use percentages
        agg_data = pd.DataFrame({
            'sentiment': ['Positive', 'Neutral', 'Negative'],
            'percentage': [
                sentiment_data['positive_pct'].mean(),
                sentiment_data['neutral_pct'].mean(),
                sentiment_data['negative_pct'].mean()
            ]
        })
    else:
        # Use counts
        total_positive = sentiment_data['positive'].sum() if 'positive' in sentiment_data.columns else 0
        total_neutral = sentiment_data['neutral'].sum() if 'neutral' in sentiment_data.columns else 0
        total_negative = sentiment_data['negative'].sum() if 'negative' in sentiment_data.columns else 0
        
        total = total_positive + total_neutral + total_negative
        
        agg_data = pd.DataFrame({
            'sentiment': ['Positive', 'Neutral', 'Negative'],
            'percentage': [
                (total_positive / total) * 100 if total > 0 else 0,
                (total_neutral / total) * 100 if total > 0 else 0,
                (total_negative / total) * 100 if total > 0 else 0
            ]
        })
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(
        agg_data['percentage'], 
        labels=agg_data['sentiment'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Set title
    ax.set_title('Distribution of Review Sentiments')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_pricing_plan_distribution(plan_counts: pd.DataFrame,
                                 figsize: Tuple[int, int] = DEFAULT_FIG_SIZE) -> plt.Figure:
    """
    Plot distribution of pricing plans.
    
    Args:
        plan_counts: DataFrame with pricing plan counts
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if plan_counts.empty or 'count' not in plan_counts.columns:
        logger.warning("Invalid pricing plan data")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Sort data
    plot_data = plan_counts.sort_values('count', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    bars = ax.bar(plot_data['plan_type'], plot_data['count'], 
                 color=sns.color_palette('viridis', len(plot_data)))
    
    # Add count and percentage labels
    for bar in bars:
        height = bar.get_height()
        percentage = height / plan_counts['count'].sum() * 100
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (height * 0.01), 
                f'{height:,.0f}\n({percentage:.1f}%)', 
                ha='center')
    
    # Set labels and title
    ax.set_xlabel('Plan Type')
    ax.set_ylabel('Number of Plans')
    ax.set_title('Distribution of Pricing Plans')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_interactive_category_plot(category_counts: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Create an interactive plotly bar chart of category distribution.
    
    Args:
        category_counts: DataFrame with category counts
        top_n: Number of top categories to show
        
    Returns:
        Plotly figure
    """
    if category_counts.empty or 'count' not in category_counts.columns:
        logger.warning("Invalid category counts data")
        fig = go.Figure()
        fig.add_annotation(text="No data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get top N categories
    if len(category_counts) > top_n:
        plot_data = category_counts.head(top_n).copy()
    else:
        plot_data = category_counts.copy()
    
    # Sort for horizontal bar chart
    plot_data = plot_data.sort_values('count')
    
    # Create hover text
    if 'percentage' in plot_data.columns:
        hover_text = [f"Category: {cat}<br>Count: {count:,}<br>Percentage: {pct:.2f}%" 
                     for cat, count, pct in zip(plot_data.index, 
                                               plot_data['count'], 
                                               plot_data['percentage'])]
    else:
        hover_text = [f"Category: {cat}<br>Count: {count:,}" 
                     for cat, count in zip(plot_data.index, plot_data['count'])]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        y=plot_data.index,
        x=plot_data['count'],
        orientation='h',
        text=plot_data['count'],
        textposition='outside',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(
            color=plot_data['count'],
            colorscale='Viridis',
            colorbar=dict(title="Count")
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Top {top_n} App Categories',
        xaxis_title='Number of Apps',
        yaxis_title='Category',
        height=600,
        margin=dict(l=100, r=20, t=50, b=50)
    )
    
    return fig

def create_interactive_time_series(time_series_df: pd.DataFrame,
                                 y_cols: List[str] = None,
                                 title: str = 'Time Series Analysis') -> go.Figure:
    """
    Create an interactive plotly time series plot.
    
    Args:
        time_series_df: DataFrame with time series data
        y_cols: List of columns to plot (if None, all numeric columns except date)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if time_series_df.empty or 'date' not in time_series_df.columns:
        logger.warning("Invalid time series data")
        fig = go.Figure()
        fig.add_annotation(text="No data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Determine columns to plot
    if y_cols is None:
        y_cols = [col for col in time_series_df.columns 
                 if col != 'date' and pd.api.types.is_numeric_dtype(time_series_df[col])]
    
    if not y_cols:
        logger.warning("No numeric columns found for time series plot")
        fig = go.Figure()
        fig.add_annotation(text="No numeric data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add each column as a trace
    for col in y_cols:
        if col in time_series_df.columns:
            fig.add_trace(go.Scatter(
                x=time_series_df['date'],
                y=time_series_df[col],
                mode='lines+markers',
                name=col
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def fig_to_base64(fig: Union[plt.Figure, go.Figure]) -> str:
    """
    Convert a matplotlib or plotly figure to base64 encoded string.
    
    Args:
        fig: Matplotlib or Plotly figure
        
    Returns:
        Base64 encoded string of the figure
    """
    buffer = BytesIO()
    
    if isinstance(fig, plt.Figure):
        fig.savefig(buffer, format='png', dpi=DEFAULT_DPI, bbox_inches='tight')
    elif isinstance(fig, go.Figure):
        fig.write_image(buffer, format='png')
    else:
        logger.error(f"Unsupported figure type: {type(fig)}")
        return ""
    
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{image_data}" 