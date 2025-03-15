#!/usr/bin/env python3
"""
Shopify App Store Analysis Script

This script performs a basic analysis of the Shopify App Store data and
generates a report with key insights.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from src.data.loader import (
    load_apps, load_categories, load_apps_categories, 
    load_key_benefits, load_pricing_plans, load_pricing_plan_features,
    process_reviews_in_chunks
)
from src.data.processor import (
    preprocess_apps, preprocess_reviews, get_app_reviews_summary,
    create_master_dataset
)
from src.analysis.stats import (
    get_popular_apps, get_category_distribution, get_rating_distribution,
    get_developer_stats, analyze_pricing_trends
)
from src.visualization.plots import (
    plot_category_distribution, plot_rating_distribution,
    plot_developer_stats, plot_pricing_plan_distribution,
    fig_to_base64
)

def main():
    """Main function to run the analysis."""
    logger.info("Starting Shopify App Store analysis")
    
    # Create output directory for reports
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    apps_df = load_apps()
    categories_df = load_categories()
    apps_categories_df = load_apps_categories()
    key_benefits_df = load_key_benefits()
    pricing_plans_df = load_pricing_plans()
    
    # Check if data was loaded successfully
    if apps_df.empty or categories_df.empty or apps_categories_df.empty:
        logger.error("Failed to load required data. Please check that the CSV files exist in the data directory.")
        sys.exit(1)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    apps_df = preprocess_apps(apps_df)
    
    # Process reviews in chunks to get summary statistics
    logger.info("Processing reviews...")
    
    def process_review_chunk(chunk):
        processed_chunk = preprocess_reviews(chunk)
        return get_app_reviews_summary(processed_chunk)
    
    # Limit to 5 chunks for a quick analysis
    reviews_summary_df = process_reviews_in_chunks(process_review_chunk, max_chunks=5)
    
    # Create master dataset
    logger.info("Creating master dataset...")
    master_df = create_master_dataset(
        apps_df=apps_df,
        apps_categories_df=apps_categories_df,
        categories_df=categories_df,
        reviews_summary_df=reviews_summary_df,
        key_benefits_df=key_benefits_df,
        pricing_plans_df=pricing_plans_df
    )
    
    # Perform analysis
    logger.info("Performing analysis...")
    
    # Get popular apps
    popular_apps = get_popular_apps(master_df, by='reviews_count', n=10)
    
    # Get category distribution
    category_dist = get_category_distribution(apps_df, apps_categories_df, categories_df)
    
    # Get rating distribution
    rating_dist = get_rating_distribution(master_df, rating_col='rating_value')
    
    # Get developer statistics
    dev_stats = get_developer_stats(master_df)
    
    # Analyze pricing trends
    pricing_trends = analyze_pricing_trends(pricing_plans_df)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Plot category distribution
    category_fig = plot_category_distribution(category_dist, top_n=15)
    category_img = fig_to_base64(category_fig)
    plt.close(category_fig)
    
    # Plot rating distribution
    rating_fig = plot_rating_distribution(rating_dist)
    rating_img = fig_to_base64(rating_fig)
    plt.close(rating_fig)
    
    # Plot developer statistics
    dev_fig = plot_developer_stats(dev_stats, top_n=15)
    dev_img = fig_to_base64(dev_fig)
    plt.close(dev_fig)
    
    # Plot pricing plan distribution
    pricing_fig = plot_pricing_plan_distribution(pricing_trends)
    pricing_img = fig_to_base64(pricing_fig)
    plt.close(pricing_fig)
    
    # Generate HTML report
    logger.info("Generating report...")
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shopify App Store Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #004c3f; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Shopify App Store Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Dataset Overview</h2>
        <ul>
            <li>Number of apps: {len(apps_df)}</li>
            <li>Number of categories: {len(categories_df)}</li>
            <li>Number of apps with reviews: {len(reviews_summary_df)}</li>
            <li>Number of apps with key benefits: {len(key_benefits_df['app_id'].unique())}</li>
            <li>Number of apps with pricing plans: {len(pricing_plans_df['app_id'].unique())}</li>
        </ul>
        
        <h2>Top 10 Most Popular Apps (by Reviews Count)</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>App</th>
                <th>Developer</th>
                <th>Rating</th>
                <th>Reviews Count</th>
            </tr>
            {''.join([f"<tr><td>{i+1}</td><td>{row['title']}</td><td>{row['developer']}</td><td>{row.get('rating_value', 'N/A')}</td><td>{row['reviews_count']}</td></tr>" for i, row in popular_apps.iterrows()])}
        </table>
        
        <h2>Category Distribution</h2>
        <div class="figure">
            <img src="{category_img}" alt="Category Distribution">
        </div>
        
        <h2>Rating Distribution</h2>
        <div class="figure">
            <img src="{rating_img}" alt="Rating Distribution">
        </div>
        
        <h2>Top Developers</h2>
        <div class="figure">
            <img src="{dev_img}" alt="Top Developers">
        </div>
        
        <h2>Pricing Plan Distribution</h2>
        <div class="figure">
            <img src="{pricing_img}" alt="Pricing Plan Distribution">
        </div>
        
        <h2>Key Insights</h2>
        <ul>
            <li>The most popular app category is <strong>{category_dist.iloc[0].name}</strong> with {category_dist.iloc[0]['count']} apps ({category_dist.iloc[0]['percentage']:.1f}% of all apps).</li>
            <li>The most prolific developer is <strong>{dev_stats.iloc[0]['developer']}</strong> with {dev_stats.iloc[0]['app_count']} apps.</li>
            <li>The most common pricing plan type is <strong>{pricing_trends.iloc[0]['plan_type']}</strong> ({pricing_trends.iloc[0]['percentage']:.1f}% of all plans).</li>
        </ul>
    </body>
    </html>
    """
    
    # Save report
    report_path = os.path.join(output_dir, f"shopify_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(report_path, 'w') as f:
        f.write(report_html)
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 