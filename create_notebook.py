import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell - Introduction
nb['cells'] = [nbf.v4.new_markdown_cell("""# Shopify App Store Analysis Demo

This notebook demonstrates how to use the Shopify App Store analysis tools to explore and visualize the dataset.""")]

# Add markdown cell - Setup
nb['cells'].append(nbf.v4.new_markdown_cell("""## Setup

First, let's import the necessary modules and set up our environment."""))

# Add code cell - Setup
nb['cells'].append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
import warnings

# Add the project root to the path so we can import our modules
sys.path.append('..')

# Configure plotting
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

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
    plot_developer_stats, plot_pricing_plan_distribution
)"""))

# Add markdown cell - Loading Data
nb['cells'].append(nbf.v4.new_markdown_cell("""## 1. Loading the Data

Let's load the various datasets we'll be working with."""))

# Add code cell - Loading Data
nb['cells'].append(nbf.v4.new_code_cell("""# Load the main datasets
apps_df = load_apps()
categories_df = load_categories()
apps_categories_df = load_apps_categories()
key_benefits_df = load_key_benefits()
pricing_plans_df = load_pricing_plans()
pricing_plan_features_df = load_pricing_plan_features()

# Display basic info about the datasets
print(f"Apps dataset: {apps_df.shape[0]} rows, {apps_df.shape[1]} columns")
print(f"Categories dataset: {categories_df.shape[0]} rows, {categories_df.shape[1]} columns")
print(f"Apps-Categories dataset: {apps_categories_df.shape[0]} rows, {apps_categories_df.shape[1]} columns")
print(f"Key Benefits dataset: {key_benefits_df.shape[0]} rows, {key_benefits_df.shape[1]} columns")
print(f"Pricing Plans dataset: {pricing_plans_df.shape[0]} rows, {pricing_plans_df.shape[1]} columns")
print(f"Pricing Plan Features dataset: {pricing_plan_features_df.shape[0]} rows, {pricing_plan_features_df.shape[1]} columns")"""))

# Add markdown cell - Exploring Apps Dataset
nb['cells'].append(nbf.v4.new_markdown_cell("""## 2. Exploring the Apps Dataset

Let's take a look at the structure of the apps dataset."""))

# Add code cells - Exploring Apps Dataset
nb['cells'].append(nbf.v4.new_code_cell("""# Display the first few rows of the apps dataset
apps_df.head()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Get a summary of the apps dataset
apps_df.info()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Preprocess the apps dataset
apps_df = preprocess_apps(apps_df)

# Check for missing values
missing_values = apps_df.isnull().sum()
print("Missing values in apps dataset:")
print(missing_values[missing_values > 0])"""))

# Add markdown cell - Exploring Categories
nb['cells'].append(nbf.v4.new_markdown_cell("""## 3. Exploring Categories

Let's look at the categories and how apps are distributed among them."""))

# Add code cells - Exploring Categories
nb['cells'].append(nbf.v4.new_code_cell("""# Display the categories
categories_df.head(10)"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Get the distribution of apps across categories
category_dist = get_category_distribution(apps_df, apps_categories_df, categories_df)

# Display the top 15 categories
category_dist.head(15)"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize the category distribution
fig = plot_category_distribution(category_dist, top_n=15)
plt.tight_layout()
plt.show()"""))

# Add markdown cell - Processing Reviews
nb['cells'].append(nbf.v4.new_markdown_cell("""## 4. Processing Reviews

The reviews dataset is large, so we'll process it in chunks."""))

# Add code cell - Processing Reviews
nb['cells'].append(nbf.v4.new_code_cell("""# Define a function to process review chunks
def process_review_chunk(chunk):
    processed_chunk = preprocess_reviews(chunk)
    return get_app_reviews_summary(processed_chunk)

# Process reviews in chunks (limit to 3 chunks for demonstration)
reviews_summary_df = process_reviews_in_chunks(process_review_chunk, max_chunks=3)

# Display the summary
print(f"Processed reviews for {len(reviews_summary_df)} apps")
reviews_summary_df.head()"""))

# Add markdown cell - Creating Master Dataset
nb['cells'].append(nbf.v4.new_markdown_cell("""## 5. Creating a Master Dataset

Let's combine all the data into a master dataset for analysis."""))

# Add code cell - Creating Master Dataset
nb['cells'].append(nbf.v4.new_code_cell("""# Create the master dataset
master_df = create_master_dataset(
    apps_df=apps_df,
    apps_categories_df=apps_categories_df,
    categories_df=categories_df,
    reviews_summary_df=reviews_summary_df,
    key_benefits_df=key_benefits_df,
    pricing_plans_df=pricing_plans_df
)

# Display the master dataset
print(f"Master dataset: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
master_df.head()"""))

# Add markdown cell - Popular Apps Analysis
nb['cells'].append(nbf.v4.new_markdown_cell("""## 6. Popular Apps Analysis

Let's identify and analyze the most popular apps in the store."""))

# Add code cells - Popular Apps Analysis
nb['cells'].append(nbf.v4.new_code_cell("""# Get the most popular apps by reviews count
popular_apps = get_popular_apps(master_df, by='reviews_count', n=10)
popular_apps"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Get the highest rated apps (with at least 10 reviews)
high_rated_apps = master_df[master_df['reviews_count'] >= 10].sort_values('rating_value', ascending=False)
high_rated_apps[['title', 'developer', 'rating_value', 'reviews_count']].head(10)"""))

# Add markdown cell - Rating Distribution Analysis
nb['cells'].append(nbf.v4.new_markdown_cell("""## 7. Rating Distribution Analysis"""))

# Add code cells - Rating Distribution Analysis
nb['cells'].append(nbf.v4.new_code_cell("""# Get the rating distribution
rating_dist = get_rating_distribution(master_df, rating_col='rating_value')
rating_dist"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize the rating distribution
fig = plot_rating_distribution(rating_dist)
plt.tight_layout()
plt.show()"""))

# Add markdown cell - Developer Analysis
nb['cells'].append(nbf.v4.new_markdown_cell("""## 8. Developer Analysis

Let's analyze the developers in the Shopify App Store."""))

# Add code cells - Developer Analysis
nb['cells'].append(nbf.v4.new_code_cell("""# Get developer statistics
dev_stats = get_developer_stats(master_df)
dev_stats.head(10)"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize the top developers
fig = plot_developer_stats(dev_stats, top_n=15)
plt.tight_layout()
plt.show()"""))

# Add markdown cell - Pricing Analysis
nb['cells'].append(nbf.v4.new_markdown_cell("""## 9. Pricing Analysis

Let's analyze the pricing plans of apps in the Shopify App Store."""))

# Add code cells - Pricing Analysis
nb['cells'].append(nbf.v4.new_code_cell("""# Analyze pricing trends
pricing_trends = analyze_pricing_trends(pricing_plans_df)
pricing_trends"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize pricing plan distribution
fig = plot_pricing_plan_distribution(pricing_trends)
plt.tight_layout()
plt.show()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Analyze pricing by category
# First, get the primary category for each app
app_primary_category = apps_categories_df.merge(categories_df, on='category_id')
app_primary_category = app_primary_category.groupby('app_id').first().reset_index()

# Merge with pricing plans
pricing_by_category = pricing_plans_df.merge(app_primary_category, on='app_id')

# Calculate average price by category for paid plans
paid_pricing = pricing_by_category[pricing_by_category['price'] > 0]
avg_price_by_category = paid_pricing.groupby('title')['price'].agg(['mean', 'median', 'count'])
avg_price_by_category = avg_price_by_category.sort_values('mean', ascending=False)

# Display the top 15 most expensive categories on average
avg_price_by_category.head(15)"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize average price by category (top 15)
plt.figure(figsize=(14, 8))
top_categories = avg_price_by_category.head(15).reset_index()
sns.barplot(x='mean', y='title', data=top_categories)
plt.title('Average Price by Category (Top 15 Most Expensive)')
plt.xlabel('Average Price ($)')
plt.ylabel('Category')
plt.tight_layout()
plt.show()"""))

# Add markdown cell - Key Benefits Analysis
nb['cells'].append(nbf.v4.new_markdown_cell("""## 10. Key Benefits Analysis

Let's analyze the key benefits of apps in the Shopify App Store."""))

# Add code cells - Key Benefits Analysis
nb['cells'].append(nbf.v4.new_code_cell("""# Display the key benefits dataset
key_benefits_df.head()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Count the number of key benefits per app
benefits_count = key_benefits_df.groupby('app_id').size().reset_index(name='benefits_count')
benefits_count.describe()"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Merge with apps data to see if more benefits correlate with popularity
benefits_analysis = benefits_count.merge(master_df[['app_id', 'title', 'reviews_count', 'rating_value']], on='app_id')

# Calculate correlation
correlation = benefits_analysis[['benefits_count', 'reviews_count', 'rating_value']].corr()
correlation"""))

nb['cells'].append(nbf.v4.new_code_cell("""# Visualize the relationship between number of benefits and reviews count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='benefits_count', y='reviews_count', data=benefits_analysis)
plt.title('Relationship Between Number of Key Benefits and Reviews Count')
plt.xlabel('Number of Key Benefits')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()"""))

# Add markdown cell - Conclusion
nb['cells'].append(nbf.v4.new_markdown_cell("""## 11. Conclusion and Key Insights

Let's summarize our findings from the Shopify App Store analysis."""))

# Add code cell - Conclusion
nb['cells'].append(nbf.v4.new_code_cell("""# Generate key insights
insights = [
    f"The Shopify App Store contains {len(apps_df)} apps across {len(categories_df)} categories.",
    f"The most popular app category is {category_dist.iloc[0].name} with {category_dist.iloc[0]['count']} apps ({category_dist.iloc[0]['percentage']:.1f}% of all apps).",
    f"The most prolific developer is {dev_stats.iloc[0]['developer']} with {dev_stats.iloc[0]['app_count']} apps.",
    f"The most common pricing plan type is {pricing_trends.iloc[0]['plan_type']} ({pricing_trends.iloc[0]['percentage']:.1f}% of all plans).",
    f"The average rating across all apps with reviews is {master_df['rating_value'].mean():.2f} out of 5.",
    f"The most reviewed app is '{popular_apps.iloc[0]['title']}' with {popular_apps.iloc[0]['reviews_count']} reviews."
]

# Display insights
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")"""))

# Add markdown cell - Next Steps
nb['cells'].append(nbf.v4.new_markdown_cell("""## 12. Next Steps

Here are some potential next steps for further analysis:

1. Perform sentiment analysis on app reviews to understand customer satisfaction
2. Analyze the relationship between pricing strategies and app popularity
3. Investigate seasonal trends in app installations and reviews
4. Build a recommendation system for Shopify merchants based on their store characteristics
5. Analyze the impact of app updates on ratings and reviews"""))

# Write the notebook to a file
with open('notebooks/shopify_analysis_demo.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
