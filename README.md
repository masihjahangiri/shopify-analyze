# Shopify App Store Analysis Framework

A comprehensive Python framework for analyzing Shopify App Store data.

## Overview

This framework provides tools for loading, preprocessing, analyzing, and visualizing Shopify App Store data. It's designed to handle large datasets efficiently and provide meaningful insights into the Shopify app ecosystem.

## Dataset

The framework is designed to work with the following CSV files:

1. **`apps.csv`** (48.63 MB)
   - Contains information about apps (id, url, title, developer, etc.)
   - 11,951 unique apps

2. **`apps_categories.csv`** (7.87 MB)
   - Join table linking apps to categories

3. **`categories.csv`** (89.19 kB)
   - Contains category information (id, title)
   - 1,889 unique categories

4. **`key_benefits.csv`** (4.93 MB)
   - Highlights main "key benefits" of each app
   - 11,564 unique apps with benefits

5. **`pricing_plan_features.csv`** (7.59 MB)
   - Features included in each pricing plan
   - 17,576 unique pricing plans

6. **`pricing_plans.csv`** (1.98 MB)
   - Each app may have multiple pricing plans
   - 9,679 unique apps with pricing plans

7. **`reviews.csv`** (380.39 MB)
   - User reviews of apps
   - 7,937 unique apps with reviews

## Project Structure

```
shopify_analyze/
├── data/                  # Data directory (place CSV files here)
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── analysis/          # Analysis modules
│   │   ├── insights.py    # Advanced insights extraction
│   │   └── stats.py       # Statistical analysis functions
│   ├── data/              # Data handling modules
│   │   ├── loader.py      # Data loading functions
│   │   └── processor.py   # Data preprocessing functions
│   ├── models/            # Data models
│   │   └── schema.py      # Schema definitions
│   ├── visualization/     # Visualization modules
│   │   └── plots.py       # Plotting functions
│   └── config.py          # Configuration settings
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/shopify_analyze.git
   cd shopify_analyze
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place the CSV files in the `data/` directory.

## Usage

### Basic Usage

```python
from src.data.loader import load_apps, load_categories, load_apps_categories
from src.data.processor import preprocess_apps, get_apps_with_categories
from src.analysis.stats import get_category_distribution
from src.visualization.plots import plot_category_distribution

# Load data
apps_df = load_apps()
categories_df = load_categories()
apps_categories_df = load_apps_categories()

# Preprocess data
apps_df = preprocess_apps(apps_df)

# Analyze data
category_dist = get_category_distribution(apps_df, apps_categories_df, categories_df)

# Visualize data
fig = plot_category_distribution(category_dist, top_n=15)
fig.show()
```

### Using the ShopifyDataset Class

```python
from src.data.loader import load_apps, load_categories, load_apps_categories
from src.data.processor import preprocess_apps
from src.models.schema import ShopifyDataset

# Load and preprocess data
apps_df = preprocess_apps(load_apps())
categories_df = load_categories()
apps_categories_df = load_apps_categories()

# Create dataset
dataset = ShopifyDataset()
dataset.load_from_dataframes(
    apps_df=apps_df,
    categories_df=categories_df,
    apps_categories_df=apps_categories_df
)

# Access data
app = dataset.get_app(12345)  # Get app by ID
category_apps = dataset.get_apps_by_category("Marketing")  # Get apps in a category
developer_apps = dataset.get_apps_by_developer("Shopify")  # Get apps by a developer
```

## Example Notebooks

Check out the notebooks in the `notebooks/` directory for examples of how to use the framework:

- `shopify_analysis_demo.ipynb`: Demonstrates basic usage of the framework

## Features

- **Data Loading**: Efficiently load and process large CSV files
- **Data Preprocessing**: Clean and transform raw data
- **Statistical Analysis**: Calculate various statistics and metrics
- **Advanced Insights**: Extract meaningful insights from the data
- **Visualization**: Create static and interactive visualizations
- **Object-Oriented Interface**: Work with the data in an object-oriented way

## License

This project is licensed under the MIT License - see the LICENSE file for details. 