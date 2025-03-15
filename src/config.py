import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv('DATA_DIR', PROJECT_ROOT / 'data')

# Dataset file paths
APPS_CSV = os.path.join(DATA_DIR, 'apps.csv')
APPS_CATEGORIES_CSV = os.path.join(DATA_DIR, 'apps_categories.csv')
CATEGORIES_CSV = os.path.join(DATA_DIR, 'categories.csv')
KEY_BENEFITS_CSV = os.path.join(DATA_DIR, 'key_benefits.csv')
PRICING_PLAN_FEATURES_CSV = os.path.join(DATA_DIR, 'pricing_plan_features.csv')
PRICING_PLANS_CSV = os.path.join(DATA_DIR, 'pricing_plans.csv')
REVIEWS_CSV = os.path.join(DATA_DIR, 'reviews.csv')

# Settings
RANDOM_SEED = 42
CHUNK_SIZE = 10000  # For processing large files in chunks 