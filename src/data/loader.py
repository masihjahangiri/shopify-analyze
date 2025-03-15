import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Callable, Iterator
import os
import logging
from tqdm import tqdm

from src.config import (
    APPS_CSV, 
    APPS_CATEGORIES_CSV, 
    CATEGORIES_CSV,
    KEY_BENEFITS_CSV,
    PRICING_PLAN_FEATURES_CSV,
    PRICING_PLANS_CSV,
    REVIEWS_CSV,
    CHUNK_SIZE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return False
    return True

def load_apps(usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load apps.csv dataset."""
    if not check_file_exists(APPS_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading apps data from {APPS_CSV}")
    return pd.read_csv(APPS_CSV, usecols=usecols)

def load_categories() -> pd.DataFrame:
    """Load categories.csv dataset."""
    if not check_file_exists(CATEGORIES_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading categories data from {CATEGORIES_CSV}")
    return pd.read_csv(CATEGORIES_CSV)

def load_apps_categories() -> pd.DataFrame:
    """Load apps_categories.csv dataset."""
    if not check_file_exists(APPS_CATEGORIES_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading app categories data from {APPS_CATEGORIES_CSV}")
    return pd.read_csv(APPS_CATEGORIES_CSV)

def load_key_benefits() -> pd.DataFrame:
    """Load key_benefits.csv dataset."""
    if not check_file_exists(KEY_BENEFITS_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading key benefits data from {KEY_BENEFITS_CSV}")
    return pd.read_csv(KEY_BENEFITS_CSV)

def load_pricing_plans() -> pd.DataFrame:
    """Load pricing_plans.csv dataset."""
    if not check_file_exists(PRICING_PLANS_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading pricing plans data from {PRICING_PLANS_CSV}")
    return pd.read_csv(PRICING_PLANS_CSV)

def load_pricing_plan_features() -> pd.DataFrame:
    """Load pricing_plan_features.csv dataset."""
    if not check_file_exists(PRICING_PLAN_FEATURES_CSV):
        return pd.DataFrame()
    
    logger.info(f"Loading pricing plan features data from {PRICING_PLAN_FEATURES_CSV}")
    return pd.read_csv(PRICING_PLAN_FEATURES_CSV)

def load_reviews_iterator(chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """
    Load reviews.csv dataset in chunks to handle large file size.
    Returns an iterator of DataFrame chunks.
    """
    if not check_file_exists(REVIEWS_CSV):
        return iter([pd.DataFrame()])  # Empty iterator
    
    logger.info(f"Loading reviews data in chunks from {REVIEWS_CSV}")
    return pd.read_csv(REVIEWS_CSV, chunksize=chunksize)

def process_reviews_in_chunks(
    processor_func: Callable[[pd.DataFrame], Union[pd.DataFrame, Dict]],
    max_chunks: Optional[int] = None
) -> Union[pd.DataFrame, Dict]:
    """
    Process reviews data in chunks with a custom processor function.
    
    Args:
        processor_func: Function that takes a DataFrame chunk and returns processed data
        max_chunks: Maximum number of chunks to process (None = all)
        
    Returns:
        Combined result of all processed chunks
    """
    results = []
    chunks_iterator = load_reviews_iterator()
    
    # Get total file size for progress bar
    file_size = os.path.getsize(REVIEWS_CSV)
    processed_bytes = 0
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing reviews") as pbar:
        for i, chunk in enumerate(chunks_iterator):
            if max_chunks is not None and i >= max_chunks:
                break
                
            # Process chunk
            result = processor_func(chunk)
            results.append(result)
            
            # Update progress bar based on approximate bytes processed
            chunk_size = chunk.memory_usage(deep=True).sum()
            processed_bytes += chunk_size
            pbar.update(chunk_size)
    
    # Combine results - handle different return types
    if isinstance(results[0], pd.DataFrame):
        return pd.concat(results, ignore_index=True)
    elif isinstance(results[0], dict):
        combined = {}
        for r in results:
            for k, v in r.items():
                if k in combined:
                    # Combine values based on their type
                    if isinstance(v, (int, float)):
                        combined[k] += v
                    elif isinstance(v, list):
                        combined[k].extend(v)
                    elif isinstance(v, pd.DataFrame):
                        combined[k] = pd.concat([combined[k], v], ignore_index=True)
                else:
                    combined[k] = v
        return combined
    else:
        return results

def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all datasets except reviews (which should be processed in chunks)."""
    return {
        'apps': load_apps(),
        'categories': load_categories(),
        'apps_categories': load_apps_categories(),
        'key_benefits': load_key_benefits(),
        'pricing_plans': load_pricing_plans(),
        'pricing_plan_features': load_pricing_plan_features()
    } 