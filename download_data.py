#!/usr/bin/env python3
"""
Download Shopify App Store Data

This script downloads the Shopify App Store dataset files from a specified URL.
"""

import os
import sys
import logging
import requests
from tqdm import tqdm
import zipfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data files to download
DATA_FILES = [
    'apps.csv',
    'apps_categories.csv',
    'categories.csv',
    'key_benefits.csv',
    'pricing_plan_features.csv',
    'pricing_plans.csv',
    'reviews.csv'
]

# Replace this with the actual URL where the data is hosted
# This is a placeholder URL
DATA_URL = "https://example.com/shopify_app_store_data/"

def download_file(url, destination):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
                
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_data_files():
    """Download all data files."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Downloading data files to {data_dir}...")
    
    success_count = 0
    for file_name in DATA_FILES:
        file_url = f"{DATA_URL}{file_name}"
        file_path = os.path.join(data_dir, file_name)
        
        logger.info(f"Downloading {file_name}...")
        if download_file(file_url, file_path):
            success_count += 1
        
    logger.info(f"Downloaded {success_count} of {len(DATA_FILES)} files.")
    
    if success_count == len(DATA_FILES):
        logger.info("All files downloaded successfully!")
    else:
        logger.warning(f"Failed to download {len(DATA_FILES) - success_count} files.")

def download_zip_file(zip_url, extract_dir):
    """Download and extract a zip file containing all data."""
    temp_zip = "temp_data.zip"
    
    logger.info(f"Downloading data archive from {zip_url}...")
    if download_file(zip_url, temp_zip):
        logger.info("Extracting files...")
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Files extracted to {extract_dir}")
            os.remove(temp_zip)
            return True
        except zipfile.BadZipFile:
            logger.error("The downloaded file is not a valid zip file.")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            return False
    else:
        logger.error("Failed to download the data archive.")
        return False

def main():
    """Main function to download the data."""
    logger.info("Starting data download")
    
    # Check if data directory exists
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data files already exist
    existing_files = [f for f in DATA_FILES if os.path.exists(os.path.join(data_dir, f))]
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing data files.")
        response = input("Do you want to re-download these files? (y/n): ")
        if response.lower() != 'y':
            logger.info("Download canceled.")
            return
    
    # Ask user which download method to use
    print("\nDownload options:")
    print("1. Download individual files")
    print("2. Download zip archive (recommended for large datasets)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        download_data_files()
    elif choice == '2':
        zip_url = input("Enter the URL of the zip archive: ")
        download_zip_file(zip_url, data_dir)
    else:
        logger.error("Invalid choice. Please enter 1 or 2.")
        return
    
    # Verify downloaded files
    missing_files = [f for f in DATA_FILES if not os.path.exists(os.path.join(data_dir, f))]
    if missing_files:
        logger.warning(f"The following files are still missing: {', '.join(missing_files)}")
        logger.warning("You may need to download these files manually.")
    else:
        logger.info("All required data files are now available!")
        logger.info("You can now run the analysis scripts.")

if __name__ == "__main__":
    main() 