#!/usr/bin/env python3
"""
Test Installation

This script verifies that the Shopify App Store Analysis framework is installed correctly.
"""

import sys
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
    'sklearn',
    'tqdm',
    'wordcloud',
    'bs4',
    'dotenv'
]

# Required modules
REQUIRED_MODULES = [
    'src.config',
    'src.data.loader',
    'src.data.processor',
    'src.models.schema',
    'src.analysis.stats',
    'src.analysis.insights',
    'src.visualization.plots'
]

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_module(module_name):
    """Check if a module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        logger.error(f"Error importing {module_name}: {e}")
        return False

def main():
    """Main function to test the installation."""
    logger.info("Testing Shopify App Store Analysis framework installation")
    
    # Check Python version
    python_version = sys.version.split()[0]
    logger.info(f"Python version: {python_version}")
    
    if sys.version_info < (3, 8):
        logger.warning("Python 3.8 or higher is recommended")
    
    # Check required packages
    logger.info("Checking required packages...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        if check_package(package):
            logger.info(f"✓ {package}")
        else:
            logger.error(f"✗ {package} - Not found")
            missing_packages.append(package)
    
    # Check required modules
    logger.info("Checking required modules...")
    missing_modules = []
    
    for module in REQUIRED_MODULES:
        if check_module(module):
            logger.info(f"✓ {module}")
        else:
            logger.error(f"✗ {module} - Not found")
            missing_modules.append(module)
    
    # Summary
    if not missing_packages and not missing_modules:
        logger.info("All required packages and modules are available!")
        logger.info("The Shopify App Store Analysis framework is installed correctly.")
        return 0
    else:
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.error("Please install the missing packages using:")
            logger.error(f"pip install {' '.join(missing_packages)}")
        
        if missing_modules:
            logger.error(f"Missing modules: {', '.join(missing_modules)}")
            logger.error("Please check that the package is installed correctly.")
        
        logger.error("The Shopify App Store Analysis framework is not installed correctly.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 