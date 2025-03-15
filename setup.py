from setuptools import setup, find_packages

setup(
    name="shopify_analyze",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "plotly>=5.14.1",
        "scikit-learn>=1.2.2",
        "jupyter>=1.0.0",
        "notebook>=6.5.4",
        "ipywidgets>=8.0.6",
        "tqdm>=4.65.0",
        "wordcloud>=1.9.2",
        "beautifulsoup4>=4.12.2",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for analyzing Shopify App Store data",
    keywords="shopify, data analysis, visualization",
    python_requires=">=3.8",
) 