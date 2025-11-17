from setuptools import setup, find_packages

setup(
    name="portfolio-mpt",
    version="0.1.0",
    description="Portfolio analysis & mean-variance optimization utilities",
    packages=find_packages(
        exclude=(
            "tests", 
            "docs", 
            "examples", 
            "streamlit_app"
        )
    ),
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.40",
        "pandas>=2.2",
        "numpy>=1.26",
        "pyarrow>=17.0.0",
        "matplotlib>=3.8",
        "scipy>=1.11",
        "cvxpy>=1.5.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0", 
            "pytest-cov>=5.0", 
            "ruff>=0.6"
        ]
    },
    entry_points={
        "console_scripts": [
            "portfolio-mpt=portfolio_mpt.cli:main"
        ]
    },
    include_package_data=True,
)
