# portfolio-mpt  
*A Python package for Modern Portfolio Theory, optimization, and interactive visualization*

`portfolio-mpt` is a Python package for fetching stock data, computing returns, estimating expected returns & covariance matrices, constructing optimized portfolios (GMV, Max Sharpe, Efficient Frontier), cleaning price data, and visualizing results.

It includes:

- A fully interactive **Streamlit dashboard**
- A modular Python package designed for reuse
- Automated data cleaning pipeline
- Quarto documentation website
- A command-line tool for quick data downloads  

---

## Features

### **Data Module**
- Fetch historical data using Yahoo Finance  
- Save raw parquet + JSON manifests  
- Automatically clean stored price data  
- Load raw or cleaned datasets  

### **Wrangling**
- Compute simple, log, or excess returns  
- Compute cumulative & rolling returns  
- Convert between simple/log-return formats  

### **Analysis**
- Expected return estimation (`mean`, `EMA`)  
- Covariance estimation (`sample`, `Ledoit-Wolf`)  
- Portfolio metrics: return, volatility, Sharpe ratio  

### **Optimization**
- Global Minimum Variance (GMV) portfolio  
- Maximum Sharpe Ratio portfolio  
- Efficient Frontier (target return or risk-aversion based)  
- Smooth convex frontier using risk-aversion sweep  

### **Visualization**
- Price and return line charts  
- Efficient Frontier (Altair scatterplot)  
- Capital Market Line (CML)  
- Pie charts of allocations  
- Simulated growth curves (Monte Carlo)  

### **Data Cleaning**
- Detect & fix timestamp issues  
- Remove duplicates  
- Reindex daily & forward/backward fill gaps  
- Save cleaned parquet files + manifests  

### **Streamlit Dashboard**
The dashboard includes:
- Price data exploration  
- Returns visualization  
- Expected returns & covariance  
- Portfolio optimization  
- Efficient frontier plot  
- Portfolio weights (tables + pie charts)  
- Monte Carlo simulation  

Launch locally:

```bash
streamlit run streamlit_app.py
```

---

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/aydelottetaylor/portfolio-mpt"
```

---

## Quickstart Example

```python
import portfolio_mpt as pm
from portfolio_mpt import data, wrangle, analysis, optimize

tickers = ["AAPL", "MSFT", "NVDA"]

# Fetch + clean data
spec = data.FetchSpec(tickers, "2023-01-01", "2024-01-01")
prices = data.fetch_prices(spec, clean=True)

# Compute returns
rets = wrangle.to_returns(prices)

# Estimate parameters
mu = analysis.expected_returns(rets)
Sigma = analysis.covariance(rets)

# Optimize
gmv = optimize.global_min_variance(mu, Sigma)
ms  = optimize.max_sharpe(mu, Sigma)

print("GMV weights:", gmv.weights)
print("Max Sharpe weights:", ms.weights)
```

---

## Documentation

Full documentation website (Quarto):

**https://<aydelottetaylor>.github.io/portfolio-mpt/**

Includes:

- Tutorial  
- API Reference  
- Examples  
- Project explanation  

---

## Streamlit Application

View the optimized portfolios interactively:

```bash
streamlit run streamlit_app.py
```

Includes:

- Ticker selection  
- Sector filtering  
- Price charts  
- Return charts  
- Efficient frontier plot  
- Capital Market Line  
- Portfolio weights  
- Monte Carlo simulation  

Access the live streamlit app at **https://portfolio-mpt.streamlit.app/**

---

## Tests

Tests cover:

- Data pipeline  
- Return transformations  
- Optimization functions  
- Cleaning utilities  

Run tests with:

```bash
pytest
```

---

## Command-Line Interface

Download a quick dataset:

```bash
portfolio-mpt data-spike --tickers AAPL MSFT NVDA
```

---

## Project Structure

```
portfolio-mpt/
│
├── portfolio_mpt/
│   ├── data.py
│   ├── wrangle.py
│   ├── analysis.py
│   ├── optimize.py
│   ├── viz.py
│   ├── clean.py
│   ├── cli.py
│   └── __init__.py
│
├── streamlit_app.py
├── requirements.txt
├── docs/
│   ├── index.qmd
│   ├── tutorial.qmd
│   ├── api.qmd
│   └── _quarto.yml
│
├── tests/
└── README.md
```

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- Yahoo Finance (`yfinance`)  
- NumPy, Pandas, SciPy  
- Altair & Matplotlib  
- Streamlit  
- Quarto  
