import streamlit as st
import pandas as pd
from portfolio_mpt import data, wrangle, analysis, optimize
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.set_page_config(page_title="Portfolio Optimizer", layout='wide')

st.title("Portfolio Optimizer App!")


@st.cache_data
def load_ticker_file():
    df = pd.read_csv("data/tickers.csv")
    df = df.dropna()
    return df

def weights_to_df(weights, tickers):
    return pd.DataFrame({
        "Ticker": tickers,
        "Weight (%)": (weights * 100).round(2)
    })

st.sidebar.header('Select inputs')

tickers_list = load_ticker_file()

sectors = ["All"] + sorted(tickers_list["Sector"].unique())

sector_choice = st.sidebar.selectbox("Filter by Sector", sectors)

if sector_choice == "All":
    filtered_df = tickers_list
else:
    filtered_df = tickers_list[tickers_list["Sector"] == sector_choice]

tickers = st.sidebar.multiselect(
    "Choose Tickers",
    filtered_df["Symbol"].tolist(),
    default=[]
)

start = st.sidebar.date_input('Start Date', pd.to_datetime('2023-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('2024-12-31'))

run_button = st.sidebar.button('Run Optimization')


# ----------------------------
# App Logic
# ----------------------------

if run_button:

    spec = data.FetchSpec(tickers, str(start), str(end))

    # ============================
    # Create Tabs
    # ============================
    prices_tab, returns_tab, estimates_tab, weights_tab, frontier_tab = st.tabs([
        "Prices",
        "Returns",
        "Estimates",
        "Optimization",
        "Efficient Frontier"
    ])


    # ============================
    # PRICES TAB
    # ============================
    with prices_tab:
        st.write("## Fetching & Cleaning Price Data")
        prices = data.fetch_prices(spec, clean=True)
        st.line_chart(prices)


    # ============================
    # RETURNS TAB
    # ============================
    with returns_tab:
        st.write("## Computing Returns")
        rets = wrangle.to_returns(prices)
        st.line_chart(rets)


    # ============================
    # ESTIMATES TAB
    # ============================
    with estimates_tab:
        st.write("## Expected Returns and Covariance")

        mu = analysis.expected_returns(rets)
        Sigma = analysis.covariance(rets)

        st.write("### Expected Returns")
        st.dataframe(mu)

        st.write("### Covariance Heatmap")
        cov_df = Sigma.reset_index().melt(id_vars="index", var_name="Asset", value_name="Covariance")
        cov_df = cov_df.rename(columns={"index": "Base"})

        heatmap = (
            alt.Chart(cov_df)
            .mark_rect()
            .encode(
                x=alt.X("Asset:O", sort=None),
                y=alt.Y("Base:O", sort=None),
                color=alt.Color("Covariance:Q", scale=alt.Scale(scheme="redblue")),
                tooltip=["Base", "Asset", "Covariance"]
            )
        )

        st.altair_chart(heatmap, use_container_width=True)


    # ============================
    # OPTIMIZATION TAB (Weights)
    # ============================
    with weights_tab:

        st.write("## Portfolio Weights")

        gmv = optimize.global_min_variance(mu, Sigma)
        ms = optimize.max_sharpe(mu, Sigma)

        def make_weights_df(weights, tickers):
            return pd.DataFrame({
                "Ticker": tickers,
                "Weight": weights,
                "Weight (%)": (weights * 100).round(2)
            })

        gmv_df = make_weights_df(gmv.weights, mu.index)
        ms_df = make_weights_df(ms.weights, mu.index)

        st.write("### Global Minimum Variance Weights")
        st.dataframe(gmv_df)

        st.write("### Maximum Sharpe Ratio Weights")
        st.dataframe(ms_df)

        st.write("### Portfolio Allocation (Pie Charts)")
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.pie(gmv_df["Weight"], labels=gmv_df["Ticker"], autopct="%1.1f%%")
            ax1.set_title("GMV Allocation")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie(ms_df["Weight"], labels=ms_df["Ticker"], autopct="%1.1f%%")
            ax2.set_title("Max Sharpe Allocation")
            st.pyplot(fig2)
            
        # ============================
        # Cumulative Return Growth Curve
        # ============================

        st.write("### Cumulative Return Growth Curve")

        gmv_port_rets = rets.mul(gmv.weights, axis=1).sum(axis=1)
        ms_port_rets = rets.mul(ms.weights, axis=1).sum(axis=1)

        gmv_cum = (1 + gmv_port_rets).cumprod()
        ms_cum = (1 + ms_port_rets).cumprod()

        cum_df = pd.DataFrame({
            "GMV Portfolio": gmv_cum,
            "Max Sharpe Portfolio": ms_cum
        })

        st.line_chart(cum_df)
        
        # ============================
        # Monte Carlo Portfolio Simulator
        # ============================

        st.write("### Monte Carlo Portfolio Simulations")

        import numpy as np

        n_sims = 25
        n_days = 365

        mu_daily = mu / 252
        Sigma_daily = Sigma / 252

        eigvals, eigvecs = np.linalg.eigh(Sigma_daily)
        eigvals[eigvals < 0] = 1e-8
        Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Simulation function
        def simulate(mu_d, Sigma_d, weights, n_days, n_sims=15):
            """
            Simulates future portfolio paths using multivariate normal returns.
            Returns an array of shape (n_days, n_sims) representing cumulative growth of $1.
            """
            sims = np.random.multivariate_normal(mu_d, Sigma_d, size=(n_days, n_sims))
            port_returns = sims @ weights
            port_growth = (1 + port_returns).cumprod(axis=0)
            return port_growth

        gmv_paths = simulate(mu_daily, Sigma_psd, gmv.weights, n_days, n_sims)
        ms_paths  = simulate(mu_daily, Sigma_psd, ms.weights, n_days, n_sims)

        st.write("#### Simulated Future Portfolio Paths (GMV)")
        st.line_chart(pd.DataFrame(gmv_paths))

        st.write("#### Simulated Future Portfolio Paths (Max Sharpe)")
        st.line_chart(pd.DataFrame(ms_paths))

        gmv_end = gmv_paths[-1, :]
        ms_end  = ms_paths[-1, :]

        gmv_avg = gmv_end.mean()
        gmv_med = np.median(gmv_end)
        gmv_min = gmv_end.min()
        gmv_max = gmv_end.max()

        ms_avg = ms_end.mean()
        ms_med = np.median(ms_end)
        ms_min = ms_end.min()
        ms_max = ms_end.max()

        summary_df = pd.DataFrame({
            "Metric": [
                "Average Ending Value",
                "Median Ending Value",
                "Min Ending Value",
                "Max Ending Value"
            ],
            "GMV Portfolio": [
                gmv_avg,
                gmv_med,
                gmv_min,
                gmv_max
            ],
            "Max Sharpe Portfolio": [
                ms_avg,
                ms_med,
                ms_min,
                ms_max
            ]
        })

        summary_df = summary_df.round(4)

        st.write("#### Summary of Ending Values")
        st.dataframe(summary_df)
        
        # ============================
        # Portfolio Statistics Panel
        # ============================

        st.write("### Portfolio Statistics Summary")

        def max_drawdown(cumulative):
            peaks = cumulative.cummax()
            dd = (cumulative - peaks) / peaks
            return float(dd.min())

        gmv_dd = max_drawdown(gmv_cum)
        ms_dd  = max_drawdown(ms_cum)

        gmv_total = gmv_cum.iloc[-1] - 1
        ms_total  = ms_cum.iloc[-1] - 1

        stats_df = pd.DataFrame({
            "Metric": [
                "Expected Return (annualized)",
                "Volatility (annualized)",
                "Sharpe Ratio",
                "Max Drawdown",
                "Total Cumulative Return"
            ],
            "GMV Portfolio": [
                gmv.ret,
                gmv.vol,
                gmv.sharpe,
                gmv_dd,
                gmv_total
            ],
            "Max Sharpe Portfolio": [
                ms.ret,
                ms.vol,
                ms.sharpe,
                ms_dd,
                ms_total
            ]
        })

        stats_df["GMV Portfolio"] = stats_df["GMV Portfolio"].round(4)
        stats_df["Max Sharpe Portfolio"] = stats_df["Max Sharpe Portfolio"].round(4)

        st.dataframe(stats_df)


    # ============================
    # EFFICIENT FRONTIER TAB
    # ============================
    with frontier_tab:

        st.write("## Efficient Frontier")
        st.markdown("""
        The **Efficient Frontier** represents the set of portfolios that offer the **highest expected return** 
        for each level of **risk (volatility)**.  
        Any portfolio **below** the curve is sub-optimal.

        - **Blue points** = efficient portfolios  
        - **Red square** = Max Sharpe portfolio  
        - **Green dashed line** = Capital Market Line (CML)  
        """)

        ef = optimize.frontier_risk_aversion(mu, Sigma)

        ef_df = pd.DataFrame({
            "return": [p.ret for p in ef],
            "volatility": [p.vol for p in ef],
            "sharpe": [p.sharpe for p in ef]
        })

        ef_chart = (
            alt.Chart(ef_df)
            .mark_circle(size=80)
            .encode(
                x=alt.X("volatility", title="Volatility (σ)"),
                y=alt.Y("return", title="Expected Return"),
                color=alt.Color("sharpe", title="Sharpe Ratio", scale=alt.Scale(scheme="blues")),
                tooltip=["volatility", "return", "sharpe"],
            )
            .interactive()
        )

        ms_point = pd.DataFrame({
            "return": [ms.ret],
            "volatility": [ms.vol],
            "sharpe": [ms.sharpe],
            "label": ["Max Sharpe"],
        })

        ms_layer = (
            alt.Chart(ms_point)
            .mark_circle(size=80, color="red")
            .encode(
                x="volatility",
                y="return",
                tooltip=["label", "volatility", "return", "sharpe"],
            )
        )

        rf = 0.0
        max_vol = ef_df["volatility"].max()

        cml_df = pd.DataFrame({
            "volatility": [0, max_vol],
            "return": [rf, rf + ms.sharpe * max_vol],
        })

        cml_layer = (
            alt.Chart(cml_df)
            .mark_line(color="green", strokeDash=[5,5], strokeWidth=2)
            .encode(
                x="volatility",
                y="return",
                tooltip=["volatility", "return"],
            )
        )

        st.altair_chart(ef_chart + ms_layer + cml_layer, width='stretch')
