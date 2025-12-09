import matplotlib.pyplot as plt

def line(df, title="Prices"):
    ax = df.plot(figsize=(9,4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    return ax

def efficient_frontier_plot(results, ms=None, cml=False, rf=0.0):
    """
    Plot the efficient frontier using matplotlib.

    Parameters
    ----------
    results : list of OptResult
        The efficient frontier results.
    ms : OptResult, optional
        The maximum sharpe portfolio point.
    cml : bool, default False
        Whether to plot the Capital Market Line.
    rf : float, default 0.0
        Risk-free rate for CML.
    """
    vols = [r.vol for r in results]
    rets = [r.ret for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vols, rets, c='blue', s=20, label="Efficient Frontier")

    if ms:
        ax.scatter(ms.vol, ms.ret, c='red', s=80, label="Max Sharpe")

    if cml and ms:
        max_vol = max(vols)
        cml_y = [rf, rf + ms.sharpe * max_vol]
        ax.plot([0, max_vol], cml_y, 'g--', label="CML")

    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    plt.tight_layout()

    return ax