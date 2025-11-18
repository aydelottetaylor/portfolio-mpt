import argparse
from .data import FetchSpec, fetch_prices
from .wrangle import to_returns
from .viz import line

def main():
    p = argparse.ArgumentParser("portfolio-mpt")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data-spike", help="Download few tickers and save to data/raw/")
    s.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","NVDA"])
    s.add_argument("--start", default="2023-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--interval", default="1d")
    s.add_argument("--force", action="store_true")

    args = p.parse_args()
    if args.cmd == "data-spike":
        spec = FetchSpec(args.tickers, args.start, args.end, args.interval)
        prices = fetch_prices(spec, force=args.force)
        rets = to_returns(prices)
        # quick visual for sanity (non-blocking in CI; fine locally)
        try:
            line(prices, title=f"Prices: {' '.join(args.tickers)}")
            import matplotlib.pyplot as plt; plt.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()
