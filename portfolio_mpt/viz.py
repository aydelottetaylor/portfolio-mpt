import matplotlib.pyplot as plt

def line(df, title="Prices"):
    ax = df.plot(figsize=(9,4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    return ax