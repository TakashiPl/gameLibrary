import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/steam_clean.csv")

fig, axes = plt.subplots(1, 2, figsize=(14,5))

sns.histplot(df["rating"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Game rating")
axes[0].set_xlabel("Rating (0-100)")
axes[0].set_ylabel("Games count")

sns.scatterplot(
    data = df,
    x="price",
    y="rating",
    hue="genre",
    style="is_free",
    s=100,
    ax=axes[1]
)

axes[1].set_title("Price vs Rating (grouped by genres)")
axes[1].set_xlabel("Price ($)")
axes[1].set_ylabel("Rating (0-100)")

plt.tight_layout()

plt.savefig("data/eda_summary.png")
print("Chart has been saved to file: data/eda_summary.png")
plt.show()