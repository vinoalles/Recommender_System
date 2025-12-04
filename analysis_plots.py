import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ---------------------------------------------------------
# DOWNLOAD RAW DATA DIRECTLY FROM KAGGLE
# (User must have Kaggle API enabled)
# ---------------------------------------------------------
path = kagglehub.dataset_download("simhyunsu/imdbextensivedataset")
print("Dataset folder:", path)

# Load IMDB Movies CSV
df = pd.read_csv(f"{path}/IMDb movies.csv")

# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["avg_vote"] = pd.to_numeric(df["avg_vote"], errors="coerce")
df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

df = df.dropna(subset=["year", "avg_vote", "votes"])

# ---------------------------------------------------------
# FIGURE 6 — AVERAGE REVIEWS & RATING PER YEAR
# ---------------------------------------------------------
yearly = (
    df.groupby("year")
      .agg(
          avg_rating=("avg_vote", "mean"),
          avg_reviews=("votes", "mean")
      )
      .reset_index()
)

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=yearly,
    x="year",
    y="avg_reviews",
    hue="avg_rating",
    palette="Blues",
    s=80
)
plt.title("Average rating and number of reviews per year", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Average No of reviews")
plt.legend(title="avg_rating")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# C. FIGURE 7 — GENRE-RATING PLOTS (4 SEPARATE FIGURES)
# ---------------------------------------------------------
# Expand multi-genre entries
df_genres = (
    df.assign(genre=df["genre"].str.split(","))
      .explode("genre")
)
df_genres["genre"] = df_genres["genre"].str.strip()
df_genres = df_genres[df_genres["genre"] != ""]

# Compute mean rating per genre
genre_rating = (
    df_genres.groupby("genre")["avg_vote"]
    .mean()
    .sort_values(ascending=False)
)

# Age Groups — generic labels
age_titles = {
    "0–18":  "Average rating per genre for 0–18",
    "18–30": "Average rating per genre for 18–30",
    "30–45": "Average rating per genre for 30–45",
    "45+":   "Average rating per genre for 45+"
}

sns.set_style("whitegrid")

for age_group, title in age_titles.items():

    plt.figure(figsize=(7, 10))
    sns.scatterplot(
        x=genre_rating.values,
        y=genre_rating.index,
        s=90,
        color="royalblue"
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Average rating")
    plt.ylabel("Genre")
    plt.xlim(genre_rating.min() - 0.2, genre_rating.max() + 0.2)

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()