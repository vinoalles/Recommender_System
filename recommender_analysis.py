#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVIE RECOMMENDER SYSTEM – METADATA-BASED COSINE SIMILARITY ENGINE
Author: Vinodhkumar Gunasekaran
License: MIT

This script loads:
1. imdb_processed_features.csv  – engineered feature dataset (public)
2. X_scaled.npy                 – standardized feature matrix for similarity search

It then:
• Fits a cosine KNN model
• Generates Top-K similarity recommendations
"""

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------
# A. LOAD PROCESSED DATA
# ---------------------------------------------------------
# These two files come from Zenodo deposit
df = pd.read_csv("/Users/vinodhkumargunasekaran/Desktop/recommender_zenodo_package/imdb_processed_features_fixed.csv")   # PUBLIC metadata only
X_scaled = np.load("/Users/vinodhkumargunasekaran/Desktop/recommender_zenodo_package/X_scaled.npy")                # Precomputed feature matrix


# ---------------------------------------------------------
# B. FIT COSINE SIMILARITY MODEL
# ---------------------------------------------------------
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(X_scaled)


# ---------------------------------------------------------
# C. RECOMMENDATION FUNCTION
# ---------------------------------------------------------
def recommend_by_imdb_id(imdb_id: str, k: int = 5):
    """
    Retrieve Top-K similar movies using cosine similarity on metadata features.

    Parameters:
        imdb_id (str): IMDb ID of the movie (e.g., 'tt0387564')
        k (int): number of recommendations to return

    Returns:
        pandas.DataFrame: recommended titles and similarity scores
    """

    # Find movie row inside processed df
    matches = df.index[df["imdb_title_id"] == imdb_id].to_list()
    if not matches:
        raise ValueError(f"IMDb ID {imdb_id} not found in processed dataset.")

    idx = matches[0]

    # Compute neighbors
    distances, indices = model.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=k + 1
    )

    similarities = 1 - distances.flatten()
    seed = df.iloc[idx]

    results = []

    for i in range(1, k + 1):  # Skip itself
        movie_idx = indices.flatten()[i]
        rec = df.iloc[movie_idx]

        results.append({
            "Seed Movie": f"{seed['original_title']} ({int(seed['year'])})",
            "Recommended Title": f"{rec['original_title']} ({int(rec['year'])})",
            "Cosine Similarity": float(similarities[i])
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# D. EXAMPLE: SAW (2004)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\nTop 5 recommendations for Saw (2004):\n")
    recs = recommend_by_imdb_id("tt0387564", k=5)
    print(recs)
    