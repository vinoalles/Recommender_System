# Movie Recommender System – Metadata-Based Similarity Engine

Author: Vinodhkumar Gunasekaran  
License: MIT  

This repository contains the full source code used to implement the movie
recommender system described in the accompanying manuscript.

The code loads preprocessed feature matrices (available on Zenodo), trains a
cosine-similarity KNN model, and generates Top-K movie recommendations using
IMDb IDs.

Zenodo Dataset (Processed Features):  
https://doi.org/10.5281/zenodo.17822412

---

## Repository Contents

### Code Files
- **recommender_analysis.py** – Loads processed metadata, performs cosine KNN, and produces recommendations.
- **analysis_plots.py** – Optional script for reproducing manuscript figures.
- **feature_metadata.json** – Machine-readable schema describing all engineered features.
- **requirements.txt** – Python package dependencies.
- **LICENSE** – MIT license.

### Data Files (hosted on Zenodo, not included in GitHub)
- `imdb_mapping.csv` – Minimal metadata (IMDb ID, title, year).
- `imdb_processed_features.csv` – Engineered numeric + one-hot encoded features.
- `X_scaled.npy` – Standardized feature matrix used for cosine similarity.

These datasets are safe to share and contain no proprietary IMDb content.

---

## Example Usage

```python
from recommender_analysis import recommend_by_imdb_id

recommend_by_imdb_id("tt0387564", k=5)

Installation
pip install -r requirements.txt


Software Availability 

ource code available from:
https://github.com/vinodhkumarg/movie-recommender-analytics

Archived source code available from:
https://doi.org/10.5281/zenodo.17822412

License: MIT License (OSI-approved)

⸻

Citation

If you use this software, please cite:

Gunasekaran, V. (2025).  
Movie Metadata Recommender System – Source Code [Software].  
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Contact

For correspondence: vinodh.gunasekaran@circana.com
---

# ✅ **3. LICENSE (MIT License — F1000 approved)**

MIT License

Copyright (c) 2025 Vinodhkumar Gunasekaran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
