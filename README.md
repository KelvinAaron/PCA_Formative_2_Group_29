# PCA Formative 2 — Group 29

Principal Component Analysis (PCA) applied to COVID-19 data across African countries, implemented from scratch using NumPy only (no scikit-learn).

## Dataset

`covid_africa.csv` — COVID-19 statistics per African country (total cases, deaths, and other numeric features). Missing values are imputed with column means.

## What the script does

1. Loads and standardizes the data (zero mean, unit variance)
2. Computes the covariance matrix
3. Performs eigendecomposition to find principal components
4. Sorts components by explained variance (descending)
5. Projects data onto the top 2 principal components
6. Plots original data (first 2 features) vs. PCA-reduced data side by side

## Usage

```bash
python formative_2.py
```

Requires: `numpy`, `matplotlib`

## Files

| File | Description |
|------|-------------|
| `formative_2.py` | Main PCA implementation |
| `covid_africa.csv` | Input dataset |
| `Template_PCA_Formative_2_Group_29.pdf` | Assignment template |
