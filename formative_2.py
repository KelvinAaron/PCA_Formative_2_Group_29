import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Standardize the data (use of numpy only allowed)

raw_data = np.genfromtxt('covid_africa.csv', delimiter=',', dtype=str, encoding='utf-8', autostrip=True)

header = raw_data[0]
data = raw_data[1:]

# Extract numeric columns and handle missing values
numeric_raw = data[:, 1:].copy()
numeric_raw[numeric_raw == ''] = 'nan'
numeric_data = numeric_raw.astype(float)
col_means = np.nanmean(numeric_data, axis=0)
for i in range(numeric_data.shape[1]):
    numeric_data[np.isnan(numeric_data[:, i]), i] = col_means[i]

# Standardize: (X - mean) / std
mean = np.mean(numeric_data, axis=0)
std = np.std(numeric_data, axis=0)
standardized_data = (numeric_data - mean) / std

standardized_data[:5]  # Display the first few rows of standardized data

# Step 3: Calculate the Covariance Matrix
cov_matrix = np.cov(standardized_data, rowvar=False)
cov_matrix

# Step 4: Perform Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues, eigenvectors

# Step 5: Sort Principal Components
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
sorted_eigenvectors = eigenvectors[:, sorted_indices]  # Sort eigenvectors accordingly
sorted_eigenvectors

# Step 6: Project Data onto Principal Components
num_components = 2  # Decide on the number of principal components to keep
reduced_data = standardized_data @ sorted_eigenvectors[:, :num_components]  # Project data onto the principal components
reduced_data[:5]

# Step 7: Output the Reduced Data
print(f'Reduced Data Shape: {reduced_data.shape}')  # Display reduced data shape
reduced_data[:5]  # Display the first few rows of reduced data  

# Step 8: Visualize Before and After PCA

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot original data (first two features for simplicity)
ax1.scatter(standardized_data[:, 0], standardized_data[:, 1], color='blue', alpha=0.6)
ax1.set_title('Original Data (First 2 Features)')
ax1.set_xlabel('Total Cases (standardized)')
ax1.set_ylabel('Total Deaths (standardized)')

# Plot reduced data after PCA
ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], color='red', alpha=0.6)
ax2.set_title('After PCA (2 Principal Components)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()