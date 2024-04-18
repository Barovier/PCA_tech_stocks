
# coding: utf-8

# **Exploring Principal Component Analysis (PCA) in Financial Data Analysis**
# 
# 

# In this project, I use Principal Component Analysis (PCA) applied to stock price data from a portfolio of  tech companies: Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), Alphabet Class A (GOOGL), Amazon (AMZN), Tesla (TSLA), Meta Platforms Class A (META), Alphabet Class C (GOOG), Broadcom (AVGO), and Advanced Micro Devices (AMD).
# 
# I begin with data collection and preparation, where I gather historical daily stock price data from the S&P 500 and reduce it down to the top 10 technology companies such as Nvidia, Amazon, Google, and Broadcom. This raw data is then structured into a time series format, ensuring each row signifies a trading day and each column represents the adjusted closing price of a specific stock. I then performed some feature engineering, where the daily returns for each stock are computed, highlighting the percentage change in adjusted closing prices. To ensure uniformity and comparability across stocks, I normalized the returns using z-score normalization. With this data, I am able to apply PCA. Here, I compute the covariance matrix of returns data, erformed eigendecomposition on the covariance matrix to obtain the principal components, and discern principal components based on their explained variance. These components serve as pivotal tools for analyzing risk, optimizing portfolios, and recognizing patterns within the stock price data. Ultimately, I evaluate and interpret the results, where I scrutinize the proportion of variance explained by principal components, examine factor loadings, and visualize the reduced-dimensional data to glean actionable insights into portfolio management and financial analysis. 
# 
# PCA can help identify a smaller set of representative stocks that capture most of the variation in the market. 
# What I want to accomplish is to identify a smaller subset of the top 10 technology stocks in the S&P 500 that can deliver superior results through diversification. Reducing the number of constituent stocks in a portfolio is one application of PCA in finance, particularly in the context of constructing more concentrated or focused portfolios. 

# This data that I will be using represents financial data spanning the period 5 years of data, from March 29, 2019, to March 28, 2024. It consists of 1259 entries, with each row corresponding to a trading day. There are 499 columns in total, each representing the adjusted closing price of a specific stock. All data in the DataFrame is of type float64, indicating numerical values. Given the size of the dataset and the data type, the memory usage is relatively moderate, occupying approximately 4.8 MB of memory.

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[19]:


#Code used to pull data (Do not run, requires internet access and may produce errors)
#import yfinance as yf
#sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#data_table = pd.read_html(sp500url)
#tickers = data_table[0]['Symbol'].tolist()  # Your original list of tickers
#failed_tickers = [
#    'BF.B',
#    'GEV',
#    'SOLV',
#    'BRK.B'
#]

# List of tickers that you want to exclude
#exclude_tickers = ['BF.B', 'GEV', 'SOLV', 'BRK.B']

# Filter out the failed tickers that you want to exclude
#filtered_tickers = [ticker for ticker in tickers if ticker not in exclude_tickers]
#yf.download('MMM', start = '2022-05-02', end = '2022-05-05')
#snp_prices = yf.download(filtered_tickers, start = '2019-03-29', end = '2024-03-29')['Adj Close']


# **Data Collection and Preparation:**
# 
# Collect historical daily stock price data for a selection of tech companies, such as Apple (AAPL), Microsoft (MSFT), and Amazon (AMZN). Break down the dataset to the top 10 technology stocks (by market performance) for closer analysis. 
# Organize the data into a time series format, where each row represents a trading day and each column represents the adjusted closing price of a particular stock.`

# In[20]:


#In this section, I read in the dataset, convert the date column to datetime for time series, 
#and filter the columns that I will use in my analysis
df = pd.read_csv('snp_prices.csv')
# Convert the 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df.info())
#subsetting the dataset to only include the stocks we need 
df_subset = df[['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'GOOG', 'AVGO', 'AMD']]
df_subset.head()


# Checking for missing values is essential before performing eigendecomposition, especially on the covariance matrix, because missing data can lead to errors or biases in the computation. Eigendecomposition involves calculating the eigenvalues and eigenvectors of the covariance matrix, which requires a complete and properly formatted dataset.

# In[21]:


# Check for missing values
missing_values = df_subset.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing values
df_subset_filled = df_subset.fillna(method='ffill')  # Forward fill missing values
df_subset_filled = df_subset_filled.fillna(method='bfill')  # Backward fill any remaining missing values

# Check if there are any missing values after filling
missing_values_after_fill = df_subset_filled.isnull().sum()
print("Missing values after filling:\n", missing_values_after_fill)


# In[23]:


# Visualization of the adjusted closing prices
plt.figure(figsize=(10, 6))

for column in df_subset_filled.columns:
    plt.plot(df_subset_filled.index, df[column], marker='o', label=column)

plt.title('Adjusted Closing Prices of Stocks')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Across all stocks, the mean adjusted closing price varies considerably, ranging from as low as 51.81 dollars for Alphabet Inc. Class C (GOOG) to as high as 504.28 dollars for Broadcom Inc. (AVGO). The standard deviation also varies significantly, indicating the dispersion of data points around the mean. For example, Advanced Micro Devices Inc. (AMD) has a relatively lower standard deviation of 36.81 dollars compared to NVIDIA Corporation (NVDA), which has a much higher standard deviation of 170.44 dollars. Additionally, the minimum and maximum values for each stock reflect the range of prices observed during the period. For instance, the minimum adjusted closing price for Tesla Inc. (TSLA) is 11.93 dollars, while the maximum is 409.97 dollars. These statistics provide valuable insights into the distribution and variability of stock prices over the specified time frame.

# **Feature Engineering:**
#    - Calculate the daily returns for each stock by taking the percentage change in the adjusted closing price.
#    - Normalize the returns data to ensure that all stocks have similar scales. In this case, I used Z-score normalization.

# In[24]:


# Calculate the daily returns for each stock
returns = df_subset_filled.pct_change()

# Normalize the returns data using Z-score normalization
returns_normalized = (returns - returns.mean()) / returns.std()


# In[25]:


# Visualize the normalized returns
plt.figure(figsize=(12, 6))
returns_normalized.plot(figsize=(12,6))
plt.title('Normalized Daily Returns of Stocks')
plt.xlabel('Date')
plt.ylabel('Normalized Returns')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


# Calculating the distribution of returns for each stock is essential for PCA and eigendecomposition. It allows for data normalization, ensuring comparability across stocks, detects anomalies, and provides insights into data characteristics like volatility and variability. This preparation ensures more accurate and meaningful results from dimensionality reduction techniques.

# In[28]:


# Calculate the distribution of returns for each stock
returns_distribution = returns.describe()

# Display the distribution
print(returns_distribution)


# In[29]:


# Plot histograms for each stock
returns.hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# Across the ten stocks that I have chosen, the mean return hovers around a relatively small positive value, indicating a general tendency for positive returns on average. However, there is considerable variability in returns, as evidenced by the standard deviation, which is notably larger than the mean for most stocks. This indicates a wide dispersion of returns around the mean, suggesting that returns can vary significantly from day to day. Additionally, the minimum and maximum returns reflect the range of returns observed during the period, with some stocks exhibiting occasional negative returns. The quartile values further illustrate the spread of returns, with the interquartile range providing a measure of the central tendency of returns while excluding outliers. Overall, these statistics offer valuable insights into the distribution and volatility of returns for each stock.

# In[30]:


#Accounting for empty row
df_final = returns.iloc[1:] 
df_final.head()


# In[31]:


#Calculate the covariance matrix
cov_matrix = df_final.cov()


# In[37]:


# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm')
plt.title('Covariance Matrix of Stock Returns', fontsize=16)
plt.xlabel('Stocks', fontsize=14)
plt.ylabel('Stocks', fontsize=14)
plt.show()


# In[38]:


#Perform eigendecomposition on the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

#Select a subset of principal components based on explained variance
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Choose the number of principal components to keep
# For example, let's keep the first 2 principal components
num_components = 2
principal_components = eigenvectors[:, :num_components]


# In[40]:


# Visualize the eigenvalues
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.5, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Principal Components')
plt.grid(True)
plt.show()

# Visualize the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.grid(True)
plt.show()


# In[39]:


# Project the original returns data onto the selected principal components
projected_data = np.dot(df_final.values, principal_components)

# Plot the projected data
plt.figure(figsize=(10, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projected Data onto Principal Components')
plt.grid(True)
plt.show()


# **Results and Interpretation**

# In[33]:


# 5.1 Proportion of Variance Explained
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.savefig('figure1.png')  # Save the figure
plt.show()


# The cumulative explained variance ratio measures the cumulative contribution of these principal components as we incrementally include more of them in the analysis. In this specific scenario, as we move from considering only the first principal component to incorporating additional components, we observe a steady increase in the cumulative explained variance ratio. For instance, when we include the first principal component, it explains approximately 61% of the total variance. As we add more components, such as the second, third, and so on, the cumulative explained variance ratio progressively rises. By the time we consider all ten principal components, the cumulative explained variance ratio reaches 100%, indicating that together, these components account for the entire variability present in the dataset. This information guides the decision-making process in determining the optimal number of principal components to retain, ensuring that a sufficient amount of variance is captured while minimizing dimensionality.

# In[34]:


# 5.2 Factor Loadings Analysis
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Factor Loadings of Stocks on Principal Components')
plt.grid(True)
plt.savefig('figure2.png')  # Save the figure
plt.show()


# In the above chart we examine the contribution of each stock to the principal components of the reduced-dimensional space. Each stock is represented by a pair of loading values, indicating its influence on the two principal components selected for analysis. A negative loading value suggests an inverse relationship with the principal component, while a positive loading value indicates a direct relationship. For instance, in the case of Stock 1, its loading on Principal Component 1 is -0.24, suggesting a weak negative relationship, while its loading on Principal Component 2 is 0.06, indicating a relatively weaker positive relationship. This analysis helps us understand how each stock contributes to the overall risk and return characteristics of the portfolio in the reduced-dimensional space.

# In[35]:


# 5.3 Visualization of Reduced-Dimensional Representation
plt.figure(figsize=(10, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Reduced-Dimensional Data')
plt.grid(True)
plt.savefig('figure3.png')  # Save the figure
plt.show()


# Scatter Plot of Reduced-Dimensional Data:
# Each point represents a stock in the reduced-dimensional space.
# 

# In[36]:


# Reconstruct the original data from the principal components
reconstructed_data = np.dot(projected_data, principal_components.T)

# Calculate the reconstruction error
reconstruction_error = np.mean((df_final.values - reconstructed_data) ** 2, axis=0)

# Print the reconstruction error for each stock
print("Reconstruction Error for Each Stock:")
for i, stock in enumerate(df_final.columns):
    print(f"{stock}: {reconstruction_error[i]:.6f}")

# Calculate the percentage of variance explained by each principal component
pct_variance_explained = 100 * eigenvalues / np.sum(eigenvalues)

# Print the percentage of variance explained by each principal component
print("\nPercentage of Variance Explained by Each Principal Component:")
for i, pct in enumerate(pct_variance_explained):
    print(f"Principal Component {i + 1}: {pct:.2f}%")

# Identify the stocks with the highest factor loadings on the first principal component
sorted_loadings = np.argsort(np.abs(principal_components[:, 0]))[::-1]
top_stocks = [df_final.columns[idx] for idx in sorted_loadings[:3]]
print(f"\nTop 3 Stocks Contributing the Most to the First Principal Component:\n{top_stocks}")


# **Reconstruction Error**:
# 
# The reconstruction error represents the mean squared error between the original data and the reconstructed data from the principal components. Smaller values indicate a better reconstruction and representation of the original data by the principal components.
# 
# - Tesla (TSLA) has the smallest reconstruction error of 0.000009, suggesting that the principal components capture the variability in Tesla's stock prices very well.
# - Meta Platforms (META) and Advanced Micro Devices (AMD) have relatively higher reconstruction errors of 0.000313 and 0.000334, respectively, indicating that their stock price movements are not as well represented by the principal components compared to other stocks.
# 
# **Percentage of Variance Explained by Principal Components**:
# - The first principal component explains 60.94% of the total variance in the dataset, which is a substantial portion of the overall variability.
# - The second principal component accounts for an additional 14.13% of the variance.
# - Together, the first two principal components capture approximately 75% of the total variance.
# - As we include more principal components, the cumulative explained variance increases, but the incremental contribution of each subsequent component diminishes.
# 
# **Top Stocks Contributing to the First Principal Component**:
# - Tesla (TSLA), NVIDIA (NVDA), and Advanced Micro Devices (AMD) have the highest factor loadings on the first principal component, indicating that they contribute the most to the primary source of variation in the dataset.
# - These stocks likely exhibit similar risk and return characteristics, and their price movements are strongly correlated with the first principal component.
# 
# Based on these results, we can make the following recommendations:
# 
# - Retaining the first two principal components may be sufficient to capture a significant portion of the variability in the stock price data while reducing the dimensionality from 10 stocks to 2 components. This can simplify portfolio analysis and optimization.
# - When constructing a concentrated portfolio, we may consider including stocks like Tesla, NVIDIA, and Advanced Micro Devices, as they are the primary drivers of the first principal component and capture a substantial amount of the overall variation.
# - Stocks like Meta Platforms and Advanced Micro Devices, which have higher reconstruction errors, may benefit from additional analysis or risk management strategies, as their price movements are not as well represented by the principal components.
# 
# It's important to note that while PCA provides valuable insights into the structure and relationships within the data, it should be combined with other financial analysis techniques and considerations when making investment decisions. Additionally, the interpretation of results may vary depending on the specific goals and risk preferences of the portfolio manager or investor.

# In[ ]:




