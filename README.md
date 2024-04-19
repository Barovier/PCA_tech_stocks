# Exploring Principal Component Analysis (PCA) in Financial Data Analysis

In this project, we delve into the application of Principal Component Analysis (PCA) to analyze stock price data from a portfolio of tech companies, including Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), Alphabet Class A (GOOGL), Amazon (AMZN), Tesla (TSLA), Meta Platforms Class A (META), Alphabet Class C (GOOG), Broadcom (AVGO), and Advanced Micro Devices (AMD).

## Objective
The main objective of this project is to utilize PCA to identify a smaller subset of technology stocks from the S&P 500 that can deliver superior results through diversification. By reducing the number of constituent stocks in a portfolio, we aim to construct more focused portfolios that capture most of the variation in the market.

## Data Collection and Preparation
The project begins with data collection and preparation, where historical daily stock price data is gathered from the S&P 500 and filtered down to the top 10 technology companies. The raw data is structured into a time series format, ensuring each row signifies a trading day, and each column represents the adjusted closing price of a specific stock. Feature engineering techniques, such as calculating daily returns and normalizing the data using z-score normalization, are applied to ensure uniformity and comparability across stocks.

## Analysis Steps
1. **Data Collection and Preparation**: Gather historical daily stock price data and organize it into a time series format.
2. **Feature Engineering**: Calculate daily returns for each stock and normalize the returns data using z-score normalization.
3. **Covariance Matrix Calculation**: Compute the covariance matrix of returns data.
4. **Eigendecomposition**: Perform eigendecomposition on the covariance matrix to obtain principal components.
5. **Analysis and Interpretation**: Evaluate the results, including the proportion of variance explained by principal components, factor loadings analysis, and visualization of reduced-dimensional data.

## Results and Interpretation
The results of the PCA analysis provide valuable insights into the structure and relationships within the stock price data. Key findings include:
- Identification of a smaller subset of technology stocks that capture most of the variability in the market.
- Evaluation of the proportion of variance explained by principal components.
- Analysis of factor loadings to understand the contribution of each stock to the principal components.
- Visualization of reduced-dimensional data to identify patterns and relationships.

Based on these findings, recommendations can be made for constructing more focused and diversified portfolios that align with specific investment goals and risk preferences.

## Conclusion
PCA serves as a powerful tool for analyzing and interpreting complex financial datasets, such as stock price data. By reducing dimensionality and identifying underlying patterns, PCA enables investors and portfolio managers to make informed decisions and optimize portfolio construction strategies. However, it is essential to combine PCA with other financial analysis techniques and considerations to develop robust investment strategies.

For detailed implementation and code examples, please refer to the Jupyter Notebook included in this repository.
