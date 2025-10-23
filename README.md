# Gold Price Prediction
![Final Plot](https://github.com/adityashelke04/gold_price_prediction/blob/f73cb85097f13aa58b120e5fa264bfcdf6447fbb/screenshots/Actual%20Price%20vs%20Predicted%20Price.png)

A machine learning project that predicts gold prices (GLD) using **Random Forest Regression** based on historical market data including stock indices, oil prices, silver prices, and currency exchange rates.

## Project Overview

This project analyzes the relationship between gold prices and various financial indicators to build a predictive model. The model uses features like S&P 500 index (SPX), US Oil prices (USO), Silver prices (SLV), and EUR/USD exchange rates to forecast gold prices.

## Dataset

The dataset contains **2,290 records** spanning from January 2008 to May 2018 with the following features:

- **Date**: Trading date
- **SPX**: S&P 500 stock index value
- **GLD**: Gold price (target variable)
- **USO**: US Oil price
- **SLV**: Silver price
- **EUR/USD**: Euro to US Dollar exchange rate

## Technologies Used

- **Python 3**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning library
  - `RandomForestRegressor`: Prediction model
  - `train_test_split`: Data splitting
  - `metrics`: Model evaluation

## Features

- Data collection and preprocessing
- Exploratory data analysis with visualizations
- Correlation analysis between features
- Random Forest Regression model
- Model performance evaluation using R-squared metric
- Train-test split for robust validation

## Model Performance

The Random Forest Regressor demonstrates strong predictive capability with the following metrics:

- **R² Score (Test Data)**: High correlation between predicted and actual values
- **Training Strategy**: 80-20 train-test split for model validation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Gold_Price_Prediction.git

# Navigate to project directory
cd Gold_Price_Prediction

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

```python
# Run the Jupyter notebook
jupyter notebook Gold_Price_Prediction.ipynb
```

The notebook follows this workflow:

1. Import necessary libraries
2. Load and explore the dataset
3. Perform data analysis and visualization
4. Check for missing values and correlations
5. Split data into training and testing sets
6. Train Random Forest Regressor model
7. Evaluate model performance
8. Make predictions on test data

## Project Structure

```
Gold_Price_Prediction/
│
├── Gold_Price_Prediction.ipynb    # Main Jupyter notebook
├── gld_price_data.csv             # Dataset file
└── README.md                      # Project documentation
```

## Key Insights

The project reveals correlations between gold prices and market indicators, demonstrating that factors like stock market performance, oil prices, precious metal prices, and currency exchange rates significantly influence gold pricing trends.

## Future Enhancements

- Implement additional regression models (XGBoost, Linear Regression, SVR)
- Feature engineering for improved accuracy
- Time series analysis with LSTM/ARIMA models
- Real-time price prediction using live market data
- Hyperparameter tuning for model optimization
- Deploy as web application with Flask/Streamlit

## License

This project is open source and available for educational purposes.

## Author

Developed as part of machine learning portfolio projects.
