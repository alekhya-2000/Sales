# Retail Sales Data Analysis

## Overview
This project analyzes retail sales data to identify trends, visualize insights, and build a simple forecasting model. The dataset includes variables such as sales figures, marketing spend, seasonality effects, and holiday impacts.

## Features
- **Data Cleaning & Preprocessing**: Handles missing values, formats date columns, and removes duplicates.
- **Exploratory Data Analysis (EDA)**: Generates visualizations to detect trends and patterns in sales data.
- **Sales Forecasting**: Uses a linear regression model to predict future sales based on past trends and external factors.
- **Visualization**: Uses Matplotlib and Seaborn to create line plots and trend charts.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The dataset consists of 365 daily records containing:
- `date`: Date of record (YYYY-MM-DD format)
- `sales`: Total sales for the day
- `marketing_spend`: Amount spent on marketing
- `holiday_flag`: Indicates whether the day was a holiday (1 for holiday, 0 for non-holiday)
- `seasonality_index`: Captures seasonal fluctuations

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/retail-sales-analysis.git
   cd retail-sales-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis script:
   ```bash
   python retail_sales_analysis.py
   ```

## Usage
- Modify `retail_sales.csv` to analyze your own sales data.
- Adjust model parameters in `retail_sales_analysis.py` as needed.
- Visualize results using Matplotlib-generated plots.

## License
This project is licensed under the MIT License.

---

### Author
Alekhya Kesapragada

