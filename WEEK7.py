# Data Analysis with Pandas and Matplotlib
# This script demonstrates data loading, exploration, analysis, and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set the style for plots
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.1)

# Task 1: Load and Explore the Dataset
print("=" * 80)
print("TASK 1: LOADING AND EXPLORING THE DATASET")
print("=" * 80)

# Try loading a dataset with error handling
try:
    # For this example, we'll use a sales dataset
    # If you don't have this file, the script will download the Iris dataset instead
    file_path = 'sales_data.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    else:
        print(f"File {file_path} not found. Using Iris dataset instead.")
        # Load iris dataset as a fallback
        from sklearn.datasets import load_iris

        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Add a date column for time series visualization
        # This is a simulated date column for demonstration purposes
        dates = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        df['date'] = dates

        # Add a sales column for demonstration
        np.random.seed(42)  # For reproducibility
        df['sales'] = np.random.randint(10, 100, size=len(df))

except Exception as e:
    print(f"An error occurred: {e}")
    print("Creating a sample dataset for demonstration.")

    # Create a simple sample dataset
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(10, 100, size=150),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=150),
        'quantity': np.random.randint(1, 10, size=150),
        'price': np.random.uniform(10, 100, size=150).round(2)
    })

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean the dataset by handling missing values
if df.isnull().values.any():
    print("\nCleaning the dataset...")
    # For numerical columns, fill missing values with the mean
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in {col} with mean: {mean_val:.2f}")

    # For categorical columns, fill missing values with the mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")

    print("Missing values after cleaning:")
    print(df.isnull().sum())
else:
    print("\nNo missing values found in the dataset.")

# Task 2: Basic Data Analysis
print("\n" + "=" * 80)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 80)

# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Check if 'species' or 'category' column exists for grouping
if 'species' in df.columns:
    groupby_col = 'species'
elif 'category' in df.columns:
    groupby_col = 'category'
else:
    # Create a category column if none exists
    df['category'] = pd.qcut(df.select_dtypes(include=np.number).iloc[:, 0], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    groupby_col = 'category'

# Find a numeric column for analysis
if 'sales' in df.columns:
    numeric_col = 'sales'
elif 'quantity' in df.columns:
    numeric_col = 'quantity'
else:
    numeric_col = df.select_dtypes(include=np.number).columns[0]

# Perform grouping and compute mean
print(f"\nMean of {numeric_col} grouped by {groupby_col}:")
group_means = df.groupby(groupby_col)[numeric_col].mean().sort_values(ascending=False)
print(group_means)

# Additional analysis: Compute correlation matrix if multiple numeric columns exist
numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    print("\nCorrelation matrix of numerical columns:")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix.round(2))

    # Find the highest correlated pairs
    corr_pairs = correlation_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]  # Remove self-correlations
    high_corr = corr_pairs[abs(corr_pairs) > 0.5].sort_values(ascending=False)

    if not high_corr.empty:
        print("\nHighly correlated pairs (|correlation| > 0.5):")
        print(high_corr)

# Task 3: Data Visualization
print("\n" + "=" * 80)
print("TASK 3: DATA VISUALIZATION")
print("=" * 80)

# Create a figure with subplots
fig = plt.figure(figsize=(20, 16))

# 1. Line chart showing trends over time
print("\nCreating a line chart showing trends over time...")
if 'date' in df.columns:
    ax1 = fig.add_subplot(2, 2, 1)

    # Group by date if there are multiple entries per date
    if df['date'].nunique() < len(df):
        time_series = df.groupby('date')[numeric_col].mean()
        time_series.plot(kind='line', ax=ax1, marker='o', linestyle='-', markersize=4)
    else:
        df.plot(x='date', y=numeric_col, kind='line', ax=ax1, marker='o', linestyle='-', markersize=4)

    ax1.set_title(f'{numeric_col.capitalize()} Trend Over Time', fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel(numeric_col.capitalize(), fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

else:
    print("No date column found for time series plot. Creating an index-based trend instead.")
    ax1 = fig.add_subplot(2, 2, 1)
    df[numeric_col].plot(kind='line', ax=ax1)
    ax1.set_title(f'{numeric_col.capitalize()} Trend', fontsize=14, pad=20)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel(numeric_col.capitalize(), fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

# 2. Bar chart for comparison across categories
print("Creating a bar chart comparing categories...")
ax2 = fig.add_subplot(2, 2, 2)
group_means.plot(kind='bar', ax=ax2, color='teal')
ax2.set_title(f'Average {numeric_col.capitalize()} by {groupby_col.capitalize()}', fontsize=14, pad=20)
ax2.set_xlabel(groupby_col.capitalize(), fontsize=12)
ax2.set_ylabel(f'Average {numeric_col.capitalize()}', fontsize=12)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(group_means):
    ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# 3. Histogram of a numerical column
print("Creating a histogram...")
ax3 = fig.add_subplot(2, 2, 3)
sns.histplot(df[numeric_col], bins=20, kde=True, ax=ax3, color='purple')
ax3.set_title(f'Distribution of {numeric_col.capitalize()}', fontsize=14, pad=20)
ax3.set_xlabel(numeric_col.capitalize(), fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)

# Calculate and display statistics
mean_val = df[numeric_col].mean()
median_val = df[numeric_col].median()
ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
ax3.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
ax3.legend()

# 4. Scatter plot to visualize relationship between two numerical columns
print("Creating a scatter plot...")
ax4 = fig.add_subplot(2, 2, 4)

# Choose two numeric columns for the scatter plot
if len(numeric_cols) > 1:
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]

    # Use different colors for categories if available
    if groupby_col in df.columns:
        for category, group in df.groupby(groupby_col):
            ax4.scatter(group[x_col], group[y_col], label=category, alpha=0.7)
        ax4.legend(title=groupby_col.capitalize())
    else:
        ax4.scatter(df[x_col], df[y_col], color='blue', alpha=0.7)

    ax4.set_title(f'Relationship between {x_col.capitalize()} and {y_col.capitalize()}', fontsize=14, pad=20)
    ax4.set_xlabel(x_col.capitalize(), fontsize=12)
    ax4.set_ylabel(y_col.capitalize(), fontsize=12)

    # Add regression line
    try:
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
        x = np.array([df[x_col].min(), df[x_col].max()])
        y = intercept + slope * x
        ax4.plot(x, y, 'r--', label=f'R²: {r_value ** 2:.2f}')
        ax4.legend()
    except:
        pass

else:
    ax4.text(0.5, 0.5, 'Not enough numerical columns for scatter plot',
             horizontalalignment='center', verticalalignment='center', fontsize=14)

# Add space between subplots
plt.tight_layout(pad=4.0)

# Add a title to the entire figure
fig.suptitle('Data Analysis Visualization', fontsize=20, y=0.98)

# Save the visualizations
try:
    plt.savefig('data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'data_analysis_visualizations.png'")
except Exception as e:
    print(f"Could not save visualization: {e}")

# Show plots
plt.show()

# Display findings and observations
print("\n" + "=" * 80)
print("FINDINGS AND OBSERVATIONS")
print("=" * 80)

# Highlight key findings from the analysis
print(f"\n1. {groupby_col.capitalize()} Analysis:")
print(
    f"   - The category with the highest average {numeric_col} is {group_means.index[0]} with a value of {group_means.iloc[0]:.2f}")
print(
    f"   - The category with the lowest average {numeric_col} is {group_means.index[-1]} with a value of {group_means.iloc[-1]:.2f}")

# Calculate and report the overall statistics
print(f"\n2. {numeric_col.capitalize()} Distribution:")
print(f"   - Mean: {df[numeric_col].mean():.2f}")
print(f"   - Median: {df[numeric_col].median():.2f}")
print(f"   - Range: {df[numeric_col].min():.2f} to {df[numeric_col].max():.2f}")
print(f"   - Standard Deviation: {df[numeric_col].std():.2f}")

# Report on time trends if applicable
if 'date' in df.columns:
    grouped_by_date = df.groupby(pd.Grouper(key='date', freq='M'))[numeric_col].mean()
    if not grouped_by_date.empty:
        max_month = grouped_by_date.idxmax().strftime('%B %Y')
        min_month = grouped_by_date.idxmin().strftime('%B %Y')
        print(f"\n3. Time Trend Analysis:")
        print(f"   - Highest average {numeric_col} was in {max_month} with {grouped_by_date.max():.2f}")
        print(f"   - Lowest average {numeric_col} was in {min_month} with {grouped_by_date.min():.2f}")

# Report on correlations if applicable
if len(numeric_cols) > 1:
    if not high_corr.empty:
        strongest_pair = high_corr.index[0]
        strongest_corr = high_corr.iloc[0]
        print(f"\n4. Correlation Analysis:")
        print(f"   - Strongest relationship found between {strongest_pair[0]} and {strongest_pair[1]}")
        print(f"   - Correlation coefficient: {strongest_corr:.2f}")
        if strongest_corr > 0:
            print(f"   - This indicates a strong positive relationship")
        else:
            print(f"   - This indicates a strong negative relationship")

print("\nAnalysis complete!")