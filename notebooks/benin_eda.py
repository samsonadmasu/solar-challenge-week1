#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for Benin Solar Data
# 
# This notebook performs exploratory data analysis on solar farm data from Benin.

# ## Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# ## Load Data

# Load the Benin solar data
# Replace with your actual file path
file_path = '../data/benin_data.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for demonstration if file doesn't exist
    print("Creating sample data for demonstration...")
    # This is just for demonstration - replace with your actual data
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
        'GHI': np.random.normal(500, 100, 1000),
        'DNI': np.random.normal(700, 150, 1000),
        'DHI': np.random.normal(200, 50, 1000),
        'ModA': np.random.normal(450, 90, 1000),
        'ModB': np.random.normal(460, 95, 1000),
        'Tamb': np.random.normal(25, 5, 1000),
        'RH': np.random.uniform(30, 90, 1000),
        'WS': np.random.exponential(3, 1000),
        'WSgust': np.random.exponential(5, 1000),
        'WSstdev': np.random.uniform(0.5, 2, 1000),
        'WD': np.random.uniform(0, 360, 1000),
        'WDstdev': np.random.uniform(5, 20, 1000),
        'BP': np.random.normal(1013, 5, 1000),
        'Cleaning': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'Precipitation': np.random.exponential(0.1, 1000),
        'TModA': np.random.normal(30, 8, 1000),
        'TModB': np.random.normal(31, 8, 1000),
    })

# ## Data Overview

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
df.head()

# Display basic information about the dataset
print("\nDataset info:")
df.info()

# ## Summary Statistics

# Calculate summary statistics for all numeric columns
print("\nSummary statistics:")
df.describe()

# ## Missing Value Analysis

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isna().sum()
print(missing_values)

# Calculate percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
columns_with_missing = missing_percentage[missing_percentage > 5].index.tolist()
print(f"\nColumns with >5% missing values: {columns_with_missing}")

# ## Outlier Detection

# Function to detect outliers using Z-score
def detect_outliers(df, columns, threshold=3):
    outliers = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers[col] = len(outlier_indices)
    return outliers

# Detect outliers in key columns
key_columns = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
outliers = detect_outliers(df, key_columns)
print("\nNumber of outliers (|Z| > 3) per column:")
for col, count in outliers.items():
    print(f"{col}: {count} outliers ({count/len(df)*100:.2f}%)")

# ## Data Cleaning

# Create a copy of the dataframe for cleaning
df_clean = df.copy()

# Handle missing values
for col in df_clean.columns:
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        # Impute missing values with median
        if df_clean[col].isna().sum() > 0:
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)

# Handle outliers (optional - comment out if you want to keep outliers)
# for col in key_columns:
#     if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
#         z_scores = np.abs(stats.zscore(df_clean[col]))
#         df_clean = df_clean[(z_scores < 3)]

# Convert timestamp to datetime if it's not already
if 'Timestamp' in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean['Timestamp']):
    df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])

# ## Time Series Analysis

# Plot time series for GHI, DNI, DHI
if 'Timestamp' in df_clean.columns:
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_clean['Timestamp'], df_clean['GHI'])
    plt.title('Global Horizontal Irradiance (GHI) Over Time')
    plt.ylabel('GHI (W/m²)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_clean['Timestamp'], df_clean['DNI'])
    plt.title('Direct Normal Irradiance (DNI) Over Time')
    plt.ylabel('DNI (W/m²)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_clean['Timestamp'], df_clean['DHI'])
    plt.title('Diffuse Horizontal Irradiance (DHI) Over Time')
    plt.xlabel('Time')
    plt.ylabel('DHI (W/m²)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ## Cleaning Impact Analysis

# Analyze the impact of cleaning on module readings
if 'Cleaning' in df_clean.columns:
    cleaning_impact = df_clean.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(cleaning_impact))
    width = 0.35
    
    plt.bar(x - width/2, cleaning_impact['ModA'], width, label='ModA')
    plt.bar(x + width/2, cleaning_impact['ModB'], width, label='ModB')
    
    plt.xlabel('Cleaning')
    plt.ylabel('Average Module Reading (W/m²)')
    plt.title('Impact of Cleaning on Module Readings')
    plt.xticks(x, ['No Cleaning (0)', 'Cleaning (1)'])
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()

# ## Correlation Analysis

# Calculate correlation matrix
corr_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'ModA', 'ModB']
corr_columns = [col for col in corr_columns if col in df_clean.columns]

if corr_columns:
    correlation = df_clean[corr_columns].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# ## Scatter Plots

# Plot scatter plots for key relationships
if all(col in df_clean.columns for col in ['WS', 'GHI']):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df_clean['WS'], df_clean['GHI'], alpha=0.5)
    plt.title('Wind Speed vs GHI')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('GHI (W/m²)')
    plt.grid(True)
    
    if 'RH' in df_clean.columns and 'Tamb' in df_clean.columns:
        plt.subplot(1, 3, 2)
        plt.scatter(df_clean['RH'], df_clean['Tamb'], alpha=0.5)
        plt.title('Relative Humidity vs Ambient Temperature')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('Ambient Temperature (°C)')
        plt.grid(True)
    
    if 'RH' in df_clean.columns:
        plt.subplot(1, 3, 3)
        plt.scatter(df_clean['RH'], df_clean['GHI'], alpha=0.5)
        plt.title('Relative Humidity vs GHI')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('GHI (W/m²)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ## Wind Analysis

# Create wind rose plot if wind direction and speed are available
if 'WD' in df_clean.columns and 'WS' in df_clean.columns:
    try:
        # Create wind rose using plotly
        fig = go.Figure()
        
        # Convert wind direction to bins
        bin_size = 30
        bins = np.arange(0, 360 + bin_size, bin_size)
        labels = bins[:-1] + bin_size/2
        
        # Group wind speed by direction
        df_clean['WD_bin'] = pd.cut(df_clean['WD'], bins=bins, labels=labels, include_lowest=True)
        wind_rose_data = df_clean.groupby('WD_bin')['WS'].mean().reset_index()
        
        fig.add_trace(go.Barpolar(
            r=wind_rose_data['WS'],
            theta=wind_rose_data['WD_bin'],
            name='Wind Speed',
            marker_color=wind_rose_data['WS'],
            marker_colorscale='Viridis',
            marker_showscale=True,
            marker_colorbar_title='Wind Speed (m/s)'
        ))
        
        fig.update_layout(
            title='Wind Rose - Average Wind Speed by Direction',
            polar=dict(
                radialaxis=dict(range=[0, wind_rose_data['WS'].max() * 1.1])
            )
        )
        
        fig.show()
    except Exception as e:
        print(f"Error creating wind rose plot: {e}")
        print("Creating a simple histogram of wind direction instead.")
        
        plt.figure(figsize=(10, 6))
        plt.hist(df_clean['WD'], bins=36, alpha=0.7)
        plt.title('Wind Direction Distribution')
        plt.xlabel('Wind Direction (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# ## Distribution Analysis

# Create histograms for GHI and Wind Speed
if 'GHI' in df_clean.columns and 'WS' in df_clean.columns:
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_clean['GHI'], kde=True)
    plt.title('Distribution of GHI')
    plt.xlabel('GHI (W/m²)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_clean['WS'], kde=True)
    plt.title('Distribution of Wind Speed')
    plt.xlabel('Wind Speed (m/s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ## Temperature Analysis

# Analyze relationship between RH, temperature, and solar radiation
if all(col in df_clean.columns for col in ['RH', 'Tamb', 'GHI']):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with color gradient
    scatter = ax1.scatter(df_clean['RH'], df_clean['Tamb'], c=df_clean['GHI'], 
                         cmap='viridis', alpha=0.6, s=50)
    ax1.set_title('Relationship between RH, Temperature, and GHI')
    ax1.set_xlabel('Relative Humidity (%)')
    ax1.set_ylabel('Ambient Temperature (°C)')
    ax1.grid(True)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('GHI (W/m²)')
    
    # Plot 2: 2D histogram / heatmap
    h = ax2.hist2d(df_clean['RH'], df_clean['Tamb'], bins=20, cmap='viridis')
    ax2.set_title('2D Histogram of RH vs Temperature')
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Ambient Temperature (°C)')
    cbar2 = plt.colorbar(h[3], ax=ax2)
    cbar2.set_label('Count')
    
    plt.tight_layout()
    plt.show()

# ## Bubble Chart

# Create bubble chart of GHI vs Tamb with bubble size = RH or BP
if all(col in df_clean.columns for col in ['GHI', 'Tamb']):
    plt.figure(figsize=(12, 8))
    
    # Determine bubble size variable
    if 'RH' in df_clean.columns:
        size_var = 'RH'
    elif 'BP' in df_clean.columns:
        size_var = 'BP'
    else:
        # Use a constant size if neither RH nor BP is available
        df_clean['const'] = 50
        size_var = 'const'
    
    # Normalize the size variable for better visualization
    size_scale = df_clean[size_var] / df_clean[size_var].max() * 200
    
    scatter = plt.scatter(df_clean['Tamb'], df_clean['GHI'], 
                         s=size_scale, alpha=0.5, 
                         c=df_clean[size_var], cmap='viridis')
    
    plt.title(f'GHI vs Ambient Temperature (Bubble Size = {size_var})')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('GHI (W/m²)')
    plt.grid(True)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(size_var)
    
    plt.tight_layout()
    plt.show()

# ## Save Cleaned Data

# Save the cleaned dataframe to CSV
output_path = '../data/benin_clean.csv'
try:
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")
except Exception as e:
    print(f"Error saving cleaned data: {e}")

# ## Key Findings

# Summarize key findings from the EDA
print("\nKey Findings from Benin Solar Data EDA:")
print("1. [Replace with your finding about GHI/DNI/DHI patterns]")
print("2. [Replace with your finding about correlations between variables]")
print("3. [Replace with your finding about cleaning impact]")
print("4. [Replace with your finding about wind patterns]")
print("5. [Replace with your finding about temperature and humidity effects]")
