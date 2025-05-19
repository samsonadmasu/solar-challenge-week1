#!/usr/bin/env python
# coding: utf-8

# # Cross-Country Comparison of Solar Data
# 
# This notebook compares solar farm data from Benin, Sierra Leone, and Togo to identify relative solar potential and key differences across countries.

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

# ## Load Cleaned Data

# Load the cleaned data for each country
try:
    benin_df = pd.read_csv('../data/benin_clean.csv')
    print(f"Benin data loaded successfully with {benin_df.shape[0]} rows and {benin_df.shape[1]} columns.")
    benin_df['Country'] = 'Benin'
except Exception as e:
    print(f"Error loading Benin data: {e}")
    benin_df = pd.DataFrame()

try:
    sierra_leone_df = pd.read_csv('../data/sierra_leone_clean.csv')
    print(f"Sierra Leone data loaded successfully with {sierra_leone_df.shape[0]} rows and {sierra_leone_df.shape[1]} columns.")
    sierra_leone_df['Country'] = 'Sierra Leone'
except Exception as e:
    print(f"Error loading Sierra Leone data: {e}")
    sierra_leone_df = pd.DataFrame()

try:
    togo_df = pd.read_csv('../data/togo_clean.csv')
    print(f"Togo data loaded successfully with {togo_df.shape[0]} rows and {togo_df.shape[1]} columns.")
    togo_df['Country'] = 'Togo'
except Exception as e:
    print(f"Error loading Togo data: {e}")
    togo_df = pd.DataFrame()

# Check if we have data for all countries
if benin_df.empty or sierra_leone_df.empty or togo_df.empty:
    print("\nWARNING: Missing data for one or more countries. Creating sample data for demonstration...")
    
    # Create sample data for demonstration
    if benin_df.empty:
        benin_df = pd.DataFrame({
            'GHI': np.random.normal(500, 100, 1000),
            'DNI': np.random.normal(700, 150, 1000),
            'DHI': np.random.normal(200, 50, 1000),
            'Tamb': np.random.normal(25, 5, 1000),
            'RH': np.random.uniform(30, 90, 1000),
            'WS': np.random.exponential(3, 1000),
            'Country': ['Benin'] * 1000
        })
    
    if sierra_leone_df.empty:
        sierra_leone_df = pd.DataFrame({
            'GHI': np.random.normal(480, 110, 1000),
            'DNI': np.random.normal(680, 160, 1000),
            'DHI': np.random.normal(190, 55, 1000),
            'Tamb': np.random.normal(27, 4, 1000),
            'RH': np.random.uniform(40, 95, 1000),
            'WS': np.random.exponential(2.5, 1000),
            'Country': ['Sierra Leone'] * 1000
        })
    
    if togo_df.empty:
        togo_df = pd.DataFrame({
            'GHI': np.random.normal(520, 105, 1000),
            'DNI': np.random.normal(720, 145, 1000),
            'DHI': np.random.normal(210, 45, 1000),
            'Tamb': np.random.normal(26, 4.5, 1000),
            'RH': np.random.uniform(35, 85, 1000),
            'WS': np.random.exponential(2.8, 1000),
            'Country': ['Togo'] * 1000
        })

# ## Combine Data for Comparison

# Ensure all dataframes have the same columns for comparison
common_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'Country']
common_columns = [col for col in common_columns if all(col in df.columns for df in [benin_df, sierra_leone_df, togo_df])]

# Select only common columns
benin_df_common = benin_df[common_columns]
sierr_leone_df_common = sierra_leone_df[common_columns]
togo_df_common = togo_df[common_columns]

# Combine the dataframes
combined_df = pd.concat([benin_df_common, sierr_leone_df_common, togo_df_common])
print(f"\nCombined data shape: {combined_df.shape}")

# ## Metric Comparison

# Define the metrics to compare
metrics = ['GHI', 'DNI', 'DHI']

# Create boxplots for each metric
for metric in metrics:
    if metric in combined_df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Country', y=metric, data=combined_df)
        plt.title(f'Distribution of {metric} by Country')
        plt.xlabel('Country')
        plt.ylabel(f'{metric} (W/mu00b2)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

# ## Summary Statistics Table

# Create a summary table comparing mean, median, and standard deviation
summary_table = pd.DataFrame()

for metric in metrics:
    if metric in combined_df.columns:
        # Calculate statistics for each country
        for country in ['Benin', 'Sierra Leone', 'Togo']:
            country_data = combined_df[combined_df['Country'] == country]
            
            # Create a row for the summary table
            row = pd.DataFrame({
                'Metric': [metric],
                'Country': [country],
                'Mean': [country_data[metric].mean()],
                'Median': [country_data[metric].median()],
                'Std Dev': [country_data[metric].std()],
                'Min': [country_data[metric].min()],
                'Max': [country_data[metric].max()]
            })
            
            summary_table = pd.concat([summary_table, row])

# Display the summary table
print("\nSummary Statistics Table:")
print(summary_table.to_string(index=False))

# ## Statistical Testing

# Perform one-way ANOVA or Kruskal-Wallis test for each metric
for metric in metrics:
    if metric in combined_df.columns:
        print(f"\nStatistical Test for {metric}:")
        
        # Extract data for each country
        benin_data = combined_df[combined_df['Country'] == 'Benin'][metric]
        sierra_leone_data = combined_df[combined_df['Country'] == 'Sierra Leone'][metric]
        togo_data = combined_df[combined_df['Country'] == 'Togo'][metric]
        
        # Check normality (simplified check)
        _, benin_p = stats.shapiro(benin_data.sample(min(1000, len(benin_data))))
        _, sierra_leone_p = stats.shapiro(sierra_leone_data.sample(min(1000, len(sierra_leone_data))))
        _, togo_p = stats.shapiro(togo_data.sample(min(1000, len(togo_data))))
        
        all_normal = all(p > 0.05 for p in [benin_p, sierra_leone_p, togo_p])
        
        if all_normal:
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(benin_data, sierra_leone_data, togo_data)
            test_name = "One-way ANOVA"
        else:
            # Perform Kruskal-Wallis test
            h_stat, p_value = stats.kruskal(benin_data, sierra_leone_data, togo_data)
            test_name = "Kruskal-Wallis"
        
        print(f"{test_name} p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"Significant difference found in {metric} between countries (p < 0.05)")
        else:
            print(f"No significant difference found in {metric} between countries (p >= 0.05)")

# ## Visual Summary

# Create a bar chart ranking countries by average GHI
if 'GHI' in combined_df.columns:
    # Calculate average GHI for each country
    avg_ghi = combined_df.groupby('Country')['GHI'].mean().reset_index()
    
    # Sort by average GHI in descending order
    avg_ghi = avg_ghi.sort_values('GHI', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_ghi['Country'], avg_ghi['GHI'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Average GHI by Country')
    plt.xlabel('Country')
    plt.ylabel('Average GHI (W/mu00b2)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# ## Key Observations

# Markdown cell with key observations
print("\nKey Observations:")
print("1. [Replace with your observation about which country has the highest/lowest solar potential]")
print("2. [Replace with your observation about variability in solar radiation across countries]")
print("3. [Replace with your observation about other environmental factors that might influence solar potential]")

# ## Additional Comparison: Temperature and Humidity

# Compare temperature and humidity across countries
if all(col in combined_df.columns for col in ['Tamb', 'RH']):
    # Temperature comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Country', y='Tamb', data=combined_df)
    plt.title('Ambient Temperature by Country')
    plt.xlabel('Country')
    plt.ylabel('Temperature (u00b0C)')
    plt.grid(True, axis='y')
    
    # Humidity comparison
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Country', y='RH', data=combined_df)
    plt.title('Relative Humidity by Country')
    plt.xlabel('Country')
    plt.ylabel('Relative Humidity (%)')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

# ## Correlation Comparison

# Compare correlation patterns across countries
if all(col in combined_df.columns for col in ['GHI', 'Tamb', 'RH']):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Loop through countries and create scatter plots
    for i, country in enumerate(['Benin', 'Sierra Leone', 'Togo']):
        country_data = combined_df[combined_df['Country'] == country]
        
        # Create scatter plot
        axes[i].scatter(country_data['Tamb'], country_data['GHI'], alpha=0.5)
        axes[i].set_title(f'{country}: Temperature vs GHI')
        axes[i].set_xlabel('Temperature (u00b0C)')
        axes[i].set_ylabel('GHI (W/mu00b2)')
        axes[i].grid(True)
        
        # Add correlation coefficient
        corr = country_data[['Tamb', 'GHI']].corr().iloc[0, 1]
        axes[i].annotate(f'Correlation: {corr:.2f}', 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# ## Recommendation

# Based on the analysis, provide a recommendation for solar investment
print("\nRecommendation for Solar Investment:")
print("Based on the cross-country comparison analysis, [replace with your recommendation]")
print("\nJustification:")
print("1. [Replace with your justification based on GHI/DNI/DHI comparison]")
print("2. [Replace with your justification based on environmental factors]")
print("3. [Replace with your justification based on statistical significance]")
