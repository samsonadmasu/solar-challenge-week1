import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from utils import load_data, plot_boxplot, plot_correlation_heatmap

# Set page config
st.set_page_config(
    page_title="Solar Data Analysis Dashboard",
    page_icon="☀️",
    layout="wide"
)

# App title and description
st.title("Solar Farm Analysis Dashboard")
st.markdown("""
This dashboard provides insights into solar farm data from Benin, Sierra Leone, and Togo.
Explore the data through various visualizations and comparisons.
""")

# Sidebar for navigation and filters
st.sidebar.title("Navigation")

# Country selection
countries = st.sidebar.multiselect(
    "Select Countries",
    ["Benin", "Sierra Leone", "Togo"],
    default=["Benin", "Sierra Leone", "Togo"]
)

# Load data
try:
    data = {}
    for country in countries:
        file_path = f"data/{country.lower().replace(' ', '_')}_clean.csv"
        if os.path.exists(file_path):
            data[country] = load_data(file_path)
            data[country]["Country"] = country
        else:
            st.sidebar.warning(f"Data file for {country} not found.")
    
    if data:
        # Combine data for comparison
        combined_data = pd.concat(data.values())
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Series", "Correlations", "Country Comparison"])
        
        with tab1:
            st.header("Data Overview")
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                metric = st.selectbox("Select Metric", ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"])
                
            with col2:
                st.write(f"Summary Statistics for {metric}")
                summary_df = pd.DataFrame({
                    "Country": [],
                    "Mean": [],
                    "Median": [],
                    "Std Dev": []
                })
                
                for country in countries:
                    if country in data:
                        summary_df = pd.concat([summary_df, pd.DataFrame({
                            "Country": [country],
                            "Mean": [data[country][metric].mean()],
                            "Median": [data[country][metric].median()],
                            "Std Dev": [data[country][metric].std()]
                        })])
                
                st.dataframe(summary_df)
            
            # Boxplot comparison
            st.subheader(f"Distribution of {metric} by Country")
            fig = plot_boxplot(combined_data, metric, "Country")
            st.pyplot(fig)
            
        with tab2:
            st.header("Time Series Analysis")
            
            # Time series visualization
            country = st.selectbox("Select Country for Time Series", countries)
            metric_ts = st.selectbox("Select Metric for Time Series", ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"])
            
            if country in data:
                # Convert timestamp to datetime if it's not already
                if "Timestamp" in data[country].columns:
                    if not pd.api.types.is_datetime64_any_dtype(data[country]["Timestamp"]):
                        data[country]["Timestamp"] = pd.to_datetime(data[country]["Timestamp"])
                    
                    # Plot time series
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(data[country]["Timestamp"], data[country][metric_ts])
                    ax.set_xlabel("Time")
                    ax.set_ylabel(metric_ts)
                    ax.set_title(f"{metric_ts} Over Time in {country}")
                    st.pyplot(fig)
                else:
                    st.error("Timestamp column not found in the data.")
            
        with tab3:
            st.header("Correlation Analysis")
            
            country_corr = st.selectbox("Select Country for Correlation Analysis", countries)
            
            if country_corr in data:
                # Select numeric columns for correlation
                numeric_cols = data[country_corr].select_dtypes(include=[np.number]).columns.tolist()
                selected_cols = st.multiselect(
                    "Select Variables for Correlation",
                    numeric_cols,
                    default=["GHI", "DNI", "DHI", "Tamb", "RH"] if all(col in numeric_cols for col in ["GHI", "DNI", "DHI", "Tamb", "RH"]) else numeric_cols[:5]
                )
                
                if selected_cols:
                    # Plot correlation heatmap
                    fig = plot_correlation_heatmap(data[country_corr], selected_cols)
                    st.pyplot(fig)
                    
                    # Scatter plot
                    st.subheader("Scatter Plot")
                    x_var = st.selectbox("X-axis", selected_cols)
                    y_var = st.selectbox("Y-axis", [col for col in selected_cols if col != x_var], index=min(1, len(selected_cols)-1))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(data[country_corr][x_var], data[country_corr][y_var], alpha=0.5)
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                    ax.set_title(f"{y_var} vs {x_var} in {country_corr}")
                    st.pyplot(fig)
            
        with tab4:
            st.header("Country Comparison")
            
            # Metrics for comparison
            comparison_metrics = st.multiselect(
                "Select Metrics for Comparison",
                ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"],
                default=["GHI"]
            )
            
            if comparison_metrics:
                # Create comparison visualizations
                for metric in comparison_metrics:
                    st.subheader(f"{metric} Comparison")
                    
                    # Bar chart for average values
                    avg_data = {country: data[country][metric].mean() for country in data if metric in data[country].columns}
                    
                    if avg_data:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(avg_data.keys(), avg_data.values())
                        ax.set_xlabel("Country")
                        ax.set_ylabel(f"Average {metric}")
                        ax.set_title(f"Average {metric} by Country")
                        
                        # Add value labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.2f}',
                                    ha='center', va='bottom')
                        
                        st.pyplot(fig)
    else:
        st.warning("No data available. Please select at least one country with available data.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("This is a placeholder dashboard. You need to add your cleaned data files to the 'data' directory.")
