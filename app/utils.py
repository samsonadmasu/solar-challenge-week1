import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load and preprocess data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if it exists
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def plot_boxplot(df, y_col, x_col, figsize=(12, 6)):
    """
    Create a boxplot for the specified columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_col : str
        Column name for y-axis
    x_col : str
        Column name for x-axis (grouping)
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f'Distribution of {y_col} by {x_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, columns, figsize=(10, 8)):
    """
    Create a correlation heatmap for the specified columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of column names to include in the heatmap
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig

def create_time_series_plot(df, time_col, value_col, title=None, figsize=(12, 6)):
    """
    Create a time series plot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Column name for time axis
    value_col : str
        Column name for value axis
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[time_col], df[value_col])
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{value_col} Over Time')
        
    ax.set_xlabel('Time')
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
