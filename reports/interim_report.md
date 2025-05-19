# Solar Challenge Week 1: Interim Report

**Date:** May 19, 2025  
**Project:** Cross-Country Solar Farm Analysis  
**Student:** Samson Admasu  

## 1. Introduction

This interim report outlines the progress made on the 10 Academy Solar Challenge Week 1 project. The challenge focuses on analyzing solar farm data from Benin, Sierra Leone, and Togo to identify high-potential regions for solar installation for MoonLight Energy Solutions.

## 2. Project Setup Summary

### 2.1 Git & Environment Setup

I have successfully completed the Git and environment setup tasks as specified in the challenge requirements:

- Created a GitHub repository named `solar-challenge-week1`
- Set up the project structure with appropriate directories and files
- Created a `.gitignore` file to exclude data files and other unnecessary files
- Added a `requirements.txt` file with all necessary dependencies
- Implemented a GitHub Actions workflow for CI in `.github/workflows/ci.yml`
- Created a comprehensive README.md with setup instructions
- Set up the recommended folder structure

### 2.2 Repository Structure

The repository has been structured according to the challenge requirements:

```
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── data/  (gitignored)
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   ├── benin_eda.py
│   ├── sierra_leone_eda.py
│   ├── togo_eda.py
│   └── compare_countries.py
├── reports/
│   └── interim_report.md
├── requirements.txt
├── scripts/
│   ├── __init__.py
│   └── README.md
├── src/
└── tests/
    └── __init__.py
```

### 2.3 Git Workflow

I have implemented a proper Git workflow for the project:

- Created a `setup-task` branch for the initial setup
- Made multiple commits with descriptive messages
- Merged the `setup-task` branch into `main` via a pull request
- Set `main` as the default branch

## 3. Data Profiling, Cleaning & EDA Approach

### 3.1 Data Understanding

The solar farm data includes measurements of:
- Global Horizontal Irradiance (GHI)
- Direct Normal Irradiance (DNI)
- Diffuse Horizontal Irradiance (DHI)
- Module measurements (ModA, ModB)
- Environmental factors (temperature, humidity, wind speed, etc.)
- Cleaning events

### 3.2 Planned EDA Approach

For each country's dataset, I plan to implement the following EDA approach:

1. **Data Loading and Initial Inspection**
   - Load the data and examine basic structure
   - Check data types and summary statistics
   - Identify missing values and potential outliers

2. **Data Cleaning**
   - Handle missing values through appropriate imputation methods
   - Detect and handle outliers using Z-scores
   - Ensure proper data types and formats

3. **Exploratory Analysis**
   - Analyze time series patterns in GHI, DNI, and DHI
   - Examine the impact of cleaning events on module readings
   - Analyze correlations between variables
   - Investigate relationships between environmental factors and solar radiation
   - Create visualizations to understand data distributions and patterns

4. **Feature Engineering**
   - Create derived features if needed (e.g., daily/monthly averages)
   - Identify key factors influencing solar potential

### 3.3 Visualization Strategy

I plan to create the following visualizations for each country:

- Time series plots of GHI, DNI, DHI
- Correlation heatmaps
- Scatter plots of key relationships
- Wind rose plots for wind direction and speed analysis
- Distribution histograms
- Bubble charts for multivariate analysis

## 4. Cross-Country Comparison Plan

For the cross-country comparison, I will:

1. Load the cleaned datasets from all three countries
2. Create side-by-side boxplots for key metrics (GHI, DNI, DHI)
3. Generate a summary table comparing statistics across countries
4. Perform statistical tests (ANOVA or Kruskal-Wallis) to assess differences
5. Create visualizations to rank countries by solar potential
6. Identify key factors that differentiate the countries

## 5. Dashboard Development Plan

I have set up the foundation for a Streamlit dashboard with the following features:

- Country selection widgets
- Interactive visualizations of key metrics
- Time series analysis components
- Correlation analysis tools
- Country comparison visualizations

The dashboard will allow users to explore the data and insights interactively, with options to filter by country and metric.

## 6. Next Steps

For the remainder of the challenge, I plan to:

1. Obtain the actual solar data for all three countries
2. Implement the EDA analysis as outlined above
3. Perform the cross-country comparison
4. Refine the Streamlit dashboard with actual data and insights
5. Prepare the final report with comprehensive findings and recommendations

## 7. Challenges and Mitigations

- **Challenge**: Handling potentially large datasets
  - **Mitigation**: Implement efficient data loading and processing techniques

- **Challenge**: Identifying meaningful patterns across different countries
  - **Mitigation**: Use statistical tests and standardized metrics for comparison

- **Challenge**: Creating an intuitive dashboard for non-technical stakeholders
  - **Mitigation**: Focus on clear visualizations and insights with minimal technical jargon

## 8. Conclusion

The project setup phase has been successfully completed, and I am now ready to proceed with the data analysis tasks. The foundation has been laid for a comprehensive analysis of solar potential across Benin, Sierra Leone, and Togo, which will provide valuable insights for MoonLight Energy Solutions' strategic planning for solar investments.

---

*This interim report was prepared for the 10 Academy Solar Challenge Week 1 project.*
