# DC Property Qualification Analysis

## Overview
This project investigates what factors determine whether a residential property in the District of Columbia is **qualified to be sold** on the housing market. Unlike traditional housing studies that model price as the response variable, this analysis focuses on **qualification status** using logistic regression.

## Research Questions
- What does it mean for a property to be “qualified” to sell?
- Is property price the most important factor in determining qualification?
- Do realtors prioritize qualification or price when listing properties?
- Can a statistically optimal regression model be constructed for qualification?

## Data
- Source: Kaggle DC Residential Property dataset
- Observations filtered and cleaned due to missing data and outliers
- Qualification assumed to be based on DC housing guidelines and inspections

## Methodology
- Logistic regression
- Stepwise variable selection (AIC & BIC)
- Interaction terms
- Model diagnostics and ROC analysis

## Key Results
- Final model demonstrated good fit based on goodness-of-fit testing
- ROC AUC ≈ 0.75 (training) and 0.72 (validation)
- Price, number of rooms, AC, condition, and ward location were significant predictors

## Document
- **DC Property Project (PDF)**  
  📄 `DC Property Project.pdf`

## Author
**Aaron Niecestro**

## Timeline Disclaimer

This analysis was conducted from January 2019 to June 2019.
