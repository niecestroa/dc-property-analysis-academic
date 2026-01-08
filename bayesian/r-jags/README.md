# Bayesian Modeling of DC Housing Prices using R jags

## Overview
This project applies Bayesian modeling techniques to identify which variables best explain **residential property prices** in Washington, DC. The analysis extends beyond classical regression approaches to incorporate uncertainty through Bayesian inference.

## Objectives
- Identify key predictors of housing prices in DC
- Compare Bayesian models to traditional regression approaches
- Evaluate model performance using DIC and posterior diagnostics

## Data
- Source: Kaggle DC Residential Property dataset
- Original dataset: ~159,000 observations
- Final dataset: ~57,000 observations
- Training/testing split: 50/50

## Methodology
- Bayesian hierarchical regression
- MCMC sampling
- Posterior distribution analysis
- Model comparison using Deviance Information Criterion (DIC)

## Key Findings
- Bathrooms, fireplaces, property age, and condition were strong predictors
- Some predictors showed wide credible intervals crossing zero
- Model provided meaningful insight but required further refinement

## Tools
- R
- rjags
- coda
- Bayesian inference

## Author
**Aaron Niecestro**

## Timeline Disclaimer

- This analysis was conducted from August 2019 to December 2019.
