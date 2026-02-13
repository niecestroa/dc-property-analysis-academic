# -*- coding: utf-8 -*-
"""
Created on February 13, 2026
Editted on February 13, 2026

Note: Worked on through multiple file and complied together into this one file on last edit date.

This is a conversion of previous R code in to python working code.

This file is for my machine learning work of DC Property Data from Kaggle to practice converting work from R to python.
"""

# ============================================================
# MODULE 0 — IMPORTS & GLOBAL SETUP
# Modern Python rewrite of the R project dependencies
# ============================================================

# -----------------------------
# Core Data Handling
# -----------------------------
import pandas as pd              # Data loading, cleaning, manipulation (Python equivalent of dplyr/tidyverse)
import numpy as np               # Numerical operations, arrays, vectorized math (R's base numeric ops)

# -----------------------------
# Visualization
# -----------------------------
import seaborn as sns            # High-level statistical visualization (R's ggplot2 equivalent)
import matplotlib.pyplot as plt  # Low-level plotting engine used by seaborn (R's base graphics)
sns.set_theme(style="whitegrid") # Consistent aesthetic for all plots

# -----------------------------
# Statistical Modeling (GLM, OLS, ANOVA, Influence)
# -----------------------------
import statsmodels.api as sm                     # Core statsmodels API (GLM, OLS, diagnostics)
import statsmodels.formula.api as smf            # R-style formula interface (y ~ x1 + x2)

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor                    # VIF for multicollinearity diagnostics
)

from statsmodels.stats.anova import anova_lm     # ANOVA for nested model comparison (partial F-tests)

import statsmodels.genmod.generalized_linear_model as glm
glm.SET_USE_BIC_LLF(True)                        # Use LLF-based BIC (future-proof, suppresses warnings)

# -----------------------------
# Machine Learning Models (sklearn)
# -----------------------------
from sklearn.metrics import (
    roc_curve, auc,                               # ROC curve + AUC (R's pROC equivalent)
    mean_squared_error,                           # Regression MSE (Ridge, LASSO, PCR, PLS, Trees, RF, Bagging)
    accuracy_score,                               # Classification accuracy (KNN, LDA, QDA, SVM, Trees, RF)
    confusion_matrix                              # Classification confusion matrix
)

from sklearn.preprocessing import (
    StandardScaler,                               # Feature scaling for ML models
    PolynomialFeatures                            # Polynomial basis expansion (R's poly(x, degree))
)

from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score
)                                                 # Splitting, tuning, cross-validation

from sklearn.neighbors import KNeighborsClassifier # KNN classification

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)                                                 # LDA & QDA

from sklearn.svm import SVC                       # Support Vector Machines (linear, poly, radial, sigmoid)

from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV
)                                                 # Linear, Ridge, LASSO regression

from sklearn.decomposition import PCA             # Principal Component Analysis (PCR foundation)
from sklearn.cross_decomposition import PLSRegression  # Partial Least Squares Regression

from sklearn.tree import (
    DecisionTreeRegressor, DecisionTreeClassifier,
    plot_tree                                     # Tree visualization (R's plot(tree))
)

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    BaggingRegressor                              # Bagging for regression trees
)

from sklearn.ensemble import BaggingClassifier    # Bagging for classification trees (R's bagging())

from sklearn.pipeline import Pipeline             # Pipeline for polynomial regression, scaling + model chaining

# -----------------------------
# Utility Libraries
# -----------------------------
import itertools                                  # Combinatorics (used for regsubsets-style feature selection)
import warnings                                   # Suppress noisy warnings when needed
warnings.filterwarnings("ignore")                 # Cleaner output (optional)

# ============================================================
# MODULE 1 — DATA LOADING & CLEANING (MODERN PYTHON VERSION)
# ============================================================

# Load Excel file
DC = pd.read_excel(
    r"C:\Users\aniec\Mirror\Programming Projects\2019.01_2024.05 DC Property Composite Analysis\2019.05.20 R - DC Property using GLM\Data\DC_Properties.xlsx"
)

# -----------------------------
# BASIC CLEANING
# -----------------------------

# Replace NA with 0 (as in R)
DC = DC.fillna(0)

# Select relevant columns (matching your R script)
cols = [
    "PRICE", "BATHRM", "HF_BATHRM", "HEAT", "AC", "ROOMS", "BEDRM",
    "AYB", "YR_RMDL", "EYB", "STORIES", "QUALIFIED", "GRADE", "CNDTN",
    "KITCHENS", "FIREPLACES", "WARD", "QUADRANT", "LATITUDE", "LONGITUDE"
]

DC = DC[cols].copy()

# Convert numeric columns
num_cols = [
    "PRICE", "BATHRM", "HF_BATHRM", "ROOMS", "BEDRM", "AYB", "YR_RMDL",
    "EYB", "STORIES", "KITCHENS", "FIREPLACES", "LATITUDE", "LONGITUDE"
]

DC[num_cols] = DC[num_cols].apply(pd.to_numeric, errors="coerce")

# Convert categorical columns
cat_cols = ["HEAT", "AC", "QUALIFIED", "GRADE", "CNDTN", "WARD", "QUADRANT"]
DC[cat_cols] = DC[cat_cols].astype(str)

# -----------------------------
# FILTERING (Python equivalent of R filters)
# -----------------------------

DC = DC[
    (DC["CNDTN"].isin(["", "Default", "Poor"]) == False) &
    (DC["GRADE"] != " No Data") &
    (DC["GRADE"] != "") &
    (DC["HEAT"] != "No Data") &
    (DC["PRICE"].between(10000, 10000000)) &
    (DC["FIREPLACES"] < 10) &
    (DC["KITCHENS"] <= 10) &
    (DC["ROOMS"] <= 40) &
    (DC["BEDRM"] <= 20) &
    (DC["STORIES"] <= 10) &
    (DC["LATITUDE"] != 0) &
    (DC["LONGITUDE"] != 0)
].copy()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Age variables
DC["AYB_age"] = np.where(DC["AYB"] == 2019, 0, 2019 - DC["AYB"])
DC["EYB_age"] = np.where(DC["EYB"] == 2019, 0, 2019 - DC["EYB"])

# Remodel age
DC["REMODEL_age"] = DC["YR_RMDL"].replace({"0": 0, "20": 0}).astype(float)
DC["REMODEL_age"] = np.where(DC["REMODEL_age"] == 0, 0, 2019 - DC["REMODEL_age"])

# Collapse Exceptional grades
DC["GRADE"] = DC["GRADE"].replace({
    "Exceptional-A": "Exceptional",
    "Exceptional-B": "Exceptional",
    "Exceptional-C": "Exceptional",
    "Exceptional-D": "Exceptional"
})

# Binary qualification
DC["QUALIFIED_2"] = (DC["QUALIFIED"] == "Q").astype(int)

# Fix AC
DC["AC"] = DC["AC"].replace({"0": "N"})

# Drop unused columns
DC = DC.drop(columns=["CNDTN", "AYB", "YR_RMDL", "EYB"])

# -----------------------------
# FINAL CLEANING
# -----------------------------

DC = DC.dropna()

# Remove extreme AYB ages
DC = DC[DC["AYB_age"] < 2000]

# Create PRICE_10K
DC["PRICE_10K"] = DC["PRICE"] / 10000

# Combine bathrooms
DC["BATHRM"] = DC["BATHRM"] + 0.5 * DC["HF_BATHRM"]

# Final column order
DC = DC[
    [
        "PRICE", "PRICE_10K", "BATHRM", "ROOMS", "BEDRM", "STORIES",
        "KITCHENS", "FIREPLACES", "LATITUDE", "LONGITUDE",
        "AYB_age", "EYB_age", "REMODEL_age",
        "HEAT", "AC", "QUALIFIED", "QUALIFIED_2", "GRADE",
        "WARD", "QUADRANT"
    ]
]

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------

np.random.seed(10000000)
DC = DC.sample(57610, replace=True)

n = len(DC)
Z = np.random.choice(n, n // 2, replace=False)

DC_train = DC.iloc[Z].copy()
DC_test = DC.drop(DC.index[Z]).copy()

# ============================================================
# MODULE 2 — DIAGNOSTICS (QQ PLOTS, HISTOGRAMS, DISTRIBUTIONS)
# ============================================================

# Make sure your dataset is named DC
# If your dataset is DC_train, then rename it:
# DC = DC_train

# Variables to check (matching your R diagnostics)
diag_vars = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES",
    "KITCHENS", "FIREPLACES",
    "AYB_age", "EYB_age", "REMODEL_age"
]

# -----------------------------
# QQ PLOTS
# -----------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for ax, var in zip(axes, diag_vars):
    stats.probplot(DC[var], dist="norm", plot=ax)
    ax.set_title(f"QQ Plot — {var}")

plt.tight_layout()
plt.show()

# -----------------------------
# HISTOGRAMS
# -----------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for ax, var in zip(axes, diag_vars):
    sns.histplot(DC[var], kde=True, ax=ax)
    ax.set_title(f"Histogram — {var}")

plt.tight_layout()
plt.show()

# ============================================================
# MODULE 3 — STEPWISE MODEL SELECTION (AIC / BIC)
# ============================================================

# -----------------------------
# Helper: compute information criterion
# -----------------------------

def _get_ic(model, criterion="aic"):
    """
    Return the chosen information criterion from a fitted statsmodels model.
    criterion: "aic" or "bic"
    Uses LLF-based BIC (bic_llf) to match global glm.SET_USE_BIC_LLF(True).
    """
    if criterion.lower() == "aic":
        return model.aic
    elif criterion.lower() == "bic":
        # Use LLF-based BIC when available
        return getattr(model, "bic_llf", model.bic)
    else:
        raise ValueError("criterion must be 'aic' or 'bic'")


# -----------------------------
# Forward Stepwise Selection
# -----------------------------

def forward_stepwise(data, response, candidates, family, criterion="aic"):
    """
    Forward stepwise selection based on AIC or BIC.

    data      : pandas DataFrame
    response  : string, name of response variable
    candidates: list of predictor terms (e.g., ["PRICE", "BATHRM", "C(AC)", ...])
    family    : statsmodels family (e.g., sm.families.Binomial())
    criterion : "aic" or "bic"
    """
    selected = []
    best_ic = np.inf
    best_model = None
    best_formula = None

    remaining = candidates.copy()

    improved = True
    while improved and remaining:
        improved = False
        ic_candidates = []

        for term in remaining:
            formula_terms = selected + [term]
            formula = response + " ~ " + " + ".join(formula_terms)

            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                ic_candidates.append((ic, term, model, formula))
            except Exception:
                continue

        if not ic_candidates:
            break

        ic_candidates.sort(key=lambda x: x[0])
        best_candidate_ic, best_term, candidate_model, candidate_formula = ic_candidates[0]

        if best_candidate_ic + 1e-8 < best_ic:
            best_ic = best_candidate_ic
            best_model = candidate_model
            best_formula = candidate_formula
            selected.append(best_term)
            remaining.remove(best_term)
            improved = True
        else:
            improved = False

    return best_model, best_formula, selected, best_ic


# -----------------------------
# Backward Stepwise Selection
# -----------------------------

def backward_stepwise(data, response, initial_terms, family, criterion="aic"):
    """
    Backward stepwise selection based on AIC or BIC.

    data        : pandas DataFrame
    response    : string, name of response variable
    initial_terms: list of predictor terms to start with
    family      : statsmodels family
    criterion   : "aic" or "bic"
    """
    selected = initial_terms.copy()
    formula = response + " ~ " + " + ".join(selected)

    try:
        model = smf.glm(formula=formula, data=data, family=family).fit()
    except Exception as e:
        raise RuntimeError(f"Initial model failed to fit: {e}")

    best_ic = _get_ic(model, criterion)
    best_model = model
    best_formula = formula

    improved = True
    while improved and len(selected) > 1:
        improved = False
        ic_candidates = []

        for term in selected:
            trial_terms = [t for t in selected if t != term]
            formula = response + " ~ " + " + ".join(trial_terms)

            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                ic_candidates.append((ic, term, model, formula))
            except Exception:
                continue

        if not ic_candidates:
            break

        ic_candidates.sort(key=lambda x: x[0])
        best_candidate_ic, removed_term, candidate_model, candidate_formula = ic_candidates[0]

        if best_candidate_ic + 1e-8 < best_ic:
            best_ic = best_candidate_ic
            best_model = candidate_model
            best_formula = candidate_formula
            selected.remove(removed_term)
            improved = True
        else:
            improved = False

    return best_model, best_formula, selected, best_ic


# -----------------------------
# All-Subsets Search (AIC/BIC)
# -----------------------------

def all_subsets_ic(data, response, candidates, family, criterion="aic"):
    """
    All-subsets model selection based on AIC or BIC.

    data      : pandas DataFrame
    response  : string, name of response variable
    candidates: list of predictor terms
    family    : statsmodels family
    criterion : "aic" or "bic"
    """
    import itertools

    best_ic = np.inf
    best_model = None
    best_formula = None
    best_subset = None

    for k in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, k):
            formula = response + " ~ " + " + ".join(subset)

            try:
                model = smf.glm(formula=formula, data=data, family=family).fit()
                ic = _get_ic(model, criterion)
                if ic < best_ic:
                    best_ic = ic
                    best_model = model
                    best_formula = formula
                    best_subset = list(subset)
            except Exception:
                continue

    return best_model, best_formula, best_subset, best_ic

# ============================================================
# MODULE 4 — BEST SUBSET SELECTION (regsubsets-style)
# ============================================================

# -----------------------------
# Compute model metrics
# -----------------------------

def compute_metrics(model, n):
    """
    Compute AIC, BIC, Adjusted R², Cp for a fitted statsmodels model.
    """
    rss = np.sum(model.resid ** 2)
    k = model.df_model + 1  # includes intercept
    sigma2 = rss / (n - k)

    metrics = {
        "AIC": model.aic,
        "BIC": model.bic_llf if hasattr(model, "bic_llf") else model.bic,
        "Adj_R2": model.rsquared_adj if hasattr(model, "rsquared_adj") else np.nan,
        "Cp": rss / sigma2 - (n - 2 * k)
    }
    return metrics


# -----------------------------
# Best Subset Selection
# -----------------------------

def best_subset_selection(data, response, predictors, max_features=None):
    """
    Perform best subset selection (like regsubsets in R).

    data       : pandas DataFrame
    response   : string, response variable
    predictors : list of predictor names
    max_features : maximum number of predictors to consider
    """
    if max_features is None:
        max_features = len(predictors)

    results = []
    n = data.shape[0]

    for k in range(1, max_features + 1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)

            try:
                model = smf.ols(formula=formula, data=data).fit()
                metrics = compute_metrics(model, n)
                results.append({
                    "subset": subset,
                    "k": k,
                    "AIC": metrics["AIC"],
                    "BIC": metrics["BIC"],
                    "Adj_R2": metrics["Adj_R2"],
                    "Cp": metrics["Cp"],
                    "model": model,
                    "formula": formula
                })
            except Exception:
                continue

    return pd.DataFrame(results)


# -----------------------------
# Plotting function (like leaps::plot.regsubsets)
# -----------------------------

def plot_subset_metric(results_df, metric="BIC"):
    """
    Plot model size vs. metric (AIC, BIC, Adj_R2, Cp).
    """
    plt.figure(figsize=(8, 5))
    grouped = results_df.groupby("k")[metric].min()

    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Number of Predictors")
    plt.ylabel(metric)
    plt.title(f"Best Subset Selection — {metric}")
    plt.grid(True)
    plt.show()
    
# ============================================================
# MODULE 5 — PARTIAL F-TESTS (ANOVA) FOR NESTED MODELS
# ============================================================

def partial_f_test(data, full_formula, reduced_formula):
    """
    Perform a partial F-test comparing a full model vs. a reduced model.

    data           : pandas DataFrame
    full_formula   : string, e.g. "PRICE_10K ~ BATHRM + ROOMS + BEDRM + ..."
    reduced_formula: string, nested version of full model

    Returns: ANOVA comparison table (like R's anova(full, reduced))
    """
    # Fit both models
    full_model = smf.ols(full_formula, data=data).fit()
    reduced_model = smf.ols(reduced_formula, data=data).fit()

    # Perform ANOVA comparison
    anova_results = anova_lm(reduced_model, full_model)

    return anova_results, full_model, reduced_model

full_formula = (
    "PRICE_10K ~ BATHRM + ROOMS + BEDRM + STORIES + QUALIFIED + GRADE + "
    "KITCHENS + FIREPLACES + WARD + LATITUDE + LONGITUDE + "
    "AYB_age + EYB_age + REMODEL_age + CONDITION"
)

reduced_formula = (
    "PRICE_10K ~ BATHRM + ROOMS + BEDRM + STORIES + QUALIFIED + GRADE + "
    "KITCHENS + FIREPLACES + WARD + LATITUDE + LONGITUDE + "
    "AYB_age + EYB_age + REMODEL_age"
)

anova_table, full_model, reduced_model = partial_f_test(
    data=DC_train,
    full_formula=full_formula,
    reduced_formula=reduced_formula
)

print(anova_table)

# ============================================================
# MODULE 6 — POLYNOMIAL REGRESSION (R poly() → Python)
# ============================================================

# ------------------------------------------------------------
# 1. Polynomial regression using statsmodels (R-style formulas)
# ------------------------------------------------------------

def poly_regression_statsmodels(data, response, predictor, degree):
    """
    Fit a polynomial regression using statsmodels formula interface.
    Equivalent to R: lm(y ~ poly(x, degree))
    """
    # Build formula: PRICE_10K ~ np.power(BATHRM, 1) + ... + np.power(BATHRM, degree)
    terms = " + ".join([f"np.power({predictor}, {d})" for d in range(1, degree + 1)])
    formula = f"{response} ~ {terms}"

    model = smf.ols(formula=formula, data=data).fit()
    return model, formula


# ------------------------------------------------------------
# 2. Polynomial regression using sklearn Pipeline
# ------------------------------------------------------------

def poly_regression_sklearn(X, y, degree):
    """
    Fit polynomial regression using sklearn Pipeline.
    Equivalent to R: lm(y ~ poly(x, degree))
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    model.fit(X, y)
    return model


# ------------------------------------------------------------
# 3. Example usage (using your DC_train dataset)
# ------------------------------------------------------------

# Example 1 — Polynomial regression for BATHRM (degree 8, like your R code)
model_bathrm, formula_bathrm = poly_regression_statsmodels(
    data=DC_train,
    response="PRICE_10K",
    predictor="BATHRM",
    degree=8
)

print("\n===== Polynomial Regression (Statsmodels) — BATHRM Degree 8 =====")
print("Formula:", formula_bathrm)
print(model_bathrm.summary())


# Example 2 — Polynomial regression for BEDRM (degree 8)
model_bedrm, formula_bedrm = poly_regression_statsmodels(
    data=DC_train,
    response="PRICE_10K",
    predictor="BEDRM",
    degree=8
)

print("\n===== Polynomial Regression (Statsmodels) — BEDRM Degree 8 =====")
print("Formula:", formula_bedrm)
print(model_bedrm.summary())


# Example 3 — sklearn polynomial regression (AYB_age degree 4)
X = DC_train[["AYB_age"]]
y = DC_train["PRICE_10K"]

poly_model_ayb = poly_regression_sklearn(X, y, degree=4)

print("\n===== Polynomial Regression (sklearn) — AYB_age Degree 4 =====")
print("Coefficients:", poly_model_ayb.named_steps["linreg"].coef_)
print("Intercept:", poly_model_ayb.named_steps["linreg"].intercept_)


# ------------------------------------------------------------
# 4. Example: Nonlinear transformations (sqrt, log, power)
# ------------------------------------------------------------

model_nonlinear = smf.ols(
    formula=(
        "PRICE_10K ~ np.sqrt(BATHRM) + np.log(ROOMS + 1) + "
        "np.power(BEDRM, 0.5) + STORIES"
    ),
    data=DC_train
).fit()

print("\n===== Nonlinear Transformations Model =====")
print(model_nonlinear.summary())

# ============================================================
# MODULE 7 — RIDGE & LASSO REGRESSION (glmnet in Python)
# ============================================================

# ------------------------------------------------------------
# 1. Helper: Build X and y matrices
# ------------------------------------------------------------

def build_xy(data, response, predictors):
    """
    Build X and y matrices for sklearn models.
    """
    X = data[predictors].copy()
    y = data[response].values
    return X, y


# ------------------------------------------------------------
# 2. Ridge Regression (Cross-Validated)
# ------------------------------------------------------------

def ridge_regression_cv(X_train, y_train, alphas=None):
    """
    Fit Ridge regression with cross-validation.
    Equivalent to R: glmnet(alpha=0)
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 200)

    ridge_model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, store_cv_values=True))
    ])

    ridge_model.fit(X_train, y_train)
    return ridge_model


# ------------------------------------------------------------
# 3. LASSO Regression (Cross-Validated)
# ------------------------------------------------------------

def lasso_regression_cv(X_train, y_train, alphas=None):
    """
    Fit LASSO regression with cross-validation.
    Equivalent to R: glmnet(alpha=1)
    """
    if alphas is None:
        alphas = np.logspace(-3, 1, 200)

    lasso_model = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=alphas, cv=10, max_iter=5000))
    ])

    lasso_model.fit(X_train, y_train)
    return lasso_model


# ------------------------------------------------------------
# 4. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

response = "PRICE_10K"

# Build matrices
X_train, y_train = build_xy(DC_train, response, predictors)
X_test, y_test   = build_xy(DC_test,  response, predictors)

# ------------------------------------------------------------
# RIDGE REGRESSION
# ------------------------------------------------------------

ridge_model = ridge_regression_cv(X_train, y_train)

ridge_alpha = ridge_model.named_steps["ridge"].alpha_
ridge_coef  = ridge_model.named_steps["ridge"].coef_

print("\n===== RIDGE REGRESSION RESULTS =====")
print("Optimal alpha (lambda):", ridge_alpha)
print("Coefficients:", ridge_coef)

# Test MSE
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Test MSE:", ridge_mse)


# ------------------------------------------------------------
# LASSO REGRESSION
# ------------------------------------------------------------

lasso_model = lasso_regression_cv(X_train, y_train)

lasso_alpha = lasso_model.named_steps["lasso"].alpha_
lasso_coef  = lasso_model.named_steps["lasso"].coef_

print("\n===== LASSO REGRESSION RESULTS =====")
print("Optimal alpha (lambda):", lasso_alpha)
print("Coefficients:", lasso_coef)

# Test MSE
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Test MSE:", lasso_mse)

# ============================================================
# MODULE 8 — PCR & PLS (Principal Components Regression / Partial Least Squares)
# ============================================================

# ------------------------------------------------------------
# 1. PCR (Principal Components Regression)
# ------------------------------------------------------------

def pcr_regression(X_train, y_train, X_test, y_test, n_components):
    """
    Perform Principal Components Regression (PCR).
    Equivalent to R: pcr(..., validation="CV")
    """
    pcr_model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("linreg", LinearRegression())
    ])

    pcr_model.fit(X_train, y_train)
    preds = pcr_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return pcr_model, mse


# ------------------------------------------------------------
# 2. Scree Plot (like R's screeplot(princomp()))
# ------------------------------------------------------------

def scree_plot(X):
    """
    Plot explained variance ratio for PCA components.
    Equivalent to R: screeplot(princomp(X))
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA().fit(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot — PCA")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# 3. PLS Regression (Partial Least Squares)
# ------------------------------------------------------------

def pls_regression(X_train, y_train, X_test, y_test, n_components):
    """
    Perform Partial Least Squares Regression.
    Equivalent to R: plsr(..., validation="CV")
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    preds = pls.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return pls, mse


# ------------------------------------------------------------
# 4. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

response = "PRICE_10K"

# Build matrices
X_train = DC_train[predictors].values
y_train = DC_train[response].values
X_test  = DC_test[predictors].values
y_test  = DC_test[response].values

# ------------------------------------------------------------
# PCR Example (choose 5 components)
# ------------------------------------------------------------

pcr_model, pcr_mse = pcr_regression(
    X_train, y_train,
    X_test, y_test,
    n_components=5
)

print("\n===== PCR RESULTS =====")
print("Test MSE:", pcr_mse)
print("Explained variance (first 5 PCs):")
print(pcr_model.named_steps["pca"].explained_variance_ratio_)


# ------------------------------------------------------------
# Scree Plot (PCA)
# ------------------------------------------------------------

print("\n===== Scree Plot (PCA) =====")
scree_plot(DC_train[predictors])


# ------------------------------------------------------------
# PLS Example (choose 5 components)
# ------------------------------------------------------------

pls_model, pls_mse = pls_regression(
    X_train, y_train,
    X_test, y_test,
    n_components=5
)

print("\n===== PLS RESULTS =====")
print("Test MSE:", pls_mse)
print("PLS Coefficients:")
print(pls_model.coef_)

# ============================================================
# MODULE 9 — KNN CLASSIFICATION (R knn() → Python)
# ============================================================

# ------------------------------------------------------------
# 1. Build classification labels (like your R "Residential")
# ------------------------------------------------------------

def build_price_classes(data, response="PRICE_10K"):
    """
    Create price categories similar to your R code:
    Reasonable vs Expensive (median split).
    """
    median_price = data[response].median()
    labels = np.where(data[response] < median_price, "Reasonable", "Expensive")
    return labels


# ------------------------------------------------------------
# 2. KNN classifier function
# ------------------------------------------------------------

def knn_classify(X_train, y_train, X_test, y_test, k):
    """
    Fit KNN classifier with scaling.
    Equivalent to R: knn(train, test, cl, k)
    """
    knn_model = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    knn_model.fit(X_train, y_train)
    preds = knn_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return preds, acc, cm


# ------------------------------------------------------------
# 3. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

# Build X and y
X_train = DC_train[predictors].values
X_test  = DC_test[predictors].values

y_train = build_price_classes(DC_train)
y_test  = build_price_classes(DC_test)

# ------------------------------------------------------------
# KNN Example (k = 5)
# ------------------------------------------------------------

preds_5, acc_5, cm_5 = knn_classify(X_train, y_train, X_test, y_test, k=5)

print("\n===== KNN (k = 5) =====")
print("Accuracy:", acc_5)
print("Confusion Matrix:\n", cm_5)


# ------------------------------------------------------------
# 4. K-sweep (k = 1 to 100)
# ------------------------------------------------------------

k_values = range(1, 101)
accuracies = []

for k in k_values:
    _, acc, _ = knn_classify(X_train, y_train, X_test, y_test, k)
    accuracies.append(acc)

# Print best K
best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)

print("\n===== K-SWEEP RESULTS =====")
print("Best K:", best_k)
print("Best Accuracy:", best_acc)

# Optional: Plot accuracy vs K
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.grid(True)
plt.show()

# ============================================================
# MODULE 10 — LDA & QDA CLASSIFICATION (R lda/qda → Python)
# ============================================================

# ------------------------------------------------------------
# 1. Build classification labels (same as R "Residential")
# ------------------------------------------------------------

def build_price_classes(data, response="PRICE_10K"):
    """
    Create price categories similar to your R code:
    Reasonable vs Expensive (median split).
    """
    median_price = data[response].median()
    labels = np.where(data[response] < median_price, "Reasonable", "Expensive")
    return labels


# ------------------------------------------------------------
# 2. LDA classifier
# ------------------------------------------------------------

def lda_classify(X_train, y_train, X_test, y_test, priors=None):
    """
    Fit LDA classifier with optional priors.
    Equivalent to R: lda(..., CV=TRUE)
    """
    lda_model = Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(priors=priors))
    ])

    lda_model.fit(X_train, y_train)
    preds = lda_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return preds, acc, cm, lda_model


# ------------------------------------------------------------
# 3. QDA classifier
# ------------------------------------------------------------

def qda_classify(X_train, y_train, X_test, y_test, priors=None):
    """
    Fit QDA classifier with optional priors.
    Equivalent to R: qda(..., CV=TRUE)
    """
    qda_model = Pipeline([
        ("scale", StandardScaler()),
        ("qda", QuadraticDiscriminantAnalysis(priors=priors))
    ])

    qda_model.fit(X_train, y_train)
    preds = qda_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return preds, acc, cm, qda_model


# ------------------------------------------------------------
# 4. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

# Build X and y
X_train = DC_train[predictors].values
X_test  = DC_test[predictors].values

y_train = build_price_classes(DC_train)
y_test  = build_price_classes(DC_test)

# ------------------------------------------------------------
# LDA Example (no priors)
# ------------------------------------------------------------

preds_lda, acc_lda, cm_lda, lda_model = lda_classify(
    X_train, y_train, X_test, y_test
)

print("\n===== LDA RESULTS (No Priors) =====")
print("Accuracy:", acc_lda)
print("Confusion Matrix:\n", cm_lda)


# ------------------------------------------------------------
# LDA Example (with priors)
# ------------------------------------------------------------

priors = [0.55, 0.45]  # same as your R code

preds_lda_p, acc_lda_p, cm_lda_p, lda_model_p = lda_classify(
    X_train, y_train, X_test, y_test, priors=priors
)

print("\n===== LDA RESULTS (With Priors) =====")
print("Accuracy:", acc_lda_p)
print("Confusion Matrix:\n", cm_lda_p)


# ------------------------------------------------------------
# QDA Example (no priors)
# ------------------------------------------------------------

preds_qda, acc_qda, cm_qda, qda_model = qda_classify(
    X_train, y_train, X_test, y_test
)

print("\n===== QDA RESULTS (No Priors) =====")
print("Accuracy:", acc_qda)
print("Confusion Matrix:\n", cm_qda)


# ------------------------------------------------------------
# QDA Example (with priors)
# ------------------------------------------------------------

preds_qda_p, acc_qda_p, cm_qda_p, qda_model_p = qda_classify(
    X_train, y_train, X_test, y_test, priors=priors
)

print("\n===== QDA RESULTS (With Priors) =====")
print("Accuracy:", acc_qda_p)
print("Confusion Matrix:\n", cm_qda_p)

# ============================================================
# MODULE 11 — SUPPORT VECTOR MACHINES (R svm() → Python)
# ============================================================

# ------------------------------------------------------------
# 1. Build classification labels (same as R "HOUSE")
# ------------------------------------------------------------

def build_house_classes(data, response="PRICE_10K"):
    """
    Create binary classes based on PRICE_10K threshold.
    Equivalent to your R code:
    HOUSE = ifelse(PRICE_10K > 44.365, "Over", "Under")
    """
    threshold = data[response].median()
    labels = np.where(data[response] > threshold, "Over", "Under")
    return labels


# ------------------------------------------------------------
# 2. SVM classifier function
# ------------------------------------------------------------

def svm_classify(X_train, y_train, X_test, y_test, kernel="linear", C=1.0, degree=3):
    """
    Fit SVM classifier with scaling.
    Equivalent to R: svm(..., kernel="linear"/"polynomial"/"radial"/"sigmoid")
    """
    svm_model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, degree=degree))
    ])

    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return preds, acc, cm, svm_model


# ------------------------------------------------------------
# 3. SVM tuning (equivalent to R's tune())
# ------------------------------------------------------------

def svm_tune(X_train, y_train):
    """
    Perform hyperparameter tuning for SVM.
    Equivalent to R:
    tune(svm, ..., ranges=list(cost=10^seq(-3,3), kernel=c(...)))
    """
    param_grid = {
        "svm__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "svm__C": np.logspace(-3, 3, 7),
        "svm__degree": [2, 3, 4]
    }

    svm_model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC())
    ])

    grid = GridSearchCV(
        svm_model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


# ------------------------------------------------------------
# 4. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

# Build X and y
X_train = DC_train[predictors].values
X_test  = DC_test[predictors].values

y_train = build_house_classes(DC_train)
y_test  = build_house_classes(DC_test)

# ------------------------------------------------------------
# SVM Example — Linear Kernel
# ------------------------------------------------------------

preds_lin, acc_lin, cm_lin, svm_lin = svm_classify(
    X_train, y_train, X_test, y_test, kernel="linear"
)

print("\n===== SVM (Linear Kernel) =====")
print("Accuracy:", acc_lin)
print("Confusion Matrix:\n", cm_lin)


# ------------------------------------------------------------
# SVM Example — Polynomial Kernel
# ------------------------------------------------------------

preds_poly, acc_poly, cm_poly, svm_poly = svm_classify(
    X_train, y_train, X_test, y_test, kernel="poly", degree=3
)

print("\n===== SVM (Polynomial Kernel) =====")
print("Accuracy:", acc_poly)
print("Confusion Matrix:\n", cm_poly)


# ------------------------------------------------------------
# SVM Example — Radial (RBF) Kernel
# ------------------------------------------------------------

preds_rbf, acc_rbf, cm_rbf, svm_rbf = svm_classify(
    X_train, y_train, X_test, y_test, kernel="rbf"
)

print("\n===== SVM (Radial RBF Kernel) =====")
print("Accuracy:", acc_rbf)
print("Confusion Matrix:\n", cm_rbf)


# ------------------------------------------------------------
# SVM Example — Sigmoid Kernel
# ------------------------------------------------------------

preds_sig, acc_sig, cm_sig, svm_sig = svm_classify(
    X_train, y_train, X_test, y_test, kernel="sigmoid"
)

print("\n===== SVM (Sigmoid Kernel) =====")
print("Accuracy:", acc_sig)
print("Confusion Matrix:\n", cm_sig)


# ------------------------------------------------------------
# SVM Tuning (GridSearchCV)
# ------------------------------------------------------------

print("\n===== SVM TUNING (GridSearchCV) =====")
svm_grid = svm_tune(X_train, y_train)

print("Best Parameters:", svm_grid.best_params_)
print("Best Accuracy:", svm_grid.best_score_)

# ============================================================
# MODULE 12 — TREES, RANDOM FORESTS, BAGGING (R tree() → Python)
# ============================================================

# ------------------------------------------------------------
# 1. Build classification labels (same as R "PRICE.CLASS")
# ------------------------------------------------------------

def build_price_class_binary(data, response="PRICE_10K"):
    """
    Equivalent to R:
    PRICE.CLASS = ifelse(PRICE_10K > median, "UnderPriced", "OverPriced")
    """
    median_price = data[response].median()
    labels = np.where(data[response] > median_price, "OverPriced", "UnderPriced")
    return labels


# ------------------------------------------------------------
# 2. Classification Tree
# ------------------------------------------------------------

def classification_tree(X_train, y_train, X_test, y_test, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return clf, preds, acc, cm


# ------------------------------------------------------------
# 3. Regression Tree
# ------------------------------------------------------------

def regression_tree(X_train, y_train, X_test, y_test, max_depth=None):
    reg = DecisionTreeRegressor(max_depth=max_depth)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    return reg, preds, mse


# ------------------------------------------------------------
# 4. Tree Cross-Validation (equivalent to R's cv.tree())
# ------------------------------------------------------------

def tree_cross_validation(X, y):
    """
    Perform CV over tree depths.
    Equivalent to R: cv.tree(tree.model)
    """
    depths = range(1, 21)
    cv_scores = []

    for d in depths:
        model = DecisionTreeRegressor(max_depth=d)
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        cv_scores.append(np.mean(scores))

    best_depth = depths[np.argmax(cv_scores)]
    return depths, cv_scores, best_depth


# ------------------------------------------------------------
# 5. Random Forests (Regression)
# ------------------------------------------------------------

def random_forest_regression(X_train, y_train, X_test, y_test, mtry=None, ntrees=500):
    """
    Equivalent to R:
    randomForest(..., mtry=..., ntree=...)
    """
    if mtry is None:
        mtry = int(np.sqrt(X_train.shape[1]))

    rf = RandomForestRegressor(
        n_estimators=ntrees,
        max_features=mtry,
        random_state=123
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return rf, preds, mse


# ------------------------------------------------------------
# 6. Bagging (Random Forest with mtry = p)
# ------------------------------------------------------------

def bagging_regression(X_train, y_train, X_test, y_test, ntrees=500):
    """
    Equivalent to R:
    randomForest(..., mtry=p)
    """
    bag = BaggingRegressor(
        n_estimators=ntrees,
        random_state=123
    )

    bag.fit(X_train, y_train)
    preds = bag.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return bag, preds, mse


# ------------------------------------------------------------
# 7. Example usage with your DC_train / DC_test
# ------------------------------------------------------------

predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES", "KITCHENS",
    "FIREPLACES", "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

response = "PRICE_10K"

# Build matrices
X_train = DC_train[predictors].values
X_test  = DC_test[predictors].values

y_train_reg = DC_train[response].values
y_test_reg  = DC_test[response].values

y_train_clf = build_price_class_binary(DC_train)
y_test_clf  = build_price_class_binary(DC_test)

# ------------------------------------------------------------
# Classification Tree
# ------------------------------------------------------------

clf_tree, preds_clf, acc_clf, cm_clf = classification_tree(
    X_train, y_train_clf, X_test, y_test_clf
)

print("\n===== CLASSIFICATION TREE =====")
print("Accuracy:", acc_clf)
print("Confusion Matrix:\n", cm_clf)

plt.figure(figsize=(12, 6))
plot_tree(clf_tree, feature_names=predictors, class_names=["UnderPriced", "OverPriced"], filled=True)
plt.title("Classification Tree")
plt.show()


# ------------------------------------------------------------
# Regression Tree
# ------------------------------------------------------------

reg_tree, preds_reg, mse_reg = regression_tree(
    X_train, y_train_reg, X_test, y_test_reg
)

print("\n===== REGRESSION TREE =====")
print("Test MSE:", mse_reg)

plt.figure(figsize=(12, 6))
plot_tree(reg_tree, feature_names=predictors, filled=True)
plt.title("Regression Tree")
plt.show()


# ------------------------------------------------------------
# Tree Cross-Validation
# ------------------------------------------------------------

depths, cv_scores, best_depth = tree_cross_validation(X_train, y_train_reg)

print("\n===== TREE CROSS-VALIDATION =====")
print("Best Depth:", best_depth)

plt.figure(figsize=(8, 5))
plt.plot(depths, cv_scores, marker="o")
plt.xlabel("Tree Depth")
plt.ylabel("CV Score (Negative MSE)")
plt.title("Tree Cross-Validation")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Random Forest Regression
# ------------------------------------------------------------

rf_model, rf_preds, rf_mse = random_forest_regression(
    X_train, y_train_reg, X_test, y_test_reg, mtry=4, ntrees=500
)

print("\n===== RANDOM FOREST REGRESSION =====")
print("Test MSE:", rf_mse)
print("Variable Importance:", rf_model.feature_importances_)

plt.figure(figsize=(8, 5))
plt.barh(predictors, rf_model.feature_importances_)
plt.title("Random Forest Variable Importance")
plt.show()


# ------------------------------------------------------------
# Bagging Regression
# ------------------------------------------------------------

bag_model, bag_preds, bag_mse = bagging_regression(
    X_train, y_train_reg, X_test, y_test_reg, ntrees=500
)

print("\n===== BAGGING REGRESSION =====")
print("Test MSE:", bag_mse)

# ============================================================
# MODULE 13 — VISUALIZATION SUITE (R → Python)
# ============================================================

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. Histograms (R: hist())
# ------------------------------------------------------------

def plot_histograms(data, vars_list):
    n = len(vars_list)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for ax, var in zip(axes, vars_list):
        sns.histplot(data[var], kde=True, ax=ax)
        ax.set_title(f"Histogram — {var}")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 2. QQ Plots (R: qqnorm(), qqline())
# ------------------------------------------------------------

def plot_qq(data, vars_list):
    n = len(vars_list)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for ax, var in zip(axes, vars_list):
        stats.probplot(data[var], dist="norm", plot=ax)
        ax.set_title(f"QQ Plot — {var}")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3. Scatterplots (R: plot(x, y))
# ------------------------------------------------------------

def scatter_plot(data, x, y, hue=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, s=40)
    plt.title(f"{y} vs {x}")
    plt.show()


# ------------------------------------------------------------
# 4. Correlation Heatmap (R: cor(), heatmap())
# ------------------------------------------------------------

def correlation_heatmap(data, vars_list):
    corr = data[vars_list].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


# ------------------------------------------------------------
# 5. Pairplot (R: pairs())
# ------------------------------------------------------------

def pair_plot(data, vars_list):
    sns.pairplot(data[vars_list], diag_kind="kde")
    plt.show()


# ------------------------------------------------------------
# 6. Geospatial Scatterplots (R: scatter LATITUDE/LONGITUDE)
# ------------------------------------------------------------

def plot_geospatial(data, hue=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="LONGITUDE",
        y="LATITUDE",
        hue=hue,
        s=20,
        alpha=0.7
    )
    plt.title("Geospatial Plot — LONGITUDE vs LATITUDE")
    plt.show()


# ------------------------------------------------------------
# 7. Regression Diagnostics (R: plot(model))
# ------------------------------------------------------------

def regression_diagnostics(model):
    """
    Equivalent to R's plot(final_model)
    Produces:
    - Residuals vs Fitted
    - QQ Plot
    - Scale-Location
    - Cook's Distance
    """
    fitted = model.fittedvalues
    residuals = model.resid
    std_resid = model.get_influence().resid_studentized_internal
    cooks = model.get_influence().cooks_distance[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
    axes[0, 0].axhline(0, color="red")
    axes[0, 0].set_title("Residuals vs Fitted")

    # QQ Plot
    sm.qqplot(std_resid, line="45", ax=axes[0, 1])
    axes[0, 1].set_title("QQ Plot")

    # Scale-Location
    sns.scatterplot(x=fitted, y=np.sqrt(np.abs(std_resid)), ax=axes[1, 0])
    axes[1, 0].set_title("Scale-Location")

    # Cook's Distance
    axes[1, 1].stem(cooks, markerfmt=",")
    axes[1, 1].set_title("Cook's Distance")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. Example usage with your DC dataset
# ------------------------------------------------------------

numeric_vars = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES",
    "KITCHENS", "FIREPLACES",
    "AYB_age", "EYB_age", "REMODEL_age"
]

# Histograms
plot_histograms(DC_train, numeric_vars)

# QQ Plots
plot_qq(DC_train, numeric_vars)

# Correlation Heatmap
correlation_heatmap(DC_train, numeric_vars)

# Pairplot
pair_plot(DC_train, ["PRICE_10K"] + numeric_vars[:4])

# Geospatial Plot
plot_geospatial(DC_train, hue="WARD")

# Regression Diagnostics (example using your final model)
# final_model = smf.ols("PRICE_10K ~ BATHRM + ROOMS + ...", data=DC_train).fit()
# regression_diagnostics(final_model)

# ============================================================
# MODULE 14 — FINAL MODEL COMPARISON TABLE (SORT + HIGHLIGHT)
# ============================================================

# ------------------------------------------------------------
# Helper: Safe MSE (for classification models)
# ------------------------------------------------------------
def safe_mse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred)
    except:
        return np.nan

# ------------------------------------------------------------
# Helper: Safe Accuracy (for regression models)
# ------------------------------------------------------------
def safe_accuracy(y_true, y_pred):
    try:
        return accuracy_score(y_true, y_pred)
    except:
        return np.nan

# ------------------------------------------------------------
# Build comparison row
# ------------------------------------------------------------
def model_row(name, y_test, y_pred, notes=""):
    return {
        "Model": name,
        "Test MSE": safe_mse(y_test, y_pred),
        "Accuracy": safe_accuracy(y_test, y_pred),
        "Notes": notes
    }

# ------------------------------------------------------------
# Build comparison table (raw DataFrame)
# ------------------------------------------------------------
def build_model_comparison(models_dict, y_test):
    rows = []
    for name, (preds, notes) in models_dict.items():
        rows.append(model_row(name, y_test, preds, notes))
    df = pd.DataFrame(rows)

    df_sorted = df.sort_values(
        by=["Test MSE", "Accuracy"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    return df_sorted

# ------------------------------------------------------------
# Highlight best models (lowest MSE, highest Accuracy)
# ------------------------------------------------------------
def highlight_best_models(df):
    styled = df.style

    if df["Test MSE"].notna().any():
        min_mse = df["Test MSE"].min()
        styled = styled.apply(
            lambda row: ["background-color: #d4f4dd" if row["Test MSE"] == min_mse else "" for _ in row],
            axis=1
        )

    if df["Accuracy"].notna().any():
        max_acc = df["Accuracy"].max()
        styled = styled.apply(
            lambda row: ["background-color: #d0e7ff" if row["Accuracy"] == max_acc else "" for _ in row],
            axis=1
        )

    return styled

# ------------------------------------------------------------
# Master function: Build + Sort + Highlight
# ------------------------------------------------------------
def final_model_table(models_dict, y_test):
    df_sorted = build_model_comparison(models_dict, y_test)
    styled = highlight_best_models(df_sorted)
    return df_sorted, styled

# ============================================================
# MODULE 15 — END‑TO‑END PIPELINE SCRIPT
# ============================================================

# -----------------------------
# 1. Load Data
# -----------------------------
DC = pd.read_csv("DC_housing_clean.csv")

# -----------------------------
# 2. Feature Engineering
# -----------------------------
predictors = [
    "BATHRM", "ROOMS", "BEDRM", "STORIES",
    "KITCHENS", "FIREPLACES",
    "LATITUDE", "LONGITUDE",
    "AYB_age", "EYB_age", "REMODEL_age"
]

response = "PRICE_10K"

X = DC[predictors].values
y = DC[response].values

# Classification labels
y_class = np.where(DC[response] > DC[response].median(), "Over", "Under")

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123
)

y_train_clf, y_test_clf = train_test_split(
    y_class, test_size=0.25, random_state=123
)

# -----------------------------
# 4. Fit Models
# -----------------------------

# OLS
ols_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
ols_preds = ols_model.predict(sm.add_constant(X_test))

# Ridge
ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_test)

# LASSO
lasso_model = LassoCV(cv=5).fit(X_train, y_train)
lasso_preds = lasso_model.predict(X_test)

# PCR
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pcr_model = LinearRegression().fit(X_train_pca, y_train)
pcr_preds = pcr_model.predict(X_test_pca)

# PLS
pls_model = PLSRegression(n_components=5).fit(X_train, y_train)
pls_preds = pls_model.predict(X_test).flatten()

# KNN
knn_model = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=7))
])
knn_model.fit(X_train, y_train_clf)
knn_preds = knn_model.predict(X_test)

# LDA
lda_model = LinearDiscriminantAnalysis().fit(X_train, y_train_clf)
lda_preds = lda_model.predict(X_test)

# QDA
qda_model = QuadraticDiscriminantAnalysis().fit(X_train, y_train_clf)
qda_preds = qda_model.predict(X_test)

# SVM
svm_model = SVC(kernel="rbf", C=1).fit(X_train, y_train_clf)
svm_preds = svm_model.predict(X_test)

# Regression Tree
tree_model = DecisionTreeRegressor(max_depth=6).fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=500, max_features=4).fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Bagging
bag_model = BaggingRegressor(n_estimators=500).fit(X_train, y_train)
bag_preds = bag_model.predict(X_test)

# -----------------------------
# 5. Build Comparison Table
# -----------------------------
models_dict = {
    "OLS": (ols_preds, "Final GLM"),
    "Ridge": (ridge_preds, f"alpha={ridge_model.alpha_}"),
    "LASSO": (lasso_preds, f"alpha={lasso_model.alpha_}"),
    "PCR": (pcr_preds, "5 components"),
    "PLS": (pls_preds, "5 components"),
    "KNN": (knn_preds, "k=7"),
    "LDA": (lda_preds, ""),
    "QDA": (qda_preds, ""),
    "SVM": (svm_preds, "RBF kernel"),
    "Regression Tree": (tree_preds, "depth=6"),
    "Random Forest": (rf_preds, "mtry=4"),
    "Bagging": (bag_preds, "500 trees")
}

df_final, styled_final = final_model_table(models_dict, y_test)

print(df_final)

# -----------------------------
# 6. Save Output
# -----------------------------
df_final.to_csv(
    r"C:\Users\aniec\Mirror\Programming Projects\2019.01_2024.05 DC Property Composite Analysis\model_comparison_results.csv",
    index=False
)

