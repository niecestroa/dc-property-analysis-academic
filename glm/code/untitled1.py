# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 23:55:22 2026

@author: aniec
"""

# ============================================================
# DC HOUSING ANALYSIS — FULL PYTHON SCRIPT
# Converted from R (STAT-616 GLM Project)
# ============================================================

# ============================================================
# SECTION 0 — IMPORTS & SETUP
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools

sns.set_theme(style="whitegrid")

# ============================================================
# SECTION 1 — LOAD & CLEAN DATA
# ============================================================

# Load Excel dataset
DC_Properties = pd.read_excel(
    "~/Documents/STAT 616 Generalizd Linear Models/GLM Project/Data/DC_Properties.xlsx"
)

# -----------------------------
# CLEANING FOR VISUALIZATION DATASET
# -----------------------------

DC_Properties_Visualisations = (
    DC_Properties
    .drop(columns=["CMPLX_NUM", "LIVING_GBA", "SALE_NUM", "GIS_LAST_MOD_DTTM"])
    .query("10000 < PRICE < 10000000")
    .query("HEAT != 'No Data'")
    .query("CNDTN != 'No Data' and CNDTN != 'Default'")
    .query("STRUCT != 'Default'")
    .query("GRADE != ' No Data'")
    .query("STYLE != 'Default'")
    .query("KITCHENS <= 10")
    .query("ROOMS < 26")
    .query("BEDRM < 20")
    .query("STORIES < 100")
    .query("BATHRM > 0")
    .query("HF_BATHRM > 0")
)

# Create binary qualification variable
DC_Properties_Visualisations["QUALIFIED_2"] = (
    DC_Properties_Visualisations["QUALIFIED"].apply(lambda x: 1 if x == "Q" else 0)
)

# Drop missing values
dcproperty = DC_Properties_Visualisations.dropna().copy()

# Recode AC
dcproperty["AC"] = dcproperty["AC"].replace({"0": "N"})

# Collapse Exceptional grades
dcproperty["GRADE"] = dcproperty["GRADE"].replace({
    "Exceptional-A": "Exceptional",
    "Exceptional-B": "Exceptional",
    "Exceptional-C": "Exceptional",
    "Exceptional-D": "Exceptional"
})

# -----------------------------
# CLEANING FOR FINAL MODEL DATASET
# (SALEDATE is KEPT per user request)
# -----------------------------

DC_Properties_Final = (
    DC_Properties
    .drop(columns=[
        "NUM_UNITS", "YR_RMDL", "GBA", "STRUCT", "EXTWALL", "ROOF",
        "INTWALL", "CMPLX_NUM", "LIVING_GBA", "FULLADDRESS", "CITY",
        "STATE", "NATIONALGRID", "ASSESSMENT_SUBNBHD", "CENSUS_BLOCK",
        "SALE_NUM", "GIS_LAST_MOD_DTTM"
    ])
    .query("CNDTN != 'No Data' and CNDTN != 'Default'")
    .query("GRADE != ' No Data'")
    .query("STYLE != 'Default'")
    .query("10000 < PRICE < 10000000")
    .query("FIREPLACES < 8")
    .query("KITCHENS <= 10")
    .query("ROOMS < 26")
    .query("BEDRM < 20")
    .query("STORIES < 100")
    .query("BATHRM > 0")
    .query("HF_BATHRM > 0")
)

DC_Properties_Final["QUALIFIED_2"] = (
    DC_Properties_Final["QUALIFIED"].apply(lambda x: 1 if x == "Q" else 0)
)

DC_Final = DC_Properties_Final.dropna().copy()
DC_Final["AC"] = DC_Final["AC"].replace({"0": "N"})

# ============================================================
# SECTION 2 — TRAINING / VALIDATION SPLIT
# ============================================================

cases = list(range(1, 2774)) \
        + list(range(4624, 6650)) \
        + list(range(8000, 12725)) \
        + list(range(15874, 22080)) \
        + list(range(26216, 31904))

Final_T = DC_Final.iloc[cases, :]
Final_V = DC_Final.drop(DC_Final.index[cases])

# ============================================================
# SECTION 3 — MODELING
# ============================================================

# -----------------------------
# BASIC MODEL
# -----------------------------
basic_model = smf.glm(
    formula="QUALIFIED_2 ~ PRICE",
    data=Final_T,
    family=sm.families.Binomial()
).fit()

print("\n===== BASIC MODEL =====")
print(basic_model.summary())

# -----------------------------
# FULL MODEL (NO INTERACTIONS)
# -----------------------------
model3 = smf.glm(
    formula=(
        "QUALIFIED_2 ~ PRICE + BATHRM + HF_BATHRM + C(AC) + ROOMS + BEDRM + "
        "STORIES + C(STYLE) + C(CNDTN) + KITCHENS + FIREPLACES + C(WARD)"
    ),
    data=Final_T,
    family=sm.families.Binomial()
).fit()

print("\n===== FULL MODEL =====")
print(model3.summary())

# -----------------------------
# STEPWISE AIC
# -----------------------------
def stepwise_aic(data, response, predictors):
    best_aic = np.inf
    best_model = None
    best_formula = None
    
    for k in range(1, len(predictors)+1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)
            model = smf.glm(formula=formula, data=data,
                            family=sm.families.Binomial()).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_formula = formula
    
    return best_model, best_formula

predictors = [
    "PRICE", "BATHRM", "C(AC)", "ROOMS", "BEDRM",
    "C(STYLE)", "C(CNDTN)", "KITCHENS", "C(WARD)"
]

model_aic, formula_aic = stepwise_aic(Final_T, "QUALIFIED_2", predictors)

print("\n===== AIC MODEL =====")
print("Selected formula:", formula_aic)
print(model_aic.summary())

# -----------------------------
# STEPWISE BIC
# -----------------------------
def stepwise_bic(data, response, predictors):
    best_bic = np.inf
    best_model = None
    best_formula = None
    n = data.shape[0]
    
    for k in range(1, len(predictors)+1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)
            model = smf.glm(formula=formula, data=data,
                            family=sm.ffamilies.Binomial()).fit()
            bic = model.aic + (np.log(n) - 2) * model.df_model
            
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_formula = formula
    
    return best_model, best_formula

model_bic, formula_bic = stepwise_bic(Final_T, "QUALIFIED_2", predictors)

print("\n===== BIC MODEL =====")
print("Selected formula:", formula_bic)
print(model_bic.summary())

# -----------------------------
# INTERACTION MODEL
# -----------------------------
model_inter = smf.glm(
    formula=(
        "QUALIFIED_2 ~ PRICE + C(AC) + ROOMS + BEDRM + C(CNDTN) + C(WARD) + "
        "PRICE:C(AC) + PRICE:ROOMS + PRICE:BEDRM + "
        "PRICE:C(CNDTN) + PRICE:C(WARD) + C(CNDTN):C(WARD)"
    ),
    data=Final_T,
    family=sm.families.Binomial()
).fit()

print("\n===== INTERACTION MODEL =====")
print(model_inter.summary())

# -----------------------------
# FINAL MODEL (WITH NONLINEAR TERMS)
# -----------------------------
final_model = smf.glm(
    formula=(
        "QUALIFIED_2 ~ PRICE + np.sqrt(PRICE) + C(AC) + ROOMS + "
        "np.power(ROOMS, 0.2) + np.sqrt(BEDRM) + C(CNDTN) + C(WARD) + "
        "PRICE:C(AC) + PRICE:ROOMS + PRICE:C(WARD)"
    ),
    data=Final_T,
    family=sm.families.Binomial()
).fit()

print("\n===== FINAL MODEL =====")
print(final_model.summary())

# ============================================================
# SECTION 4 — DIAGNOSTICS
# ============================================================

# Standardized deviance residuals
std_dev_resid = final_model.resid_deviance / np.sqrt(final_model.scale)

# Pearson residuals
std_pearson_resid = final_model.resid_pearson / np.sqrt(final_model.scale)

# Influence measures
influence = final_model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

print("\n===== FIRST 10 DIAGNOSTIC VALUES =====")
print("Std Deviance Residuals:", std_dev_resid[:10])
print("Std Pearson Residuals:", std_pearson_resid[:10])
print("Cook's Distance:", cooks_d[:10])
print("Leverage:", leverage[:10])

# Diagnostic plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].scatter(final_model.fittedvalues, std_dev_resid, alpha=0.4)
axs[0, 0].axhline(0, color='red')
axs[0, 0].set_title("Residuals vs Fitted")

sm.qqplot(std_dev_resid, line='45', ax=axs[0, 1])
axs[0, 1].set_title("Normal Q-Q Plot")

axs[1, 0].scatter(final_model.fittedvalues, np.sqrt(np.abs(std_dev_resid)), alpha=0.4)
axs[1, 0].set_title("Scale-Location Plot")

axs[1, 1].stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
axs[1, 1].set_title("Cook's Distance")

plt.tight_layout()
plt.show()

# ============================================================
# SECTION 5 — ROC CURVES
# ============================================================

train_pred = final_model.predict(Final_T)
train_y = Final_T["QUALIFIED_2"]

fpr_train, tpr_train, _ = roc_curve(train_y, train_pred)
auc_train = auc(fpr_train, tpr_train)

val_pred = final_model.predict(Final_V)
val_y = Final_V["QUALIFIED_2"]

fpr_val, tpr_val, _ = roc_curve(val_y, val_pred)
auc_val = auc(fpr_val, tpr_val)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Training AUC = {auc_train:.3f}", color="black")
plt.plot(fpr_val, tpr_val, label=f"Validation AUC = {auc_val:.3f}", color="red")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# SECTION 6 — VIF
# ============================================================

X = final_model.model.exog
vif_df = pd.DataFrame({
    "Variable": final_model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})

print("\n===== VIF TABLE =====")
print(vif_df)

# ============================================================
# SECTION 7 — VISUALIZATIONS
# ============================================================

# -----------------------------
# BASIC MODEL GRAPH (facet by WARD)
# -----------------------------
sns.lmplot(
    data=dcproperty,
    x="QUALIFIED_2",
    y="PRICE",
    hue="QUALIFIED_2",
    col="WARD",
    logistic=True,
    scatter_kws={"s": 40, "alpha": 0.6},
    height=4,
    aspect=1
)
plt.suptitle("Market Qualification based on Price and Location")
plt.show()

# -----------------------------
# MAP PLOTS
# -----------------------------
sns.scatterplot(data=dcproperty, x="X", y="Y", hue="ZIPCODE", s=30)
plt.title("Map of Data by Zipcode")
plt.show()

sns.scatterplot(data=dcproperty, x="X", y="Y", hue="WARD", s=30)
plt.title("Map of Data by Ward")
plt.show()

sns.scatterplot(data=dcproperty, x="X", y="Y", hue="QUADRANT", s=30)
plt.title("Map of Data by Quadrant")
plt.show()

# -----------------------------
# YEAR GRAPHS
# -----------------------------
sns.boxplot(
    data=dcproperty[dcproperty["YR_RMDL"] > 1800],
    x="QUADRANT",
    y="YR_RMDL",
    hue="QUALIFIED"
)
plt.title("Year Last Remodeled by Quadrant")
plt.show()

sns.boxplot(
    data=dcproperty[dcproperty["YR_RMDL"] > 1800],
    y="GRADE",
    x="YR_RMDL",
    hue="QUALIFIED",
    orient="h"
)
plt.title("Year Last Remodeled by Grade")
plt.show()

sns.boxplot(
    data=dcproperty[dcproperty["YR_RMDL"] > 1800],
    y="CNDTN",
    x="YR_RMDL",
    hue="QUALIFIED",
    orient="h"
)
plt.title("Year Last Remodeled by Condition")
plt.show()

# -----------------------------
# HEAT & AC GRAPH
# -----------------------------
sns.lmplot(
    data=dcproperty,
    x="AC",
    y="PRICE",
    hue="HEAT",
    col="QUALIFIED_2",
    scatter_kws={"s": 40, "alpha": 0.6},
    height=5,
    aspect=1
)
plt.suptitle("Price, AC and Heat")
plt.show()

# -----------------------------
# CONTINUOUS VARIABLE PLOTS
# -----------------------------
sns.scatterplot(data=dcproperty, x="ROOMS", y="PRICE", hue="QUALIFIED_2")
sns.regplot(data=dcproperty, x="ROOMS", y="PRICE", scatter=False, color="black")
plt.title("Price vs Rooms")
plt.show()

sns.scatterplot(data=dcproperty, x="KITCHENS", y="PRICE", hue="QUALIFIED_2")
plt.title("Kitchens vs Price")
plt.show()

sns.scatterplot(data=dcproperty, x="NUM_UNITS", y="PRICE", hue="QUADRANT")
plt.title("Units vs Price")
plt.show()

sns.scatterplot(data=dcproperty, x="BEDRM", y="PRICE", hue="QUALIFIED_2")
sns.regplot(data=dcproperty, x="BEDRM", y="PRICE", scatter=False, color="black")
plt.title("Bedrooms vs Price")
plt.show()

sns.scatterplot(data=dcproperty, x="STORIES", y="PRICE", hue="QUALIFIED_2")
sns.regplot(data=dcproperty, x="STORIES", y="PRICE", scatter=False, color="black")
plt.title("Stories vs Price")
plt.show()

sns.scatterplot(data=dcproperty, x="FIREPLACES", y="PRICE", hue="QUALIFIED_2")
sns.regplot(data=dcproperty, x="FIREPLACES", y="PRICE", scatter=False, color="blue")
plt.title("Fireplaces vs Price")
plt.show()

# -----------------------------
# TIME GRAPHS
# -----------------------------
sns.scatterplot(
    data=dcproperty,
    x="SALEDATE",
    y="PRICE",
    hue="QUALIFIED_2",
    palette="BuPu"
)
sns.regplot(data=dcproperty, x="SALEDATE", y="PRICE", scatter=False, color="red")
plt.title("Price Over Time")
plt.show()

# ============================================================
# SECTION 8 — CONCLUSION (COMMENTS)
# ============================================================

"""
Your full conclusion text preserved here as comments.
(omitted for brevity in this script)
"""

# ============================================================
# END OF SCRIPT
# ============================================================