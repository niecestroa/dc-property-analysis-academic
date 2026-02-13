# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 12, 2026
Editted on Thu Feb 13, 2026

@author: aniec
"""

# ============================================================
# DC HOUSING ANALYSIS — FULL PYTHON SCRIPT
# Converted from R to Python to help develop Python coding skills
# ============================================================

# ============================================================
# SECTION 0 — IMPORTS & SETUP
# ============================================================

import pandas as pd              # Data loading and manipulation
import numpy as np               # Numerical operations
import seaborn as sns            # Statistical visualization
import matplotlib.pyplot as plt  # Plotting library
import statsmodels.api as sm     # Core statsmodels API
import statsmodels.formula.api as smf  # R-style formula interface
from sklearn.metrics import roc_curve, auc  # ROC and AUC metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF diagnostics
import itertools                 # Efficient combinatorics utilities

# Enable LLF-based BIC to avoid FutureWarning and use the standard definition
import statsmodels.genmod.generalized_linear_model as glm   # Access GLM internals
glm.SET_USE_BIC_LLF(True)                                   # Use log-likelihood-based BIC (bic_llf)

sns.set_theme(style="whitegrid")

# ============================================================
# SECTION 1 — LOAD & CLEAN DATA
# ============================================================

DC_Properties = pd.read_excel(
    r"C:\Users\aniec\Mirror\Programming Projects\2019.01_2024.05 DC Property Composite Analysis\2019.05.20 R - DC Property using GLM\Data\DC_Properties.xlsx"
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

DC_Properties_Visualisations["QUALIFIED_2"] = (
    DC_Properties_Visualisations["QUALIFIED"].apply(lambda x: 1 if x == "Q" else 0)
)

dcproperty = DC_Properties_Visualisations.dropna().copy()

dcproperty["AC"] = dcproperty["AC"].replace({0: "N", "0": "N"})

dcproperty["GRADE"] = dcproperty["GRADE"].replace({
    "Exceptional-A": "Exceptional",
    "Exceptional-B": "Exceptional",
    "Exceptional-C": "Exceptional",
    "Exceptional-D": "Exceptional"
})

# -----------------------------
# CLEANING FOR FINAL MODEL DATASET
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
DC_Final["AC"] = DC_Final["AC"].replace({0: "N", "0": "N"})


# ============================================================
# SECTION 2 — TRAINING / VALIDATION SPLIT
# ============================================================

cases = (
    list(range(1, 2774)) +
    list(range(4624, 6650)) +
    list(range(8000, 12725)) +
    list(range(15874, 22080)) +
    list(range(26216, 31904))
)

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


# ============================================================
# STEPWISE AIC (ALL-SUBSETS)
# ============================================================

def stepwise_aic(data, response, predictors):
    best_aic = np.inf
    best_model = None
    best_formula = None

    for k in range(1, len(predictors)+1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)

            try:
                model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Binomial()
                ).fit()

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_formula = formula

            except Exception:
                continue

    return best_model, best_formula


predictors = [
    "PRICE", "BATHRM", "C(AC)", "ROOMS", "BEDRM",
    "C(STYLE)", "C(CNDTN)", "KITCHENS", "C(WARD)"
]

model_aic, formula_aic = stepwise_aic(Final_T, "QUALIFIED_2", predictors)

print("\n===== AIC MODEL =====")
print("Selected formula:", formula_aic)
print(model_aic.summary())


# ============================================================
# STEPWISE BIC (ALL-SUBSETS, LLF-BASED)
# ============================================================

def stepwise_bic(data, response, predictors):
    best_bic = np.inf
    best_model = None
    best_formula = None

    for k in range(1, len(predictors)+1):
        for subset in itertools.combinations(predictors, k):
            formula = response + " ~ " + " + ".join(subset)

            try:
                model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Binomial()
                ).fit()

                bic = model.bic_llf  # LLF-based BIC (correct + future-proof)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_formula = formula

            except Exception:
                continue

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
sns.scatterplot(data=dcproperty, x="LONGITUDE", y="LATITUDE", hue="ZIPCODE")
plt.title("Map of Data by Zipcode")
plt.show()

sns.scatterplot(data=dcproperty, x="LONGITUDE", y="LATITUDE", hue="WARD", s=30)
plt.title("Map of Data by Ward")
plt.show()

sns.scatterplot(data=dcproperty, x="LONGITUDE", y="LATITUDE", hue="QUADRANT", s=30)
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
To conclude we have created a binary logistic model for what determines whether a property is qualified enough to sell. Although the AIC is very high as 16,748, it seems that the model fits that data well. It was a long analysis process but we could complete it even with the time restrictions we had. Now we will answer the 7 questions we had at the beginning of this study. The answers to our seven questions were as follows: 1) We were not able to hear back from Chris, the provider of this dataset on Kaggle, we so can still not answer what the qualification column in the original dataset means. 2) The qualifications for a residential property to be sold on the market is that the paperwork is completed and submitted, the bank approves any transaction that the buyers and sellers need and the inspection of the property is passed. 3) From all our analysis so far, we can conclude that property pricing is the most important factor in determining whether a property is qualified to go on the market 4) From all our analysis so far, we can conclude realtors do care whether the property is qualified to sell and money is the most important to them. Since the more the realtor sells and the higher the price, the property is sold for the more money from the deal they receive. 4) We believe that we were creating the most optimal regression for modeling properties based on our response variable being qualification. Overall though a multiple linear type regression would work best when it comes to making housing, property, and apartment models. 5) We did follow the previous housing model approaches for predictor variables at the beginning of our model building but our analysis later had different predictor variables from those models creating using linear regression. 6) Yes, money is the most important thing. We will not say how this defines this world since we do not wish to be labeled as pessimistic people. Thankfully, we could answer our questions based on our analysis results and work, yet this does not mean we will stop the analysis being conducted. \
For future analysis, we would do many things differently and add many different types of things.  We will do the following things in the future: 1) conduct more time analysis and visualizations, 2) conduct some sentiment analysis on the street, neighborhood, and State one lives in since people are sometimes superstitious, 3) Try to see if we can hear back on what qualification meant in the dataset, 4) Add a few more variables – for example: Heat, and the interaction terms of heat and AC, 5) Collect data from realtor’s websites and fill in the information ourselves since we were losing data constantly, 6) Add neighborhood rating, neighborhood review 7) Collect data from the surrounding states (West Virginia, Virginia, Maryland). \
In conclusion, through all our analysis and graphs our model might have been able to measure the odds for determining qualifications. This model is nowhere near good enough to be published or presented in a conference. This was a great learning experience for careers even though the model we created is still inferior to a linear regression pricing model since we believe that money is the most important to realtors and people when it comes to the housing market. If you wish to know more about the data and analysis we have completed visit reference link 1. As a famous person once said, “failure is the mother of success.”
"""

"""
I worked with Kingsley Iyawe in STAT-616 Generalized Linear Models to complete this project and report. He deserves some credit for this report.
"""

# ============================================================
# SECTION 9 — REFERENCES
# ============================================================

"""
1. https://aaronniecestro.shinyapps.io/DC-Housing/ 
2. McKay, Allie W. “Farmers' Markets vs. Food Deserts: Which Are Winning in DC?” The Capital's Markets, 31 July 2014, thecapitalsmarkets.wordpress.com/2014/07/31/farmers-markets-vs-food-deserts-which-is-winning-in-dc/.
3. Johnson, Matt. “Washington's Systemic Streets.” Greater Greater Washington, ggwash.org/view/2530/washingtons-systemic-streets.
4. “Money Is The Root Of All Evil Stock Photos and Images.” Alamy, www.alamy.com/stock-photo/money-is-the-root-of-all-evil.html.
5. “Types of Housing Models and Programs.” The 519, www.the519.org/education-training/lgbtq2s-youth-homelessness-in-canada/types-of-housing-models-and-programs.
6. Dobbins, Tim, and John Burke. “Predicting Housing Prices with Linear Regression Using Python, Pandas, and Statsmodels.” Learn Data Science - Tutorials, Books, Courses, and More, www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/.
7. Corsini, Kenneth Richard. “STATISTICAL ANALYSIS OF RESIDENTIAL HOUSING PRICES IN AN UP AND DOWN REAL ESTATE MARKET: A GENERAL FRAMEWORK AND STUDY OF COBB COUNTY, GA .” A Thesis Presented to The Academic Faculty, Georgia Institute of Technology, Dec. 2009, smartech.gatech.edu/bitstream/handle/1853/31763/Corsini_Kenneth_R_200912_mast.pdf.
8. "Regression Data for Inclusionary Housing Simulation Model | DataSF | City, and County of San Francisco." San Francisco Data, data.sfgov.org/Economy-and-Community/Regression-data-for-Inclusionary-Housing-Simulatio/vcwn-f2xk/data.
9. Leonard, Kimberlee. “What Forms Are Needed to Sell a Home by Owner?” Home Guides | SF Gate, 29 Dec. 2018, homeguides.sfgate.com/forms-needed-sell-home-owner-7271.html.
10. Leonard, Kimberlee. “What Is the Procedure for Closing a for Sale by Owner House Sale?” Home Guides | SF Gate, 15 Dec. 2018, homeguides.sfgate.com/procedure-closing-sale-owner-house-sale-65511.html.
"""

# ============================================================
# END OF SCRIPT
# ============================================================
