# Simple Linear Regression

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm

# # 1. Load your time series data
# # Each column is a sector indicator or GNI per capita
# df = pd.read_csv("kenya_sector_gni_data.csv")

# df = df.dropna()
# df['log_gni'] = np.log(df['gni pc ppp'])

# # Define independent variables
# X = df[['healthcare', 'agriculture', 'infrastructure', 'technology', 'transport', 'education']]
# X = sm.add_constant(X)  # Adds intercept

# # Define dependent variable
# y = df['log_gni']

# print(y)

# # Run the regression
# model = sm.OLS(y, X).fit()

# # Print the regression summary
# print(model.summary())

# elasticities = model.params.drop("const")
# print("Sector Elasticities (E_i):\n", elasticities)

# for i in elasticities:
#     print(i)

# Ridge regression

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("kenya_sector_gni_data.csv")
# Drop missing values
df = df.dropna()

# Take log of GNI per capita PPP
df['log_gni'] = np.log(df['gni pc ppp'])

# Define X and y
X = df[['healthcare', 'agriculture', 'infrastructure', 'technology', 'transport', 'education']]
y = df['log_gni']

# Standardise features and apply Ridge regression with cross-validation
alphas = np.logspace(-6, 3, 100)  # search over a wide range of regularization strengths

ridge_model = make_pipeline(
    StandardScaler(),  # standardises X for regularisation to work well
    RidgeCV(alphas=alphas, cv=5)  # 5-fold cross-validation
)

ridge_model.fit(X, y)

# Extract the trained model
ridge = ridge_model.named_steps['ridgecv']

# Print alpha selected
print(f"Selected alpha (regularisation strength): {ridge.alpha_}")

# Get coefficients (elasticities)
coefficients = ridge.coef_
sectors = X.columns
elasticities = pd.Series(coefficients, index=sectors)

print("Ridge-Regularised Sector Elasticities (E_i):\n", elasticities)

# Print as percentages
print("\nElasticities as % impact per unit increase in sector index:")
for i in elasticities:
    print(f"{i * 100:.3f}%")

