import pandas as pd
import statsmodels.api as sm

# Step 1: Load your data
df = pd.read_csv("gov_sensitivity.csv")  # Replace with your actual file name

# Step 2: Calculate year-on-year change in governance score
df['GOV_CHANGE'] = df['NORM'].diff()  # Δγ(t) = γ(t) - γ(t-1)

# Step 3: Shift AID EFFECTIVE so it's aligned with the governance change in the next year
# (i.e. aid at time t explains change in governance at t+1)
df['AID_EFFECTIVE_LAG'] = df['AID EFFECTIVE'].shift()

# Step 4: Drop NA rows (first year will have missing values due to diff and shift)
df = df.dropna()

# Step 5: Run OLS regression
X = sm.add_constant(df['AID_EFFECTIVE_LAG'])  # Predictor (with constant)
y = df['GOV_CHANGE']                          # Target variable

print(df)
model = sm.OLS(y, X).fit()

# Step 6: Output results
print(model.summary())
