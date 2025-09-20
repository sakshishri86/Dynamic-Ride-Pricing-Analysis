import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page config
st.set_page_config(page_title="Dynamic Pricing EDA", layout="wide")

st.title(" Dynamic Pricing Strategy Explorer")
st.markdown("Visual analysis and dynamic pricing implementation based on ride data.")

# Step 2: Load and Preprocess Data
st.header(" Data Loading and Preprocessing")

full_data = pd.read_csv("dynamic_pricing.csv")

columns_to_use = [
    'Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration',
    'Historical_Cost_of_Ride', 'Vehicle_Type'
]

data = full_data[columns_to_use].copy()
data.columns = [
    'Number of Riders', 'Number of Drivers', 'Expected Ride Duration',
    'Historical Cost of Ride', 'Vehicle Type'
]

st.write("### Preview of Data")
st.dataframe(data.head())

# Step 3: EDA
st.header("Exploratory Data Analysis")

# --- Graph 1: Correlation Heatmap ---
st.subheader("Correlation Matrix")
fig1, ax1 = plt.subplots(figsize=(2.8, 2))
correlation_matrix = data.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1, cbar=False,
            annot_kws={"size": 6})
ax1.tick_params(labelsize=6)
st.pyplot(fig1)
st.caption("Strong correlation between ride duration and cost.")

# --- Graph 2: Distribution of Expected Ride Duration ---
st.subheader("Expected Ride Duration Distribution")
fig2, ax2 = plt.subplots(figsize=(3, 2.2))
sns.histplot(data['Expected Ride Duration'], kde=True, bins=30, color='skyblue', ax=ax2)
ax2.set_xlabel('', fontsize=6)
ax2.set_ylabel('', fontsize=6)
ax2.tick_params(labelsize=6)
st.pyplot(fig2)

# --- Graph 3: Ride Duration vs Historical Cost ---
st.subheader("Ride Duration vs Historical Cost")
fig3, ax3 = plt.subplots(figsize=(3, 2.2))
sns.scatterplot(x='Expected Ride Duration', y='Historical Cost of Ride',
                data=data, alpha=0.5, ax=ax3, s=8)
ax3.set_xlabel('', fontsize=6)
ax3.set_ylabel('', fontsize=6)
ax3.tick_params(labelsize=6)
st.pyplot(fig3)

# --- Graph 4: Historical Cost Distribution ---
st.subheader("Historical Cost of Ride Distribution")
fig4, ax4 = plt.subplots(figsize=(3, 2.2))
sns.histplot(data['Historical Cost of Ride'], kde=True, bins=30, color='salmon', ax=ax4)
ax4.set_xlabel('', fontsize=6)
ax4.set_ylabel('', fontsize=6)
ax4.tick_params(labelsize=6)
st.pyplot(fig4)

# Step 4: Apply Pricing Strategy
st.header("Dynamic Pricing Strategy")

data['demand_supply_ratio'] = data['Number of Riders'] / data['Number of Drivers']
high_demand_threshold = data['demand_supply_ratio'].quantile(0.75)
low_demand_threshold = data['demand_supply_ratio'].quantile(0.25)

data['New Fare'] = data['Historical Cost of Ride']
data.loc[data['demand_supply_ratio'] > high_demand_threshold, 'New Fare'] *= 1.2
data.loc[data['demand_supply_ratio'] < low_demand_threshold, 'New Fare'] *= 0.8
data['New Fare'] = data['New Fare'].round(2)

st.write("### Dynamic Pricing Applied ")
st.dataframe(data[['Expected Ride Duration', 'Historical Cost of Ride', 'demand_supply_ratio', 'New Fare']].head())
# Regression Analysis
st.header("Regression Analysis for Fare Prediction")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Prepare data for regression
# Drop 'Vehicle Type' for now or one-hot encode it if needed. For simplicity, let's focus on numerical features first.
# Also, we will use 'Historical Cost of Ride' as the target variable to see how well we can predict it
# or 'New Fare' if we want to predict the dynamically adjusted fare. Let's predict 'New Fare'.
regression_data = data[[
    'Number of Riders', 'Number of Drivers', 'Expected Ride Duration',
    'Historical Cost of Ride', 'demand_supply_ratio', 'New Fare'
]].copy()

# Define features (X) and target (y)
X = regression_data[[
    'Number of Riders', 'Number of Drivers', 'Expected Ride Duration',
    'Historical Cost of Ride', 'demand_supply_ratio'
]]
y = regression_data['New Fare']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Linear Regression Model")

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)

st.write(f"**R-squared (Linear Regression):** {r2_linear:.4f}")
st.write(f"**Mean Squared Error (Linear Regression):** {mse_linear:.4f}")
st.write(f"**Root Mean Squared Error (Linear Regression):** {rmse_linear:.4f}")

st.markdown("""
A higher R-squared value indicates that the model explains a larger proportion of the variance in the target variable.
""")

st.subheader("Random Forest Regressor Model")

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest Regressor
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

st.write(f"**R-squared (Random Forest):** {r2_rf:.4f}")
st.write(f"**Mean Squared Error (Random Forest):** {mse_rf:.4f}")
st.write(f"**Root Mean Squared Error (Random Forest):** {rmse_rf:.4f}")

st.markdown("""
Random Forest often performs better than Linear Regression due to its ability to capture non-linear relationships and interactions between features.
""")

# Feature Importance (for Random Forest)
st.subheader("Feature Importance from Random Forest")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig7, ax7 = plt.subplots(figsize=(4, 2.5))
sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax7, palette='viridis')
ax7.set_xlabel("Importance", fontsize=7)
ax7.set_ylabel("Feature", fontsize=7)
ax7.tick_params(labelsize=7)
ax7.set_title("Feature Importance in Predicting New Fare", fontsize=8)
st.pyplot(fig7)
st.caption("This plot shows which features were most influential in the Random Forest model's prediction of the new fare.")



# Step 5: Strategy Impact Visualization
st.header("📈 Strategy Impact Visualization")

# --- Graph 5: Old Fare vs New Fare ---
st.subheader("Old vs New Fare Distribution")
fig5, ax5 = plt.subplots(figsize=(3.2, 2.2))
sns.histplot(data['Historical Cost of Ride'], color="red", label="Old Fare", kde=True, alpha=0.4, ax=ax5)
sns.histplot(data['New Fare'], color="green", label="New Fare", kde=True, alpha=0.4, ax=ax5)
ax5.legend(fontsize=6)
ax5.set_xlabel('', fontsize=6)
ax5.set_ylabel('', fontsize=6)
ax5.tick_params(labelsize=6)
st.pyplot(fig5)

# --- Graph 6: New Fare by Vehicle Type ---
st.subheader("New Fare Share by Vehicle Type")
avg_fare_by_vehicle = data.groupby('Vehicle Type')['New Fare'].mean()
fig6, ax6 = plt.subplots(figsize=(2.5, 2.5))
ax6.pie(avg_fare_by_vehicle, labels=avg_fare_by_vehicle.index, autopct='%1.1f%%',
        startangle=90, colors=['lightcoral', 'lightskyblue'],
        textprops={'fontsize': 6})
st.pyplot(fig6)

# Step 6: Final Conclusion
st.header(" Final Project Conclusion")
st.markdown("""
- **Old Pricing Flaw:** Ride cost was almost entirely duration-based.
- **Dynamic Strategy:** Adjusted fare based on demand/supply ratio using quantiles.
- **Outcome:** Introduced price diversity — higher in high demand, lower in low demand.
- **Business Impact:** Boosted potential profitability and user satisfaction.
""")
