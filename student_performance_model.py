import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a synthetic dataset
data = {
    'Attendance': np.random.randint(50, 101, 200),
    'Study_Hours': np.random.uniform(1, 10, 200),
    'Parental_Education': np.random.randint(1, 4, 200),  # 1: High School, 2: Bachelor, 3: Master
    'Economic_Status': np.random.randint(1, 4, 200),  # 1: Low, 2: Medium, 3: High
    'Extra_Curricular': np.random.randint(0, 2, 200),  # 0: No, 1: Yes
    'Final_Grade': np.random.uniform(50, 100, 200)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Head:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# Pairplot to visualize relationships
sns.pairplot(df, diag_kind='kde')
plt.show()

# Step 3: Data Preprocessing
X = df.drop('Final_Grade', axis=1)
y = df['Final_Grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([50, 100], [50, 100], color='red', linestyle='--')
plt.title('Actual vs Predicted Final Grades')
plt.xlabel('Actual Final Grades')
plt.ylabel('Predicted Final Grades')
plt.show()

# Step 6: Feature Importance
importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names)
plt.title('Feature Importance')
plt.show()

# Save the model
import joblib
joblib.dump(model, 'student_performance_model.pkl')
