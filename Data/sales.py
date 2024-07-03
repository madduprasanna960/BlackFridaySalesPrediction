import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and Inspect Data
data = pd.read_csv('BlackFridaySales.csv')
print(data.head())
print(data.info())

# Step 2: Preprocess Data
# Fill missing values
data.fillna(-999, inplace=True)

# Encode categorical features
label_encoders = {}
categorical_columns = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['Purchase']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Exploratory Data Analysis
# Plot distribution of Purchase
sns.histplot(data['Purchase'], bins=30, kde=True)
plt.title('Distribution of Purchase Amount')
plt.show()

# Check the data types to ensure correct selection of numeric data
print(data.dtypes)

# Select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
print(numeric_data.head())

# Plot correlation heatmap
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Step 4: Feature Engineering
# Create interaction features (example)
data['Age_Gender_Interaction'] = data['Age'] * data['Gender']

# Step 5: Model Selection and Training
# Define features and target
X = data.drop(['User_ID', 'Product_ID', 'Purchase'], axis=1)
y = data['Purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Conclusion
print("The model achieved an R-squared value of {:.2f}, indicating that {:.2f}% of the variance in the purchase amounts can be explained by the model.".format(r2, r2*100))
