import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Ipl_Dataset_2022.csv'  # Update this path if needed
data = pd.read_csv(file_path)

# Drop unnecessary columns and rows with missing target values
data = data.drop(columns=['Unnamed: 0', 'COST IN ₹ (CR.', 'Cost IN $ (000)', '2021 Squad'])
data = data.dropna(subset=['COST IN ₹ (CR.)'])

# Encode categorical features
label_encoder = LabelEncoder()
for column in ['Player', 'Base Price', 'TYPE', 'Team']:
    data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.drop(columns=['COST IN ₹ (CR.)'])
y = data['COST IN ₹ (CR.)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))

# KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
y_pred_knn = knn_reg.predict(X_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))

# Logistic Regression (classifying if cost > 10 crores)
y_binary = (y > 10).astype(int)
log_reg = LogisticRegression().fit(X_train, y_binary.loc[X_train.index])
y_pred_log = log_reg.predict(X_test)
log_accuracy = accuracy_score(y_binary.loc[X_test.index], y_pred_log)

# Output results
print(f"Linear Regression RMSE: {lin_rmse:.2f}")
print(f"KNN Regression RMSE: {knn_rmse:.2f}")
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

# Scatter plots
plt.figure(figsize=(15, 5))

# Linear Regression Scatter Plot
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred_lin)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Linear Regression Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# KNN Regression Scatter Plot
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test, y=y_pred_knn)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('KNN Regression Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Logistic Regression Scatter Plot
plt.subplot(1, 3, 3)
sns.scatterplot(x=y_binary.loc[X_test.index], y=y_pred_log)
plt.title('Logistic Regression Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
