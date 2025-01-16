# Import essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data from local CSV file
titanic_data = pd.read_csv(r"C:\Users\ishar\Downloads\Titanic-Dataset.csv")  # Adjust to your file location
print("Dataset successfully loaded!")

# Display missing values per column
print("Missing values summary:\n", titanic_data.isnull().sum())

# Fill missing data with sensible defaults
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)
titanic_data['Embarked'].fillna('S', inplace=True)  # Using 'S' as the most common embarkation point

# Encode categorical values for machine learning
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
titanic_data['Embarked'] = titanic_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# Add new features to enrich the model
titanic_data['FamilyCount'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['FareGroup'] = pd.qcut(titanic_data['Fare'], 4, labels=[1, 2, 3, 4])
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=[0, 16, 32, 48, 64, 80], labels=[1, 2, 3, 4, 5])

# Convert newly created features to integers
titanic_data['FareGroup'] = titanic_data['FareGroup'].astype(int)
titanic_data['AgeGroup'] = titanic_data['AgeGroup'].astype(int)

# Drop columns that are not relevant for the prediction
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Age', 'SibSp', 'Parch', 'Cabin'], axis=1, inplace=True)

# Check final data types to ensure compatibility
print("Data types after cleaning:\n", titanic_data.dtypes)

# Separate features from the target variable
features = titanic_data.drop('Survived', axis=1)
target = titanic_data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Balance the data using SMOTE for better model performance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Applied SMOTE to handle class imbalance.")

# Set up parameter grid for tuning the RandomForestClassifier
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Use GridSearchCV to find the best combination of parameters
search = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, scoring='accuracy')
search.fit(X_train_balanced, y_train_balanced)
optimal_model = search.best_estimator_

print("Best parameters found:", search.best_params_)

# Test the model on the unseen test set
y_predicted = optimal_model.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, y_predicted))
print("Classification Report:\n", classification_report(y_test, y_predicted))

# Visualize feature importance
importance_values = optimal_model.feature_importances_
feature_names = X_train.columns
sorted_indices = np.argsort(importance_values)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importance_values[sorted_indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[sorted_indices], rotation=90)
plt.show()
