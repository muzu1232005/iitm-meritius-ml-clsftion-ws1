# Fraudulent Credit Card Transaction Detection Project

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
data = pd.read_csv('fraud_txn.csv')

# Data Cleaning and Preprocessing
# Drop unnecessary columns
data.drop(columns=['Unnamed: 0','trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num'], inplace=True)

# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)

# Fraud Detection Model Implementation

## Data Preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Encode categorical variables
categorical_cols = ['merchant', 'category', 'gender']
numeric_cols = ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', scaler, numeric_cols),
    ('cat', ohe, categorical_cols)
])

# Feature Engineering
data['transaction_hour'] = pd.to_datetime(data['unix_time'], unit='s').dt.hour

# Define target and features
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

## Polynomial Features

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

## Dimensionality Reduction


from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_poly)
X_test_pca = pca.transform(X_test_poly)

## Model Training and Evaluation


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    y_pred_train = model.predict(X_train_pca)
    print(f'--- {name} ---')
    print("############## Classification Report: Train ###################")
    print('Accuracy:', accuracy_score(y_train, y_pred_train))
    print('Classification Report:\n', classification_report(y_train, y_pred_train))
    print('Confusion Matrix:\n', confusion_matrix(y_train, y_pred_train))
    print("############## Classification Report: Train ###################")
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('\n')

