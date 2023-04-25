# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the dataset into DataFrame.
    df = pd.read_csv('C:/Users/User/PycharmProjects/Python_Tutorial/cardio.csv')

    # Drop the id column
    df.drop('id', axis=1, inplace=True)

    df['age'] = np.floor_divide(df['age'], 365)

    # Split the dataset into features and target
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    return df, X, y


@st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    '''
    # Scale the input data to non-negative values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform feature selection
    skb = SelectKBest(chi2, k=11)
    X_new = skb.fit_transform(X_scaled, y)

    # Get the selected feature names
    selected_features = X.columns[skb.get_support()]

    # Print the selected feature names
    print(selected_features)

    # Create the model
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the parameter grids for each algorithm
    lr_params = {'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 20)}
    dt_params = {'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8]}
    rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 4, 6, 8]}

    # Define the classifiers
    rf = RandomForestClassifier(random_state=42)

    # Define the GridSearchCV objects
    rf_cv = GridSearchCV(rf, rf_params, scoring='accuracy', cv=5, n_jobs=-1)

    # Fit the GridSearchCV objects
    rf_cv.fit(X_train, y_train)

    print("Random Forest:")
    print("Best hyperparameters: ", rf_cv.best_params_)
    print("Training score: ", rf_cv.best_score_)
    print("Test score: ", rf_cv.score(X_test, y_test))
    print()

    # Define the classifiers with the best hyperparameters
    rf_best = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=200, random_state=42)

    # Fit the classifiers
    rf_best.fit(X_train, y_train)

    # Make predictions
    rf_pred = rf_best.predict(X_test)

    print("Random Forest:")
    print(classification_report(y_test, rf_pred))
    print()

    # Get the model score
    score = rf_best.score(X_train, y_train)

    print(score)
    
    # Return the values
    return rf_best, score
    '''
    '''
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    '''
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=200, random_state=42)

    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)
    #score = accuracy_score(y_test, y_pred)

    # Return the values
    return model, score


def predict(X, y, features):
    # Get model and model score
    rf_best, score = train_model(X, y)

    # Predict the value
    prediction = rf_best.predict(np.array(features).reshape(1, -1))

    return prediction, score
