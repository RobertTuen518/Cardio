# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    # Load the pre-trained model
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    # Get the model score
    score = model.score(X_train, y_train)
    print(score)

    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)

    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score