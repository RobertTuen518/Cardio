"""This is the main module to run the app"""

# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from random_forest import load_data
#from web_functions import load_data

# Import predict page
import predict

# Configure the app
st.set_page_config(
    page_title = 'Disease Risk Predictor',
    page_icon = 'random',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)

# Dictionary for pages
pages = {
    "Prediction": predict,
}

# Create a sidebar
# Add title to sidear
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(pages.keys()))

# Loading the dataset.
df, X, y = load_data()

# Call the app funciton of selected page to run
if page in ["Prediction"]:
    pages[page].app(df, X, y)
else:
    pages[page].app()
