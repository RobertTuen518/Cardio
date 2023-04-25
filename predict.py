# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from random_forest import predict
#from web_functions import predict

def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Disease Risk Prediction")

    # Add a brief description
    st.markdown(
        """
            <p style="color:#FF4B4B; font-size:25px">
                This app uses <b>Random Forest Classifier</b> for the Cardiovascular Disease Risk Prediction.
            </p>
        """, unsafe_allow_html=True)

    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    age = st.number_input('Age', min_value=0, max_value=100, value=0)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    height = st.number_input('Height (cm)', min_value=0, max_value=300, value=0)
    weight = st.number_input('Weight (kg)', min_value=0, max_value=500, value=0)
    systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=0, max_value=300, value=0)
    diastolic_bp = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=0, max_value=300, value=0)
    cholesterol = st.selectbox('Cholesterol Level', options=['Normal', 'Above Normal', 'Well Above Normal'])
    glucose = st.selectbox('Glucose Level', options=['Normal', 'Above Normal', 'Well Above Normal'])
    smoke = st.selectbox('Do you smoke?', options=['Yes', 'No'])
    alcohol = st.radio("Do you like to drink alcohol?", ("Yes", "No"))
    active = st.radio("Are you a active person?", ("Yes", "No"))
    '''
    # Create a dictionary to map categorical variables
    gender = {'Male': 1, 'Female': 2}
    cholesterol = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    glucose = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    smoke = {'Yes': 1, 'No': 0}
    alcohol = {'Yes': 1, 'No': 0}
    active = {'Yes': 1, 'No': 0}
    
    # Create a list to store all the features
    features = [age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoke, alcohol, active]
    '''
    # Create dictionaries to map categorical variables
    gender_dict = {'Male': 1, 'Female': 2}
    cholesterol_dict = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    glucose_dict = {'Normal': 1, 'Above Normal': 2, 'Well Above Normal': 3}
    smoke_dict = {'Yes': 1, 'No': 0}
    alcohol_dict = {'Yes': 1, 'No': 0}
    active_dict = {'Yes': 1, 'No': 0}

    # Map categorical variables to numerical values
    gender_num = gender_dict[gender]
    cholesterol_num = cholesterol_dict[cholesterol]
    glucose_num = glucose_dict[glucose]
    smoke_num = smoke_dict[smoke]
    alcohol_num = alcohol_dict[alcohol]
    active_num = active_dict[active]

    # Create a list to store all the features
    features = [age, gender_num, height, weight, systolic_bp, diastolic_bp, cholesterol_num, glucose_num, smoke_num,
                alcohol_num, active_num]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)

        st.success("Predicted Sucessfully")

        # Print the output according to the prediction
        if prediction == 1:
            st.info('You are at high risk of developing a cardiovascular disease.')
        else:
            st.info('You are at low risk of developing a cardiovascular disease.')

        # Print the score of the model
        st.write("The accuracy score of this model is", score)
