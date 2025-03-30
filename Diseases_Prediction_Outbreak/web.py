import os
import pickle  # pre-trained model loading
import streamlit as st  # web app framework

# Set page configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks', layout='wide', page_icon="🧑‍⚕️")

# Hide Streamlit header and footer for a cleaner look
hide_streamlit_style = """
            <style>
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-container {
        background-color: #F5F5F5;
        padding: 25px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Construct dynamic paths for the models based on the current file's directory
current_dir = os.path.dirname(__file__)
diabetes_model_path = os.path.join(current_dir, "training_models", "diabetes_model.pkl")
heart_model_path = os.path.join(current_dir, "training_models", "heart_model.pkl")
parkinson_model_path = os.path.join(current_dir, "training_models", "parkinson.pkl")

# Load the pre-trained models
with open(diabetes_model_path, "rb") as f:
    diabetes_model = pickle.load(f)
with open(heart_model_path, "rb") as f:
    heart_disease_model = pickle.load(f)
with open(parkinson_model_path, "rb") as f:
    parkinsons_model = pickle.load(f)

# Sidebar selection
with st.sidebar:
    st.markdown("<h2> Disease Prediction System </h2>", unsafe_allow_html=True)
    selected = st.radio("Select Prediction Type", 
                         ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'])
    st.markdown("---")
    st.write("This tool uses pre-trained ML models to predict disease risk.")

# Begin main container for the app content
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction using ML')
    st.info("Please fill in the details below and click on Submit.")

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', placeholder="e.g., 2")
        with col2:
            Glucose = st.text_input('Glucose level', placeholder="e.g., 120")
        with col3:
            Bloodpressure = st.text_input('Blood Pressure value', placeholder="e.g., 70")
        with col1:
            SkinThickness = st.text_input('Skin Thickness', placeholder="e.g., 20")
        with col2:
            Insulin = st.text_input('Insulin level', placeholder="e.g., 80")
        with col3:
            BMI = st.text_input('BMI value', placeholder="e.g., 30")
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder="e.g., 0.5")
        with col2:
            Age = st.text_input('Age of the person', placeholder="e.g., 35")
        submitted_diabetes = st.form_submit_button("Submit")

    if submitted_diabetes:
        try:
            user_input = [Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')
        except Exception as e:
            st.error("Error: Please ensure all inputs are valid numeric values.")

elif selected == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction using ML')
    st.info("Enter the following details and click Submit:")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.text_input('Age', placeholder="e.g., 54")
        with col2:
            Sex = st.text_input('Gender (0=male, 1=female)', placeholder="0 or 1")
        with col3:
            cp = st.text_input('Chest Pain Type (cp)', placeholder="e.g., 3")
        with col1:
            trestbps = st.text_input('Resting BP (trestbps)', placeholder="e.g., 140")
        with col2:
            chol = st.text_input('Cholesterol (chol)', placeholder="e.g., 250")
        with col3:
            fbs = st.text_input('Fasting Blood Sugar (fbs)', placeholder="0 or 1")
        with col1:
            restecg = st.text_input('Resting ECG (restecg)', placeholder="e.g., 0")
        with col2:
            thalach = st.text_input('Max Heart Rate (thalach)', placeholder="e.g., 150")
        with col3:
            exang = st.text_input('Exercise Induced Angina (exang)', placeholder="0 or 1")
        with col1:
            oldpeak = st.text_input('Oldpeak', placeholder="e.g., 1.0")
        with col2:
            slop = st.text_input('Slope', placeholder="e.g., 2")
        with col3:
            ca = st.text_input('No. of Major Vessels (ca)', placeholder="e.g., 1")
        with col1:
            thal = st.text_input('Thal', placeholder="e.g., 3")
        submitted_heart = st.form_submit_button("Submit")

    if submitted_heart:
        try:
            user_input = [Age, Sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slop, ca, thal]
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                st.success('The person has Heart Disease')
            else:
                st.success('The person does not have any Heart Disease')
        except Exception as e:
            st.error("Error: Please ensure all inputs are valid numeric values.")

elif selected == 'Parkinsons Prediction':
    st.header('Parkinsons Prediction using ML')
    st.info("Fill out the measures below and click Submit:")

    with st.form("parkinsons_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Fo = st.text_input('MDVP:fo(Hz)', placeholder="e.g., 120")
        with col2:
            Fhi = st.text_input('MDVP:fhi(Hz)', placeholder="e.g., 150")
        with col3:
            Flo = st.text_input('MDVP:flo(Hz)', placeholder="e.g., 110")
        with col1:
            Jitter = st.text_input('MDVP:jit(%)', placeholder="e.g., 0.005")
        with col2:
            Jitterabs = st.text_input('MDVP:jitter(Abs)', placeholder="e.g., 0.00005")
        with col3:
            Rap = st.text_input('MDVP:rap', placeholder="e.g., 0.003")
        with col1:
            Ppq = st.text_input('MDVP:PPQ', placeholder="e.g., 0.005")
        with col2:
            DDp = st.text_input('Jitter:DDP', placeholder="e.g., 0.010")
        with col3:
            Shimmer = st.text_input('MDVP:shimmer', placeholder="e.g., 0.03")
        with col1:
            Shimmers = st.text_input('MDVP:shimmer(db)', placeholder="e.g., 0.23")
        with col2:
            APQ3 = st.text_input('MDVP:APQ3', placeholder="e.g., 0.05")
        with col3:
            APQ5 = st.text_input('MDVP:APQ5', placeholder="e.g., 0.05")
        with col1:
            APQ = st.text_input('MDVP:APQ', placeholder="e.g., 0.07")
        with col2:
            DDA = st.text_input('Shimmer:DDA', placeholder="e.g., 0.02")
        with col3:
            NHR = st.text_input('NHR', placeholder="e.g., 0.04")
        with col1:
            HNR = st.text_input('HNR', placeholder="e.g., 22")
        with col2:
            stat = st.text_input('Status', placeholder="e.g., 1")
        with col3:
            RPDE = st.text_input('RPDE', placeholder="e.g., 0.5")
        with col1:
            DFA = st.text_input('DFA', placeholder="e.g., 0.9")
        with col2:
            spread1 = st.text_input('Spread1', placeholder="e.g., 0.1")
        with col3:
            spread2 = st.text_input('Spread2', placeholder="e.g., 0.1")
        with col1:
            D2 = st.text_input('D2', placeholder="e.g., 2.0")
        submitted_parkinson = st.form_submit_button("Submit")

    if submitted_parkinson:
        try:
            user_input = [Fo, Fhi, Flo, Jitter, Jitterabs, Rap, Ppq, DDp, Shimmer, Shimmers, APQ3,
                          APQ5, APQ, DDA, NHR, HNR, stat, RPDE, DFA, spread1, spread2, D2]
            user_input = [float(x) for x in user_input]
            park_prediction = parkinsons_model.predict([user_input])
            if park_prediction[0] == 1:
                st.success('The person is having Parkinsons')
            else:
                st.success('The person is not having Parkinsons')
        except Exception as e:
            st.error("Error: Please ensure all inputs are valid numeric values.")

st.markdown("</div>", unsafe_allow_html=True)
