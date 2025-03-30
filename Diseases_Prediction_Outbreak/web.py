import os
import pickle  # pre-trained model loading
import streamlit as st  # web app framework

# Set page configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

# --- Custom CSS to remove the white div above titles ---
st.markdown(
    """
    <style>
    /* Remove the default padding/margin above titles */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0;
        padding-top: 0;
    }
    /* Optionally, tighten spacing for other elements */
    .main-container {
        padding-top: 0px;  /* Reduce top padding */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Section ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">Select Prediction Type</div>', unsafe_allow_html=True)
    
    selected = st.radio("", 
                        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'], 
                        index=0)
    
    st.markdown('<div class="sidebar-content">This tool uses pre-trained ML models to predict disease risk.</div>', unsafe_allow_html=True)

# --- Main Content Section ---
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Paths for your models
current_dir = os.path.dirname(__file__)
diabetes_model_path = os.path.join(current_dir, "training_models", "diabetes_model.pkl")
heart_model_path = os.path.join(current_dir, "training_models", "heart_model.pkl")
parkinson_model_path = os.path.join(current_dir, "training_models", "parkinson.pkl")

# Load models
with open(diabetes_model_path, "rb") as f:
    diabetes_model = pickle.load(f)
with open(heart_model_path, "rb") as f:
    heart_disease_model = pickle.load(f)
with open(parkinson_model_path, "rb") as f:
    parkinsons_model = pickle.load(f)

# --- Prediction Type Logic ---
if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction using ML')
    # Form layout for diabetes input
    # (Rest of your code...)

elif selected == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction using ML')
    # Form layout for heart disease input
    # (Rest of your code...)

elif selected == 'Parkinsons Prediction':
    st.header('Parkinsons Prediction using ML')
    st.info("Fill out the measures below and click Submit:")

    # Form layout for Parkinsons input
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
            user_input = [Fo, Fhi, Flo, Jitter, Jitterabs, Rap, Ppq, DDp, Shimmer, Shimmers,
                          APQ3, APQ5, APQ, DDA, NHR, HNR, stat, RPDE, DFA, spread1, spread2, D2]
            user_input = [float(x) for x in user_input]
            park_prediction = parkinsons_model.predict([user_input])
            if park_prediction[0] == 1:
                st.success('The person is having Parkinsons')
            else:
                st.success('The person is not having Parkinsons')
        except Exception as e:
            st.error("Error: Please ensure all inputs are valid numeric values.")

st.markdown("</div>", unsafe_allow_html=True)
