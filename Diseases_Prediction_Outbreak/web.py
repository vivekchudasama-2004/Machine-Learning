import os
import pickle  # pre trained model loading
import streamlit as st  # web app
from streamlit import status
# from streamlit_option_menu import option_menu

st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon="🧑‍⚕️")

diabetes_model = pickle.load(open(r".\training_models\diabetes_model.pkl", 'rb'))
heart_disease_model = pickle.load(open(r"C:\Users\Vivek\PycharmProjects\Diseases_Prediction_Outbreak\training_models\heart_model.pkl", 'rb'))
parkinsons_model = pickle.load(open(r"C:\Users\Vivek\PycharmProjects\Diseases_Prediction_Outbreak\training_models\parkinson.pkl", 'rb'))


with st.sidebar :
    selected = option_menu('Prediction of disease outbreak system',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)


if selected == 'Diabetes Prediction' :
    st.title('Diabetes Prediction using Ml')
    col1, col2, col3 = st.columns(3)
    with col1 :
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2 :
        Glucose = st.text_input('Glucose level')
    with col3 :
        Bloodpressure = st.text_input('Blood Pressure value')
    with col1 :
        SkinThickness = st.text_input('Skin Thickness value')
    with col2 :
        Insulin = st.text_input('Insulin level')
    with col3 :
        BMI = st.text_input('BMI  value')
    with col1 :
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2 :
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result') :
        user_input = [Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else :
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)

#heart

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Ml')
    col1, col2, col3 = st.columns(3)
    with col1 :
        Age= st.text_input('Age')
    with col2 :
        Sex = st.text_input('Gender(0=male, 1=female)')
    with col3 :
        cp = st.text_input('cp')
    with col1 :
        trestbps = st.text_input('Trestbps')
    with col2 :
        chol = st.text_input('chol')
    with col3 :
        fbs = st.text_input('fbs')
    with col1 :
        restecg = st.text_input('restecg')
    with col2 :
        thalach = st.text_input('thalach')
    with col3 :
        exang = st.text_input('exang')
    with col1 :
        oldpeak = st.text_input('oldpeak')
    with col2 :
        slop = st.text_input('slop')
    with col3 :
        ca = st.text_input('ca')
    with col1 :
        thal =st.text_input('thal')


    heart_diagnosis = ''
    if st.button('Heart disease Test Result') :
        user_input = [Age,Sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slop,ca,thal]
        user_input = [float(x) for x in user_input]
        hert_prediction = heart_disease_model.predict([user_input])
        if hert_prediction[0] == 1 :
            heart_diagnosis = 'The person have Heart Disease '
        else :
            heart_diagnosis = 'The person is dose not have any Heart Disease'
    st.success(heart_diagnosis)

#parkison

if selected == 'Parkinsons prediction':
    st.title('Parkinsons prediction using Ml')
    col1, col2, col3 = st.columns(3)
    with col1 :
        Fo = st.text_input('MDVP:fo(Hz)')
    with col2 :
        Fhi = st.text_input('MDVP:fhi(Hz)')
    with col3 :
        Flo = st.text_input('MDVP:flo(Hz)')
    with col1 :
        Jitter = st.text_input('MDVP:jit(%)')
    with col2 :
        Jitterabs= st.text_input('MDVP:jitter(Abs)')
    with col3 :
        Rap = st.text_input('MDVP:rap')
    with col1 :
        Ppq = st.text_input('MDVP:PPQ')
    with col2 :
        DDp = st.text_input('Jitter:DDP')
    with col3 :
        Shimmer = st.text_input('MDVP:shimmer')
    with col1 :
        Shimmers = st.text_input('MDVP:shimmer(db)')
    with col2 :
        APQ3 = st.text_input('MDVP:APQ3')
    with col3 :
        APQ5 = st.text_input('MDVP:APQ5')
    with col1 :
        APQ = st.text_input('MDVP:APQ')
    with col2 :
        DDA = st.text_input('Shimmer:DDA')
    with col3 :
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2 :
        status = st.text_input('Status')
    with col3:
        RPDE = st.text_input('RPDE')
    with col1:
        DFA = st.text_input('DFA')
    with col2 :
        spread1 = st.text_input('Spread1')
    with col3 :
        spread2 = st.text_input('Spread2')
    with col1 :
        D2 = st.text_input('D2')


    park_diagnosis = ''
    if st.button('Parkinsons Test Result') :
        user_input = [Fo,Fhi,Flo,Jitter,Jitterabs,Rap,Ppq,DDp,Shimmer,Shimmers,APQ3,
                      APQ5,APQ,DDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2]
        user_input = [float(x) for x in user_input]
        park_prediction = parkinsons_model.predict([user_input])
        if park_prediction[0] == 1 :
            park_diagnosis = 'The person is having parkinsons'
        else :
            park_diagnosis = 'The person is not having parkinsons'
    st.success(park_diagnosis)
