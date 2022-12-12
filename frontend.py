import streamlit as st
import requests

# Give the Name of the Application
st.title('Prediction Text Suicide')

# Create Submit Form
with st.form(key='form_parameters'):
    text = st.text_area('text',max_chars = 5000)
    submitted = st.form_submit_button('Predict')

# inference
if submitted:
    URL = 'https://deploym1-nurliyanahaan.koyeb.app/predict'
    param = {
    'text': text
    }

    r = requests.post(URL, json=param)
    if r.status_code == 200:
        res = r.json()
        st.title('Text Prediction is {}'.format(res['label_names']))
    else:
        st.title("Unexpected Error")
        st.write(r.status_code)
