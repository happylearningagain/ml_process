import streamlit as st
import requests
from PIL import Image


header_images = Image.open("assets/breastcancer.jpg")
st.image(header_images, use_column_width=True)

# Add some information about the service

st.title("Breast Cancer Prediction")
st.subheader("Just enter variabel below then click Predict button")

# Create form of input
with st.form(key = "prediction_form"):
    # Create select box input
    clump = st.number_input( 
        label ="1.Enter Clump Thickness value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10"        
    )

    # Create box for number input
    cellsize = st.number_input(
        label = "2.Enter Uniformity of cell size's value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )
    
    cellshape = st.number_input(
        label = "3.Enter Uniformity of cell shape's value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )

    adhesion = st.number_input(
        label = "4.Enter Marginal adhesion Value:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 0 to 10"
    )

    epitelsize = st.number_input(
        label = "5.Enter Single epithelial cell size's value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )

    barenuclei = st.number_input(
        label = "6.Enter Bare nuclei value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )

    blandchrom = st.number_input(
        label = "7.Enter Bland chromatin value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )

    normnuc = st.number_input(
        label = "8.Enter Normal nucleoli value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )

    mitoses = st.number_input(
        label = "9.Enter Mitoses value",
        min_value = 1,
        max_value = 10,
        help = "Integer value from 1 to 10" 
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            'Clump_thickness': clump,
            'Uniformity_of_cell_size': cellsize,
            'Uniformity_of_cell_shape': cellshape,
            'Marginal_adhesion': adhesion,
            'Single_epithelial_cell_size': epitelsize,
            'Bare_nuclei': barenuclei,
            'Bland_chromatin': blandchrom,
            'Normal_nucleoli': normnuc,
            'Mitoses': mitoses
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://127.0.0.1:8000/predict", json = raw_data).json()
        
        # Parse the prediction result
        if res["res"] == '2':
            st.warning("Predicted Breast Cancer Type: Benign.")
        else:
            st.success("Predicted Breast Cancer Type: Malignant.")