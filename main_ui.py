import streamlit as st
import tensorflow as tf
from pathlib import Path
from utils import *

st.title("Behavior Prediction by Jobads Text Description and Location")
st.caption(
    "By using job advertistment and location as input, model will predict the behavior of user either View or Apply."
)

jobdas_text_input: str = st.text_area(
    "Insert your best job description!", "Excecutive of..."
)
work_location = st.selectbox(
    "Pick a location!",
    sorted(
        [
            "Melbourne",
            "Sydney",
            "Hawkes Bay",
            "Auckland",
            "Northern QLD",
            "Newcastle, Maitland & Hunter",
            "Gosford & Central Coast",
            "Northland",
            "Rockhampton & Capricorn Coast",
            "Adelaide",
            "ACT",
            "Brisbane",
            "Sunshine Coast",
            "Mildura & Murray",
            "Perth",
            "Wagga Wagga & Riverina",
            "Bay of Plenty",
            "Wollongong, Illawarra & South Coast",
            "Port Hedland, Karratha & Pilbara",
            "Wellington",
            "Blue Mountains & Central West",
            "Cairns & Far North",
            "Southern Highlands & Tablelands",
            "Canterbury",
            "Yorke Peninsula & Clare Valley",
            "Albury Area",
            "Waikato",
            "Toowoomba & Darling Downs",
            "Tasman",
            "Gold Coast",
            "Taranaki",
            "Manawatu",
            "Asia Pacific",
            "Kalgoorlie, Goldfields & Esperance",
            "South West Coast VIC",
            "Bairnsdale & Gippsland",
            "Dubbo & Central NSW",
            "Ballarat & Central Highlands",
            "Geraldton, Gascoyne & Midwest",
            "Alice Springs & Central Australia",
            "Mackay & Coalfields",
            "Darwin",
            "West Coast",
            "Yarra Valley & High Country",
            "Bunbury & South West",
            "Mandurah & Peel",
            "Broome & Kimberley",
            "Southland",
            "Somerset & Lockyer",
            "Port Macquarie & Mid North Coast",
        ]
    ),
)
st.code("Click to predict user behavior!")
button = st.button("Predict")

if button:
    if not jobdas_text_input:
        st.error("No input in Text input.")

    if not work_location:
        st.error("No input in Location input.")

    else:
        with st.spinner("Predicting..."):
            try:
                user_inputs = input_pipeline([jobdas_text_input], [work_location])
            except Exception as e1:
                st.error("ERROR e1: input_pipeline()")
                st.error(f"ERROR e1:{e1}")
            try:
                model_preds = predict(user_inputs)
            except Exception as e2:
                st.error("ERROR e2: predict()")
                st.error(f"ERROR e2:{e2}")
            try:
                output, pred_probs = process_prediction(model_preds)
                pred_probs = round(float(pred_probs), 2)
            except Exception as e3:
                st.error("ERROR e3: process_prediction()")
                st.error(f"ERROR e3:{e3}")

            if output == "V":
                st.header("Results:")
                st.subheader("Predicted behavior: View")
                st.subheader(f"Probability: {pred_probs}")
                st.balloons()
            else:
                st.header("Results:")
                st.subheader("Predicted behavior: Application")
                st.subheader(f"Probability: {pred_probs}")
                st.balloons()
