import streamlit as st
import tensorflow as tf
from pathlib import Path
from utils import *

st.title("Behavior Prediction Model Based on on Jobads Description")

jobdas_text_input: str = st.text_input("Insert your best job description!")
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
        try:
            user_input = input_pipeline(jobdas_text_input, work_location)
        except Exception as e1:
            st.error("ERROR e1: input_pipeline()")
            print(f"ERROR e1:{e1}")
        try:
            model_preds = predict(user_input)
        except Exception as e2:
            st.error("ERROR e2: predict()")
            print(f"ERROR e2:{e2}")
        try:
            output, pred_probs = process_prediction(model_preds)
        except Exception as e3:
            st.error("ERROR e3: process_prediction()")
            print(f"ERROR e1:{e3}")

        if output == "V":
            st.caption("User predicted behavior: View")
            st.caption(f"Probability: {pred_probs}")
        else:
            st.caption("User predicted behavior: Application")
            st.caption(f"Probability: {pred_probs}")
