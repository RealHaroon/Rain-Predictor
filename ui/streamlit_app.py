import streamlit as st
import requests

st.title("Rain Predictor (MVP)")

api_base = st.text_input("API base URL","http://127.0.0.1:8000")

if st.button("Check API health"):
    r=requests.get(f"{api_base}/health",timeout=10)
    st.json(r.json())