import streamlit as st
import requests

st.title("Rain Predictor (24h)")

api_base = st.text_input("API base URL", "http://127.0.0.1:8000")
city = st.text_input("City", "Karachi")
country_code = st.text_input("Country code", "PK")

if st.button("Predict rain (next 24h)"):
    r = requests.post(
        f"{api_base}/predict/by-city",
        json={"city": city, "country_code": country_code},
        timeout=30,
    )
    if r.status_code != 200:
        st.error(r.text)
    else:
        data = r.json()
        st.write(f"Probability of rain in next 24h: {data['probability']:.3f}")
        st.write("Will rain (24h):", data["will_rain_24h"])
