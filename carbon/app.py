import streamlit as st
import requests
import json
import re

# OpenRouter API setup
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://apillm.mobiloittegroup.com/api/generate"

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background-color: black;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>label, .stSelectbox>label, .stSlider>label {
        font-size: 18px;
        color: #FFFFFF;
        font-weight: bold;
    }
    .stTextArea>label {
        font-size: 18px;
        color: #FFFFFF;
        font-weight: bold;
    }
    .result-box {
        background-color: #1C2526;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-size: 16px;
        line-height: 1.6;
        color: #FFFFFF;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-size: 36px;
    }
    .total-emission {
        font-size: 28px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("🌿 Your Monthly Carbon Footprint Calculator")

with st.form(key="carbon_form"):
    st.subheader("Tell us about yourself")

    # Age
    age = st.slider("Your age", 10, 100, 30)

    # Mode of transport
    transport = st.selectbox("Primary mode of transport", [
                             "Foot", "Bicycle", "Bus", "Car", "Motorcycle"])

    # Commute distance with unit selection
    distance_unit = st.selectbox("Distance unit", ["Miles", "Kilometers"])
    max_distance = 1000 if distance_unit == "Miles" else 1600  # 1600 km ≈ 1000 miles
    unit_label = "miles" if distance_unit == "Miles" else "km"
    transport_distance = st.slider(
        f"Monthly travel distance ({unit_label})", 0, max_distance, 100)

    # Commute habits description
    commute_habits = st.text_input("Describe your commute habits (e.g., additional trips by car, bike, or other modes)",
                                   "Mostly commute to work, occasional weekend drives.")

    # Vehicle details for Car or Motorcycle
    vehicle_model = None
    vehicle_year = None
    if transport in ["Car", "Motorcycle"]:
        vehicle_model = st.text_input(
            f"{transport} model number (e.g., Toyota Prius)", "")
        vehicle_year = st.slider(
            f"Year {transport.lower()} was purchased", 1980, 2025, 2020)

    # Eating habits
    diet = st.selectbox("Eating habits", ["Vegetarian", "Non-Vegetarian"])

    # Garbage habits
    garbage_kg = st.slider("Approx. monthly garbage (kg)", 0, 100, 10)
    recycle_percent = st.slider(
        "Percentage of garbage recycled (%)", 0, 100, 50)
    garbage_description = st.text_area("Describe your garbage habits (e.g., types of waste, composting)",
                                       "I throw out food scraps, plastic packaging, and some paper.")

    # Electricity consumption
    electricity_kwh = st.slider(
        "Monthly electricity consumption (kWh)", 0, 2000, 200)
    country = st.text_input("Country of residence", "United States")

    # Flight details
    flight_details = st.text_input(
        "Describe your flight details if any (e.g., number of flights per month, destinations)", "None")

    # Additional information
    additional_info = st.text_area("Additional details (e.g., area, hobbies, other activities)",
                                   "I live in a suburban area and occasionally fly for work.")

    # Submit button
    submit_button = st.form_submit_button(
        label="Calculate My Carbon Footprint")

# Process form submission
if submit_button:
    # Format input data into a prompt
    prompt = f"""
    I have the following details about a person:
    - Age: {age}
    - Primary mode of transport: {transport}, monthly distance: {transport_distance} {unit_label}
    - Commute habits: {commute_habits}
    """
    if transport in ["Car", "Motorcycle"]:
        prompt += f"  - {transport} model number: {vehicle_model}, purchased in {vehicle_year}\n"
    prompt += f"""
    - Eating habits: {diet}
    - Monthly garbage: {garbage_kg} kg, {recycle_percent}% recycled
    - Garbage habits description: {garbage_description}
    - Monthly electricity consumption: {electricity_kwh} kWh
    - Country of residence: {country}
    - Flight details: {flight_details}
    - Additional information: {additional_info}

    Based on this, estimate their approximate monthly carbon emissions in kg CO2e. Provide a detailed breakdown of emissions from transport, diet, electricity, waste, and flights (if applicable). Include a line at the start of your response stating 'Total monthly carbon emissions: X kg CO2e' where X is the total estimated value (use 'approximately' if the estimate is not exact). If you cannot provide an exact total at the start, clearly state the total somewhere in the response. Then, suggest specific ways to reduce their carbon footprint, including the potential impact of planting trees or plants if applicable. Return the response as a well-structured text.
    """

    # Call OpenRouter API
    headers = {
        
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        with st.spinner("Calculating your carbon footprint..."):
            response = requests.post(
                OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            carbon_response = result["response"]

        # Extract total emission line with flexibility
        total_match = re.search(
            r"Total monthly carbon emissions: (approximately )?(\d+\.?\d*) kg CO2e", carbon_response, re.IGNORECASE)
        if total_match:
            total_emission = total_match.group(0)
            carbon_response = carbon_response.replace(
                total_emission, "").strip()
        else:
            # Fallback: Look for any number followed by "kg CO2e" as a potential total
            fallback_match = re.search(
                r"(\d+\.?\d*)\s*kg CO2e", carbon_response)
            total_emission = f"Total monthly carbon emissions: {fallback_match.group(0)}" if fallback_match else "Total not found in response"
            if fallback_match:
                carbon_response = carbon_response.replace(
                    fallback_match.group(0), "[Total extracted above]").strip()

        # Display result
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("Your Carbon Footprint")
        st.markdown(
            f"<div class='total-emission'>{total_emission}</div>", unsafe_allow_html=True)
        st.write(carbon_response)
        st.markdown("</div>", unsafe_allow_html=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting the AI model: {e}")
        st.write("Please check your API key or try again later.")