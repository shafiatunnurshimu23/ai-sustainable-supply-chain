# Final, Fully Integrated dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Set up the page configuration ---
st.set_page_config(layout="wide")

# --- Helper Function to Reduce Memory Usage ---
# This is still necessary for the initial raw data load.
def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if 'datetime' in str(col_type): continue
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            if len(df[col].unique()) / len(df[col]) < 0.5: df[col] = df[col].astype('category')
    return df

# --- Load a trained model and required components ---
@st.cache_resource
def load_model_and_components():
    try:
        model = joblib.load('random_forest_classifier.joblib')
        model_cols = joblib.load('model_columns.joblib')
        return model, model_cols
    except FileNotFoundError:
        return None, None

# --- Master function to load and process all data from raw sources ---
@st.cache_data
def load_and_process_all_data():
    print("--- CACHE MISS: Source file changed. Re-running all data processing. ---")
    try:
        supply_df = pd.read_excel('Supply chain logistics problem.xlsx')
        world_energy_df = pd.read_csv('World Energy Consumption.csv')
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: A source file is missing. Details: {e}")
        return None

    # Clean and Optimize
    supply_df.columns = supply_df.columns.str.lower().str.replace(' ', '_')
    supply_df = optimize_memory(supply_df)
    world_energy_df = optimize_memory(world_energy_df)
    supply_df.dropna(subset=['tpt', 'weight', 'carrier'], inplace=True)

    # Process Energy Data
    co2_col = next((col for col in ['carbon_intensity_elec', 'co2'] if col in world_energy_df.columns), 'carbon_intensity_elec')
    energy_col = next((col for col in ['primary_energy_consumption'] if col in world_energy_df.columns), 'primary_energy_consumption')
    energy_2019_df = world_energy_df[world_energy_df['year'] == 2019].copy()
    energy_2019_df = energy_2019_df[['country', co2_col, energy_col]].dropna()
    energy_2019_df['carbon_intensity_of_energy'] = energy_2019_df[co2_col] / (energy_2019_df[energy_col] + 1e-9)
    country_lookup_df = energy_2019_df[['country', 'carbon_intensity_of_energy']].rename(columns={'country': 'origin_country'})

    # Feature Engineering and Merging
    AVG_KM_PER_DAY = 350
    supply_df['estimated_distance_km'] = supply_df['tpt'].astype('int32') * AVG_KM_PER_DAY
    carrier_to_mode_map = {carrier: 'Sea' for carrier in supply_df['carrier'].unique()}
    supply_df['transportation_mode'] = supply_df['carrier'].map(carrier_to_mode_map)
    emission_factors_kg_km = {'Sea': 0.000015}
    supply_df['emission_factor'] = supply_df['transportation_mode'].map(emission_factors_kg_km)
    supply_df['transport_footprint_kg'] = supply_df['estimated_distance_km'] * supply_df['weight'] * supply_df['emission_factor']
    port_to_country_map = {'PORT09': 'United States'}
    supply_df['origin_country'] = supply_df['origin_port'].map(port_to_country_map).fillna('United States')
    
    enriched_df = pd.merge(supply_df, country_lookup_df, on='origin_country', how='left')
    enriched_df['carbon_intensity_of_energy'].fillna(country_lookup_df['carbon_intensity_of_energy'].mean(), inplace=True)
    ENERGY_PER_KG = 1.5
    enriched_df['manufacturing_footprint_kg'] = enriched_df['weight'] * ENERGY_PER_KG * enriched_df['carbon_intensity_of_energy']
    enriched_df['total_footprint_kg'] = enriched_df['transport_footprint_kg'] + enriched_df['manufacturing_footprint_kg']
    
    return enriched_df

# --- Main Dashboard Logic ---
st.title('AI-Powered Sustainable Supply Chain Dashboard')

# Load model and historical data
model, model_cols = load_model_and_components()
df = load_and_process_all_data()

if df is not None:
    # --- The Predictive Forecaster Tool ---
    st.markdown("---")
    st.header('🚀 Shipment Carbon Forecaster')
    st.write("Enter the details of a planned shipment to predict its emission category.")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # TPT is not a strong predictor, so we can give it a smaller range.
        tpt_input = st.slider('Transport Time (TPT) in Days', 1, 10, 2)
        
        # *** THE KEY CHANGE IS HERE: Update the range based on your findings ***
        # We set the max value much higher to allow for "extreme" inputs.
        unit_quantity_input = st.number_input('Number of Units', min_value=1, max_value=600000, value=1000, step=100)
        st.caption("Try setting units above ~5300 to see a potential change.")

    with col2:
        carrier_input = st.selectbox('Select Carrier', options=df['carrier'].unique())
        transportation_mode_input = 'Sea' # Based on our data, this is the only mode
    
    with col3:
        carbon_intensity_input = st.number_input('Origin Carbon Intensity (proxy)', value=0.03, disabled=True)
        st.text("") 
        predict_button = st.button('Predict Emission Category', type="primary")

    if predict_button:
        if model is not None and model_cols is not None:
            input_data = {
                'tpt': [tpt_input],
                'unit_quantity': [unit_quantity_input],
                'carbon_intensity_of_energy': [carbon_intensity_input],
                'carrier': [carrier_input],
                'transportation_mode': [transportation_mode_input]
            }
            input_df = pd.DataFrame(input_data)
            input_encoded = pd.get_dummies(input_df)
            final_input = input_encoded.reindex(columns=model_cols, fill_value=0)

            prediction = model.predict(final_input)
            prediction_proba = model.predict_proba(final_input)

            if prediction[0] == 1:
                st.error(f"Prediction: **HIGH-EMISSION SHIPMENT** (Confidence: {prediction_proba[0][1]:.0%})")
            else:
                st.success(f"Prediction: **LOW-EMISSION SHIPMENT** (Confidence: {prediction_proba[0][0]:.0%})")
        else:
            st.error("Model files not found. Please run the ML training script to generate them.")

    # --- Historical Data Analysis Section ---
    st.markdown("---")
    st.header('Historical Data Analysis')
    
    total_footprint_tons = df['total_footprint_kg'].sum() / 1000
    st.metric("Total Historical Footprint (tons CO2e)", f"{total_footprint_tons:,.0f}")
    
    st.subheader("Explore Historical Shipments")
    st.dataframe(df)