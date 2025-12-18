import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ImagoAI | Corn Vomitoxin Predictor",
    page_icon="->",
    layout="centered"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .result-value {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model_path = "model.joblib"
    scaler_path = "scaler.joblib"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/norman-the-zombie/master/norman.png", width=100) # Placeholder for logo
    st.title("Settings")
    st.info("Ensuring high-accuracy spectral analysis for corn safety.")
    if model and scaler:
        st.success("Model & Scaler Loaded")
    else:
        st.error("Model/Scaler missing!")

# --- MAIN UI ---
st.title("Corn Vomitoxin Predictor")
st.markdown("---")
st.write("Upload your spectral data (448 features) to predict the **Vomitoxin level (ppb)**.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file containing HSI spectra", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(data.head(5))

        # Check if features match (assuming first column might be ID)
        if data.shape[1] < 448:
            st.error(f"Error: Expected at least 448 spectral features, but found {data.shape[1]}.")
        else:
            # Handle case where first column is hsi_id
            if data.iloc[:, 0].dtype == 'object':
                ids = data.iloc[:, 0]
                features = data.iloc[:, 1:449]  # Take 448 features
            else:
                ids = [f"Sample_{i}" for i in range(len(data))]
                features = data.iloc[:, 0:448]

            if st.button("Run Prediction Analysis"):
                if model is None:
                    st.error("Critical Error: Model not found. Please upload 'model.joblib' and 'scaler.joblib' to the project folder.")
                else:
                    with st.spinner('Analyzing Spectral Signature...'):
                        # 1. Transform features
                        X_scaled = scaler.transform(features)
                        
                        # 2. Predict (Log Domain)
                        log_preds = model.predict(X_scaled)
                        
                        # 3. Inverse Log Transform
                        # We use expm1 because the model was trained on log1p(y)
                        final_preds = np.expm1(log_preds.flatten())
                        
                        # 4. Results Display
                        st.markdown("### Prediction Results")
                        results_df = pd.DataFrame({
                            "Sample ID": ids,
                            "Predicted Vomitoxin (ppb)": np.round(final_preds, 2)
                        })
                        
                        # Display best result
                        top_val = results_df.iloc[0, 1]
                        st.markdown(f"""
                            <div class="prediction-card">
                                <p>Primary Sample Prediction</p>
                                <div class="result-value">{top_val} ppb</div>
                            </div>
                            <br>
                        """, unsafe_allow_html=True)
                        
                        st.table(results_df)
                        st.download_button(
                            label="Download Results as CSV",
                            data=results_df.to_csv(index=False),
                            file_name="vomitoxin_predictions.csv",
                            mime="text/csv"
                        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("Pro Tip: Your CSV should have 448 numeric columns representing the spectral bands.")
    
    # Example input for manual testing if no file
    with st.expander("Or enter sample values manually"):
        manual_input = st.text_area("Paste 448 comma-separated values here:", placeholder="0.41, 0.39, ...")
        if st.button("Predict Manual Entry"):
            # Logic for manual entry could go here
            st.warning("Manual entry functionality requires exact 448 values. Please use file upload for best results.")
