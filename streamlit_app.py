import streamlit as st
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib

class PredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float 
    Churn: str
    
def preprocess_input(input_data: PredictionInput) -> pd.DataFrame:
    selected_colomns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No','OnlineSecurity_No','OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_Yes',
       'DeviceProtection_No','DeviceProtection_Yes', 'TechSupport_No','TechSupport_Yes', 'StreamingTV_No','StreamingTV_Yes',
       'StreamingMovies_No','StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check' , "Churn"]
    # Create a DataFrame from the input_data
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Perform label encoding for relevant columns
    le = LabelEncoder()
    for c in ["PaperlessBilling", "gender", "Partner", "Dependents", "PhoneService", "Churn"]:
        input_df[c] = le.fit_transform(input_df[c])

    # Perform one-hot encoding for categorical columns
    for c in ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
              "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]:
        dummy_data = pd.get_dummies(input_df[c], prefix=c)
        input_df = pd.concat([input_df, dummy_data], axis=1)
        input_df.drop(c, axis=1, inplace=True)

    # Select only the relevant columns based on selected_colomns
    preprocessed_input_df = input_df[selected_colomns]
    scaler = StandardScaler()
    preprocessed_input_df = scaler.fit_transform(preprocessed_input_df)


    return preprocessed_input_df
    
# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("logisticReg.pkl")

model = load_model()

def predict(input_data):
     # Preprocess the input data
    preprocessed_input = preprocess_input(input_data)

    # Extract the relevant columns for prediction
    prediction_input = preprocessed_input.values

    # Predict using the model
    prediction = model.predict(prediction_input)
    
    if prediction == 0: 
        return "No"
    else : return 1

def main():
    st.title("Churn Prediction App")
    st.write("Enter customer information to predict churn.")

    # Input fields for user to enter data
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.text_input("Tenure (months):")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service_options = ["DSL", "Fiber optic", "No"]
    internet_service = st.selectbox("Internet Service", internet_service_options)
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.text_input("Monthly Charges:")
    total_charges = st.text_input("Total Charges:")
    
    if tenure and monthly_charges and total_charges and st.button("Predict"):
        # Create a PredictionInput instance with the provided data
        input_data = PredictionInput(
            gender=gender,
            SeniorCitizen=int(senior_citizen),
            Partner=partner,
            Dependents=dependents,
            tenure=int(tenure),
            PhoneService=phone_service,
            MultipleLines=multiple_lines,
            InternetService=internet_service,
            OnlineSecurity=online_security,
            OnlineBackup=online_backup,
            DeviceProtection=device_protection,
            TechSupport=tech_support,
            StreamingTV=streaming_tv,
            StreamingMovies=streaming_movies,
            Contract=contract,
            PaperlessBilling=paperless_billing,
            PaymentMethod=payment_method,
            MonthlyCharges=float(monthly_charges),
            TotalCharges=float(total_charges),
        )

        # Predict using the preprocessed input
        prediction = predict(input_data)

        # Display the prediction
        st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")

if __name__ == "__main__":
    main()

