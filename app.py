from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = FastAPI()

# Load your trained model
model = joblib.load("logisticReg.pkl")

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

def preprocess_input(input_data: PredictionInput) -> pd.DataFrame:
    selected_colomns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No','OnlineSecurity_No','OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_Yes',
       'DeviceProtection_No','DeviceProtection_Yes', 'TechSupport_No','TechSupport_Yes', 'StreamingTV_No','StreamingTV_Yes',
       'StreamingMovies_No','StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check' ]
    # Create a DataFrame from the input_data
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Perform label encoding for relevant columns
    le = LabelEncoder()
    for c in ["PaperlessBilling" , "gender" , "Partner" ,"Dependents" ,"PhoneService" ,"PaperlessBilling"]:
        input_df[c] = le.fit_transform(input_df[c])

    # Perform one-hot encoding for categorical columns
    for c in ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]:
        dummy_data = pd.get_dummies(input_df[c], prefix=c)
        input_df = pd.concat([input_df, dummy_data], axis=1)
        input_df.drop(c, axis=1, inplace=True)
    # Select only the relevant columns based on selected_colomns
    #print(input_df)
    #preprocessed_input_df = input_df[selected_colomns]
    zeros_data = np.zeros((len(input_df), len(selected_colomns)))
    zeros_df = pd.DataFrame(zeros_data, columns=selected_colomns)

    # Iterate through the selected columns and add them to input_df with zeros if they exist, or directly add zeros if they don't
    for col in selected_colomns:
        if col in input_df.columns:
            zeros_df[col] = input_df[col]
        else:
            input_df[col] = zeros_df[col]

# Update input_df with the combined data
    input_df = zeros_df

    return input_df

@app.post("/predict/")
async def predict_churn(input_data: PredictionInput):
    # Preprocess the input data
    preprocessed_input = preprocess_input(input_data)

    # Extract the relevant columns for prediction
    prediction_input = preprocessed_input.values

    # Scale the input data
    scaler = StandardScaler()
    prediction_input = scaler.fit_transform(prediction_input)

    # Predict using the model
    prediction = model.predict(prediction_input)

    result = "Churn" if prediction == 1 else "No Churn"
    return {"prediction": result}

def main():
    import uvicorn
    uvicorn.run("app:app", port=8080, reload=True)

if __name__ == "__main__":
    main()
