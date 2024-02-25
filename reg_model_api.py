from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Create a FastAPI instance
app = FastAPI()

# Create a class that will have the request body data model
class request_body(BaseModel):
    study_hours: float

# Load the model
grade_model = joblib.load('regression_model.pkl')

# Create a route for the API
@app.post('/predict')
def predict(data : request_body):
  # Prepare data for prediction
    input_feature = [[data.study_hours]]
  # Predict
    y_pred = grade_model.predict(input_feature)[0].astype(int)
  # Return the prediction
    return {'grade_test': y_pred.tolist()}