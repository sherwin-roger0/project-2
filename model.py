import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
    
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

classes = pd.read_excel("class.xlsx")

# Load the stacking classifier model
stacking_classifier = joblib.load('stacking_classifier_model.pkl')

# Load the scaling parameters
scaling_params = joblib.load('scaling_params.pkl')

scaler = MinMaxScaler()
scaler.data_min_ = scaling_params["min"]
scaler.data_max_ = scaling_params["max"]
scaler.min_ = scaling_params["min_"]
scaler.scale_ = scaling_params["scaler_"]

def predictCrop(N_input,P_input ,K_input,temperature_input,humidity_input,ph_input,rainfall_input ):
    value = stacking_classifier.predict(scaler.transform([[N_input, P_input, K_input, temperature_input, humidity_input, ph_input, rainfall_input]]))[0]
    return classes["class"][value:value+1]


class predictCropInput(BaseModel):
    """Inputs for the predictCropTool"""

    N_input: float = Field(description="The N value must be passed to function if u don't know the N value ask the user")
    P_input: float = Field(description="The P value must be passed to function if u don't know the P value ask the user")
    K_input: float = Field(description="The K value must be passed to function if u don't know the K value ask the user")
    temperature_input: float = Field(description="The temperature value must be passed to function if u don't know the temperature value ask the user")
    humidity_input: float = Field(description="The humidity value must be passed to function if u don't know the humidity value ask the user")
    ph_input: float = Field(description="The ph value must be passed to function if u don't know the ph value ask the user")
    rainfall_input: float = Field(description="The rainfall value must be passed to function if u don't know the rainfall value ask the user")


class predictCropTool(BaseTool):
    name = "predictCropTool"
    description = """
        Useful when you want to predict the crop.
        never give random values fo the prediction
        the values N,P,K,Temperature ,humidity,ph and rainfall are the values used for prediction if u dont know these values ask the user for these values all of these values will be float or integer if the user says invalid answer tell the user to give proper value u can ask the user multiple times until u get the proper value otherwise dont use this too
        """
    args_schema: Type[BaseModel] =predictCropInput
    
    def _run(self, N_input: float,P_input: float,K_input: float,temperature_input: float,humidity_input: float,ph_input: float,rainfall_input: float):
        predictCrop_response = predictCrop(N_input,P_input ,K_input,temperature_input,humidity_input,ph_input,rainfall_input )
        return predictCrop_response

    def _arun(self, N_input: float,P_input: float,K_input: float,temperature_input: float,humidity_input: float,ph_input: float,rainfall_input: float):
        predictCrop_response = predictCrop(N_input,P_input ,K_input,temperature_input,humidity_input,ph_input,rainfall_input )
        return predictCrop_response