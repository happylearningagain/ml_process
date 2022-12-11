from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()

model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    Clump_thickness: int
    Uniformity_of_cell_size: int
    Uniformity_of_cell_shape: int
    Marginal_adhesion: int
    Single_epithelial_cell_size: int
    Bare_nuclei: int
    Bland_chromatin: int
    Normal_nucleoli: int
    Mitoses: int

fastapp = FastAPI()

@fastapp.get("/")
def home():
    return "Hello, FastAPI up!"

@fastapp.post("/predict")
async def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    
    # Convert dtype
    data = data[config_data["predictors"]].astype(int)

    # Check range data
    try:
        data_pipeline.dataChecking(data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # scaling data
    data = preprocessing.standardizerData(data,2)

    # Predict data
    y_pred = str(model_data.predict(data))[1]


    return {"res" : y_pred, "error_msg": ""}


if __name__ == "__main__":
    uvicorn.run("api:fastapp", host = "0.0.0.0", port = 8000, reload = True)