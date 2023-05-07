from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

import logging
import numpy as np

from joblib import load

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)

class InputFeatures(BaseModel):
    feature_values: list[float]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/models/predict")
def make_prediction(input_features: InputFeatures):
    svc = load('models/svc.joblib')
    n_feat = svc.n_features_in_
    try:
        feats = np.array(
            input_features.feature_values
        ).reshape((1, n_feat))
    except: #TODO handle exception properly
        message = f"Incorrect number of input features. Sent: {len(input_features.feature_values)}, required: {n_feat}"
        logger.debug(message)
        return {"error": message}
    probs = svc.predict_proba(feats)
    logger.debug(probs.shape)
    output = {}
    for i, cl in enumerate(svc.classes_):
        output[int(cl)] = probs[0][i]
    return output