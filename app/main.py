import os
import joblib
import importlib
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

import logging
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)

class InputFeatures(BaseModel):
    feature_values: dict
    model_name: str

class TrainingDataset(BaseModel):
    data: dict
    targets: list
    model_save_name: str
    model_module: str
    model_class: str
    model_hyperparameters: dict

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/models/predict")
def make_prediction(input_features: InputFeatures):
    svc = joblib.load(os.path.join('models', f'{input_features.model_name}.joblib'))
    feat_names = input_features.feature_values.keys()
    logger.debug(input_features.feature_values)
    try:
        n_rows = len(input_features.feature_values[list(feat_names)[0]])
    except TypeError:
        # single row passed
        n_rows = 1
    feats = pd.DataFrame(input_features.feature_values, index=range(n_rows))
    # ensure that the column orders match
    for feat_name in feats.columns:
        if feat_name not in svc.feature_names_in_:
            msg = f"{feat_name} provided but not in original features"
            logger.debug(feat_name)
            return {'error': msg} 
    feats = feats.reindex(columns=svc.feature_names_in_)
    probs = svc.predict_proba(feats)
    
    output = {
        'model_name': input_features.model_name,
        'feature_values': input_features.feature_values,
        'predictions': {}
    }
    for i, cl in enumerate(svc.classes_):
        output['predictions'][int(cl)] = list(probs[:, i])
    return output

@app.post("/models/train")
def train_model(training_data: TrainingDataset):
    n_feat = len(training_data.data)
    n_rows = len(training_data.targets)
    logger.debug("Training data has %d features and %d rows", n_feat, n_rows)
    X = pd.DataFrame(training_data.data)
    y = training_data.targets
    logger.debug(y)

    class_module = importlib.import_module(training_data.model_module)
    classifier_class = getattr(class_module, training_data.model_class)
    classifier = classifier_class(**training_data.model_hyperparameters)
    classifier.fit(X, y)
    train_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
    output = {
        'train_auc': train_auc
    }
    joblib.dump(classifier, os.path.join("models", f"{training_data.model_save_name}.joblib"))
    return output