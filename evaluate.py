import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score
import os

from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI']= "https://dagshub.com/manmohan-creator/mymlpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']= "manmohan-creator"
os.environ['MLFLOW_TRACKING_PASSWORD']= "28c2d2929e6427530838d43e345b359ba75beda6"

params= yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path, model_path):
    data= pd.read_csv(data_path)
    X= data.drop(columns=['Outcome'])
    y= data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/manmohan-creator/mymlpipeline.mlflow")

    model= pickle.load(open(model_path, 'rb'))
    predictions= model.predict(X)
    accuracy= accuracy_score(y,predictions)
    mlflow.log_metric("accuracy", accuracy)
    print(f"model accuracy: {accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])