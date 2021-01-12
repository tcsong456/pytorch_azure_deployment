from azureml.core.model import Model
import os
import torch
import numpy as np

def init():
    global model
    print(os.getenv('AZUREML_MODEL_DIR'))
    model_path = Model.get_model_path(os.getenv('AZUREML_MODEL_DIR').split('/')[-2])
    model = torch.load(model_path)

def run(dataloader):
    result = None
    model.eval()
    for image in dataloader:
        preds = model(image)
        preds = preds.data.numpy()
        if result is None:
            result = preds if result is None else np.vstack([result,preds])
    
    return result
    
    #%%
