from azureml.core.model import Model
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils import data
import cv2
import json
import pandas as pd
from azureml.core import Dataset,Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

class ImageDataset(data.Dataset):
    def __init__(self,
                 df,
                 mode='train',
                 transform=None):
        assert mode in ['train','val']
        self.transform = transform
        self.df = df[df.split == mode]
        self.mode = mode
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        cls,image_id = row['class'],row['image_id']
        path = os.path.join(self.mode,cls,image_id)
        image = cv2.imread(path)
        if self.transform is not None:
            image = self.transform(image=image)
        
        return image['image']
    
    def __len__(self):
        return len(self.df)

def init():
    global model
    print(os.getenv('AZUREML_MODEL_DIR'))
    model_path = Model.get_model_path(os.getenv('AZUREML_MODEL_DIR').split('/')[-2])
    model = torch.load(model_path)

def run(raw_data):
    print('running inference!')
    raw_data = json.loads(raw_data)
    df = pd.DataFrame(raw_data['data'],columns=['split','class','image_id'])
    auth = ServicePrincipalAuthentication(tenant_id='de4772cb-0d03-4f05-a9ad-e2581268c37c',
                                          service_principal_id='de6944f7-6f0c-4c66-9e41-ce53650629d6',
                                          service_principal_password='S0eGD_d~65n.hsUmm~scgS.ctp~F3m82ey')
    ws = Workspace.get(name='aml-workspace',
                       resource_group='aml-resource-group',
                       subscription_id='64c727c2-4f98-4ef1-a45f-09eb33c1bd59',
                       auth=auth)
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore,'pytorch'))
    dataset.download('.',overwrite=True)
    if not os.path.exists('train') or not os.path.exists('val'):
        raise FileNotFoundError('train or val or both are not found on local drive')
    
    trans = A.Compose([
                      A.Resize(256,256),
                      A.CenterCrop(224,224),
                      A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ToTensor()])
    dataset = ImageDataset(df,'val',trans)
    dl = data.DataLoader(dataset,batch_size=8)
    print('successfully build dl')
    
    result = None
    for image in dl:
        preds = model(image)
        preds = preds.data.numpy()
        result = preds if result is None else np.vstack([result,preds])
    
    return result
    
    #%%
#service.get_logs()