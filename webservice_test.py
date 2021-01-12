from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import AksWebservice
from azureml.core import Dataset
from env_variables import ENV
from utils import create_or_use_workspace,use_or_create_datastore
import json
import argparse
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils import data
import cv2
import os
import pandas as pd
from azureml.core.model import Model
import torch

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

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input_datapath',type=str,required=True)
    args = parser.parse_args()
    data_path = args.input_datapath

    env = ENV()
    with open('config.json','r') as f:
        config  = json.load(f)
    
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = create_or_use_workspace(env.workspace_name,
                                 env.subscription_id,
                                 env.resource_group,
                                 auth=auth)
    
    aks_service = AksWebservice(ws,env.aks_service_name)
    print(f'scoring url:{aks_service.scoring_uri}')
    
    datastore = use_or_create_datastore(ws = ws)
    dataset = Dataset.File.from_files(path=(datastore,data_path))
    df = dataset.download('.',overwrite=True)
    if not os.path.exists('train') or not os.path.exists('val'):
        raise FileNotFoundError('train or val or both are not found on local drive')
    
    info = {col:[] for col in ['split','class','image_id']}
    for path in df:
        split_info = path.split('/')
        split,cls,image_id = split_info[-3],split_info[-2],split_info[-1]
        info['split'].append(split)
        info['class'].append(cls)
        info['image_id'].append(image_id)
    df = pd.DataFrame.from_dict(info,orient='columns')
    del_index = df[df.image_id == 'desktop.ini'].index[0]
    df.drop([del_index],inplace=True)

    trans = A.Compose([
                      A.Resize(256,256),
                      A.CenterCrop(224,224),
                      A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ToTensor()])
    dataset = ImageDataset(df,'val',trans)
    dl = data.DataLoader(dataset,batch_size=env.batch_size)
    print('successfully build dl')
    model = Model(name='pytorch_model.pt',
                  version='9')
    model_path = Model.get_model_path(model_nam=model.name,
                                      version=model.version,
                                      _workspace=ws)
    model = torch.load(model_path)
    print('model loaded')
    for image in dl:
        print(model(image))

    try:
        preds = aks_service.run(dl)
        print(preds)
    except Exception as error:
        print(error)
        exit(1)

if __name__ == '__main__':
    main()
    #%%
#from azureml.core import Workspace
#from azureml.core.webservice import AksWebservice
#ws = Workspace.get(name='aml-workspace',
#                   subscription_id='64c727c2-4f98-4ef1-a45f-09eb33c1bd59',
#                   resource_group='aml-resource-group')
#service = AksWebservice(ws,'pytorch1')
#%%
#from azureml.core.model import Model
#import torch
#model = Model(name='pytorch_model.pt',
#              version='9',
#              workspace=ws)
#model_path = Model.get_model_path(model_name=model.name,
#                                  version=model.version,
#                                  _workspace=ws)
#model = torch.load(model_path)