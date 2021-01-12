from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import AksWebservice
from azureml.core import Dataset
from env_variables import ENV
from utils import create_or_use_workspace,use_or_create_datastore
import json
import argparse
import os
import pandas as pd

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

    try:
        preds = aks_service.run(df)
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
