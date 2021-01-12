import urllib
from zipfile import ZipFile
import os
from utils import create_or_use_workspace,use_or_create_datastore
from azureml.core.authentication import ServicePrincipalAuthentication
import json
from glob import glob
import cv2
import hickle as hkl

def download_data():
    """Download and extract the training data."""
    # download data
    data_file = './fowl_data.zip'
    download_url = 'https://azureopendatastorage.blob.core.windows.net/testpublic/temp/fowl_data.zip'
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, 'r') as zip:
        print('extracting files...')
        zip.extractall()
        print('finished extracting')
        data_dir = zip.namelist()[0]

    # delete zip file
    os.remove(data_file)

    images = {}
    for split in ['val']:
        for cls in ['chickens','turkeys']:
            path = os.path.join(data_dir,split,cls,'*')
            for p in glob(path):
                img = cv2.imread(p)
                img_id = p.split('\\')[-1]
                images[img_id] = img
    
    hkl.dump(images,'image_dict.hkl')
    
    return data_dir

def upload_data():
    with open('config.json',"r") as f:
        config = json.load(f)
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = create_or_use_workspace(workspace_name=config['workspace_name'],
                                 subscription_id=config['subscription_id'],
                                 resource_group=config['resource_group'],
                                 auth=auth)
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name='pytorch_deployment_datastore')
    src_dir = download_data()
    target_path = 'pytorch'
    datastore.upload(src_dir=src_dir,
                     target_path=target_path,
                     overwrite=True,
                     show_progress=True)
    
    

if __name__ == '__main__':
    upload_data()

#%%
