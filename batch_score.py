from azureml.core import Run
from azureml.core.model import Model
import argparse
from torchvision import datasets, transforms
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from utils import use_or_create_datastore,get_models

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name',type=str,required=True)
    arg('--model-version',type=str)
    arg('--tag-name',type=str)
    arg('--tag-value',type=str)
    arg('--step-input',type=str,required=True)
    arg('--scoring-file-name',type=str,required=True)
    arg('--loss-file-name',type=str,required=True)
    args = parser.parse_args()
    
    run = Run.get_context()
    exp = run.experiment
    print(exp)
    ws = exp.workspace
    model_name = args.model_name
    model_version = args.model_version
    tag_name = args.tag_name
    tag_value = args.tag_value
    loss_file_name = args.loss_file_name
    score_file_name = args.scoring_file_name
    
    tags = None
    if tag_name is not None or tag_value is not None:
        if tag_name is None or tag_value is None:
            raise ValueError('if either tag_name or tag_value is provided,then \
                             the other parameter must be provided as well')
        tags = [(tag_name,tag_value)]
    
    model = get_models(ws=ws,
                       model_name=model_name,
                       model_version=model_version,
                       tags=tags)
    model_path = Model.get_model_path(model_name=model_name,
                                      version=model.version)
    model = torch.load(model_path)
    model = model.eval()
    trans = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    dataset = datasets.ImageFolder(os.path.join(args.step_input, 'val'),trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    result = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if len(preds) == 4:
            result = np.array(preds) if result is None else np.vstack([result,np.array(preds)])
        loss = criterion(outputs,labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    dataset_size = len(dataset)
    epoch_loss = np.round(running_loss / dataset_size,5)
    
    keyvault = ws.get_default_keyvault()
    account_key = keyvault.get_secret('aad-secret')
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name='diabetesamlsa_out',
                                        container_name='diabetesamlsa',
                                        storage_account_name='diabetesamlsa',
                                        storage_access_key=account_key,
                                        overwrite=True,
                                        use_default=False
                                        )
    
    loss = pd.DataFrame([epoch_loss],columns=['loss'])
    loss.to_csv(loss_file_name,index=False,sep=' ')
    if not os.path.exists(loss_file_name):
        raise FileNotFoundError(f'{loss_file_name} file not found')
    datastore.upload_files(files=[loss_file_name],
                           target_path='pytorch_batchscore_results/loss',
                           overwrite=True)
    
    result = pd.DataFrame(result,columns=['1st','2nd','3rd','4th'])
    result.to_csv(score_file_name,index=False)
    if not os.path.exists(score_file_name):
        raise FileNotFoundError(f'{score_file_name} file not found')
    datastore.upload_files(files=[score_file_name],
                           target_path='pytorch_batchscore_results/preds',
                           overwrite=True)

if __name__ == '__main__':
    main()
#%%
