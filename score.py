from azureml.core.model import Model
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils import data
import cv2

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

def run(df):
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
