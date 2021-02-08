import os
import argparse
import json
from azureml.core import Run
from azureml.core.model import Model
import torch

def register_model(run_id,
                   exp,
                   model_name,
                   model_tags,
                   model_path):
    try:
       tags = {'title':'pytorch_deployment',
               'experiment_name':exp.name,
               'run_id':run_id}
       if len(model_tags) > 0:
           tags.update(model_tags)
       model = Model.register(workspace=exp.workspace,
                              model_path=model_path,
                              model_name=model_name,
                              tags=tags)
       print(f'{model_name} has been registered,version is: {model.version}')
    except Exception as error:
        print(error)
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name',type=str,
        default='pytorch_model.pkl') 
    arg('--step-input',type=str)
    args = parser.parse_args()
    
    run = Run.get_context()
    exp = run.experiment
    model_name = args.model_name
    step_input = args.step_input
    
    with open('config.json','r') as f:
         config = json.load(f)
    try:
        register_tags = config['registration']
    except KeyError:
        register_tags = {'tags':[]}
    model_tags = {}
    for param in register_tags['params']:
        try:
            mtag = run.parent.get_metrics()[param]
            model_tags[param] = mtag
        except KeyError:
            print('could not find key in tags')
    
    for tag in register_tags['tags']:
        try:
            if tag == 'best_val_acc':
                mtag = max(run.parent.get_metrics()[tag])
                model_tags[tag] = mtag
        except KeyError:
            print('could not find key in tags')
    
    model_path = os.path.join(step_input,model_name)
    print(f'loading from {model_path}')
    model = torch.load(model_path)
    
    if model is not None:
        run_id = run.id
        register_model(run_id=run_id,
                       exp=exp,
                       model_name=model_name,
                       model_tags=model_tags,
                       model_path=model_path)
    else:
        print('no model is found')
        exit(1)

if __name__ == '__main__':
    main()


