from azureml.core import Workspace, Run, Experiment
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_metrics',type=str,required=True)
    arg('--model_dir',type=str,required=True)
    arg('--src_model_name',type=str,required=True,default='src_pytorch_model.pt')
    arg('--exp_name',type=str,default='PYTORCH_DEPLOYMENT',required=True)
    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace
    run_details = run.get_details()
    print(run_details)
    
    data_metrics = args.data_metrics
    print(data_metrics) 
    data_metrics_parent = os.path.dirname(data_metrics)
    with open(os.path.join(data_metrics_parent,'data_metrics')) as f:
        metrics = json.load(f)
    print(metrics)
    
    best_val_acc = 0
    best_run_id = None
    for run in metrics.keys():
        current_run_metric = metrics[run]
        if 'best_val_acc' in current_run_metric:
            acc = current_run_metric['best_val_acc'][-1]
            if acc > best_val_acc:
                best_val_acc = acc
                best_run_id = run
    
    exp = Experiment(workspace=ws,
                     name=args.exp_name)
    best_run = Run(exp,best_run_id)
    
    model_dir= os.path.join(args.model_dir,args.src_model_name)
    tags = {'run_id':best_run_id,
            'best_val_acc':best_val_acc}
    best_run.register_model(model_name=args.src_model_name,
                            model_path=model_dir,
                            tags=tags)

if __name__ == '__main__':
    main()
#%%