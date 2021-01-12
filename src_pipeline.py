from utils import *
from env_variables import ENV
from azureml.core import ScriptRunConfig,RunConfiguration,Dataset,Datastore
from azureml.pipeline.core import Pipeline,PipelineData,PipelineParameter
from azureml.pipeline.steps import PythonScriptStep,ParallelRunConfig,ParallelRunStep,HyperDriveStep
from azureml.data.datapath import DataPath,DataPathComputeBinding
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.core import Experiment
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BayesianParameterSampling,choice,uniform,HyperDriveConfig,PrimaryMetricGoal
import os
import json
from env_variables import ENV

def main():
    env = ENV()
    with open('config.json','r') as f:
        config = json.load(f)
    auth = ServicePrincipalAuthentication(tenant_id=config['tenant_id'],
                                          service_principal_id=config['service_principal_id'],
                                          service_principal_password=config['service_principal_password'])
    ws = create_or_use_workspace(env.workspace_name,
                                 env.subscription_id,
                                 env.resource_group,
                                 auth=auth)
    gpu_compute_target = use_or_create_computetarget(ws=ws,
                                                     env=env,
                                                     compute_name='gpu-cluster')
    cpu_compute_target = use_or_create_computetarget(ws=ws,
                                                     env=env,
                                                     compute_name='train-cluster')
    environment = use_or_create_environment(ws=ws,
                                            env_name=env.env_name,
                                            conda_dependencies=env.conda_dependencies,
                                            create_new_env=env.create_new_env,
                                            )
    datastore = use_or_create_datastore(ws=ws,
                                        datastore_name=env.datastore_name
                                        )
    
    runconfig = RunConfiguration()
    runconfig.environment = environment

    keyvault = ws.get_default_keyvault()
    keyvault.set_secrets({'aad-secret':config['aad_secret']})
    if not os.path.exists('upload.sh'):
        raise FileNotFoundError('please create a upload.sh shell to upload data to datstore first')
    with open('shell.txt') as f:
        content = f.read()
        if len(content) > 1:
            pass
        else:
            raise ValueError('please run upload.sh shell first')
    
    datapath = DataPath(datastore=datastore,
                        path_on_datastore='pytorch')
    input_data_param = PipelineParameter('input_data_path',default_value=datapath)
    input_datapath = (input_data_param,DataPathComputeBinding(mode='mount'))
#    output = PipelineData('output',datastore=datastore)
    outputs = 'outputs'
    os.makedirs(outputs,exist_ok=True)
    model_name_param = PipelineParameter(name='model_name',default_value=env.src_model_name)
    exp_name_param = PipelineParameter('exp_name',default_value=env.experiment_name)
    data_metrics = PipelineData('data_metrics',datastore=datastore)
#    model_epoch_param = PipelineParameter(name='epochs',default_value=15)

    est = Estimator(entry_script='pytorch_train.py',
                    source_directory='.',
                    node_count=1,
                    compute_target=gpu_compute_target,
                    environment_definition=environment)
    
    params = BayesianParameterSampling({
                                        '--num_epochs':choice(5,10),
                                        '--learning_rate':uniform(0.001,0.1),
                                        '--momentum':uniform(0.5,0.9)
                                        })
    
    hdc = HyperDriveConfig(estimator=est,
                           hyperparameter_sampling=params,
                           primary_metric_name='best_val_acc',
                           primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                           max_total_runs=env.max_total_runs,
                           max_concurrent_runs=env.max_concurrent_runs,
                           max_duration_minutes=env.max_duration_minutes)
    
    train_step = HyperDriveStep(name='train_step',
                                estimator_entry_script_arguments=['--output_dir',outputs,
                                                                  '--data-path',input_datapath,
                                                                  '--model-name',model_name_param],
                                inputs=[input_datapath],
#                                outputs=[outputs],
                                hyperdrive_config=hdc,
                                metrics_output=data_metrics,
                                allow_reuse=True)
    print('train_step built')   
    
    register_step = PythonScriptStep(script_name='register_src.py',
                                     name='register_step',
                                     arguments=['--data_metrics',data_metrics,
                                                '--model_dir',outputs,
                                                '--src_model_name',model_name_param,
                                                '--exp_name',exp_name_param],
                                     inputs=[data_metrics],
                                     compute_target=cpu_compute_target,
                                     runconfig=runconfig,
                                     source_directory='.',
                                     allow_reuse=False
                                     )
    register_step.run_after(train_step)
    
    pipeline = Pipeline(workspace=ws,
                        steps=[train_step,register_step])
    pipeline.publish(name=env.hds_pipeline_name,
                     description='hyperdrivestep pipeline',
                     version=env.build_id)

if __name__ == '__main__':
    main()