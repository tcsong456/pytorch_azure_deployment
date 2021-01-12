from utils import *
from env_variables import ENV
from azureml.core import ScriptRunConfig,RunConfiguration,Dataset,Datastore
from azureml.pipeline.core import Pipeline,PipelineData,PipelineParameter
from azureml.pipeline.steps import PythonScriptStep,ParallelRunConfig,ParallelRunStep
from azureml.data.datapath import DataPath,DataPathComputeBinding
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.core import Experiment
import os
import json

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
#                                            use_gpu=True,
#                                            enable_docker=True
                                            )
    datastore = use_or_create_datastore(ws=ws,
#                                        config=config,
                                        datastore_name=env.datastore_name
                                        )
    
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
    datapath_input = PipelineParameter('datapath_input',default_value=datapath)
    input_datapath = (datapath_input,DataPathComputeBinding(mode='mount'))
    
    output = PipelineData('output',datastore=datastore)
    output_path = PipelineData('output_path',datastore=datastore)
    model_name_param = PipelineParameter(name='model_name',default_value=env.model_name)
    model_epoch_param = PipelineParameter(name='epochs',default_value=15)
    model_version_param = PipelineParameter(name='model_version',default_value='none')
    model_tag_name_param = PipelineParameter(name='model_tag_name',default_value='')
    model_tag_value_param = PipelineParameter(name='model_tag_value',default_value='')
    scoring_file_name_param = PipelineParameter(name='scoring_file_name',default_value='')
    loss_file_name_param = PipelineParameter(name='loss_file_name',default_value='')

    runconfig = RunConfiguration()
    runconfig.environment = environment
    
    train_step = PythonScriptStep( name='train_step',
                                   source_directory='.',
                                   script_name='pytorch_train.py',
                                   arguments=['--output_dir',output,
                                              '--data-path',input_datapath,
                                              '--model-name',model_name_param,
                                              '--num_epochs',model_epoch_param],
                                   inputs=[input_datapath],
                                   outputs=[output],
                                   compute_target=gpu_compute_target,
                                   runconfig=runconfig,
                                   allow_reuse=True
                                   )
    print('train_step built')
    
    register_step = PythonScriptStep(name='register_step',
                                     source_directory='.',
                                     script_name='register_prc.py',
                                     arguments=['--step-input',output,
                                                '--model-name',model_name_param],
                                     inputs=[output],
                                     compute_target=cpu_compute_target,
                                     runconfig=runconfig,
                                     allow_reuse=False
                                       )
    print('register_step built')
    register_step.run_after(train_step)
    
    scoring_step = PythonScriptStep(name='scoring_step',
                                    source_directory='.',
                                    script_name='batch_score.py',
                                    arguments=['--model-name',model_name_param,
                                               '--model-version',model_version_param,
                                               '--tag-name',model_tag_name_param,
                                               '--tag-value',model_tag_value_param,
                                               '--scoring-file-name',scoring_file_name_param,
                                               '--loss-file-name',loss_file_name_param,
                                               '--step-input',input_datapath
                                               ],
                                    inputs=[input_datapath],
                                    outputs=[output_path],
                                    compute_target=gpu_compute_target,
                                    runconfig=runconfig,
                                    allow_reuse=False
                                    )
    scoring_step.run_after(register_step)
    print('scoring_step built')
    
    pipeline = Pipeline(workspace=ws,
                        steps=[train_step,register_step,scoring_step])
    pipeline.publish(name=env.pipeline_name,
                     description='pytorch_deployment_sample',
                     version=env.build_id)
    
if __name__ == '__main__':
    main()

                              