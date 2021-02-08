import json
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import InferenceConfig,Model
from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute,ComputeTarget
from azureml.exceptions import ComputeTargetException
from azureml.core import Dataset
from env_variables import ENV
from utils import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str,required=True)
    args = parser.parse_args()
    
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
    
    model = get_models(ws=ws,
                      model_name=args.model_name
                      )
    
    environment = use_or_create_environment(ws=ws,
                                            env_name=env.env_name_scoring,
                                            create_new_env=env.create_new_env_scoring,
                                            conda_dependencies=env.conda_dependencies_scoring)
            
    try:
        aks_compute = AksCompute(ws,env.aks_service_name)
    except:
        aks_config = AksCompute.provisioning_configuration(cluster_purpose='DevTest')
        aks_compute = ComputeTarget.create(workspace=ws,
                                           name=env.aks_service_name,
                                           provisioning_configuration=aks_config)
        aks_compute.wait_for_completion(show_output=True)
    
    inference_config = InferenceConfig(entry_script='score.py',
                                       environment=environment,
                                       source_directory='.')
    
    aks_config = AksWebservice.deploy_configuration()
    aks_service_name = env.aks_service_name
    
    aks_service = Model.deploy(workspace=ws,
                               name=aks_service_name,
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=aks_config,
                               deployment_target=aks_compute,
                               overwrite=True
                               )
    aks_service.wait_for_deployment(show_output=True)
    print(aks_service.state)

if __name__ == '__main__':
    main()
#%%
