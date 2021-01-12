from azureml.core import Workspace,Environment,Datastore,Model
from azureml.core.compute import AmlCompute,ComputeTarget
from azureml.core.runconfig import DEFAULT_CPU_IMAGE,DEFAULT_GPU_IMAGE
from azureml.exceptions import ComputeTargetException,WorkspaceException,ProjectSystemException,UserErrorException
from typing import Optional

def create_or_use_workspace(workspace_name,
                            subscription_id,
                            resource_group,
                            auth=None,
                            location='westeurope'):
    try:
        create_new = False
        try:
            ws = Workspace.get(name=workspace_name,
                               subscription_id=subscription_id,
                               resource_group=resource_group,
                               auth=auth)
        except WorkspaceException:
            print('wrong workspace name provided')
            create_new = True
        except UserErrorException:
            print('no access to the subscription provided,probably non-existing subscritpion')
            create_new = True
        except ProjectSystemException:
            print('wrong resource_group name provided')
            create_new = True
        
        if create_new:
            ws = Workspace.create(name=workspace_name,
                                  subscription_id=subscription_id,
                                  resource_group=resource_group,
                                  create_resource_group=True,
                                  location=location,
                                  auth=auth)
        return ws
    except Exception as error:
        print(error)
        exit(1)

def use_or_create_computetarget(ws,
                                env,
                                compute_name,
                                cpu_cluster=True,
                                batch_scoring=False,
                                ):
    try:
        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]
            if compute_target is not None and type(compute_target) == AmlCompute:
                print(f'found exsting compute target: {compute_name}')
        else:
            if cpu_cluster:
                if batch_scoring:
                    vm_size = env.vm_size_cpu_scoring
                else:
                    vm_size = env.vm_size_cpu
            else:
                if batch_scoring:
                    vm_size = env.vm_size_gpu_scoring
                else:
                    vm_size = env.vm_size_gpu
            compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                   vm_priority=env.vm_priority if not batch_scoring else env.vm_priority_scoring,
                                                                   min_nodes=int(env.min_nodes) if not batch_scoring else env.min_nodes_scoring,
                                                                   max_nodes=int(env.max_nodes) if not batch_scoring else env.max_nodes_scoring)
            compute_target = ComputeTarget.create(workspace=ws,
                                                  name=compute_name,
                                                  provisioning_configuration=compute_config)
            compute_target.wait_for_completion(show_output=True)
        return compute_target
    except ComputeTargetException as error:
        print(error)
        exit(1)

def use_or_create_environment(ws,
                              env_name,
                              conda_dependencies:Optional[str],
                              enable_docker=False,
                              use_gpu=False,
                              create_new_env=False):
    try:
        assert env_name is not None,'environment name must be specified'
        env_list = Environment.list(ws)
        if env_name in env_list and not create_new_env == 'true':
            environment = env_list[env_name]
        elif create_new_env == 'true':
            environment = Environment.from_conda_specification(name=env_name,
                                                               file_path=conda_dependencies)
        else:
            raise Exception('Nor env available and created,please enable create_new_env function')
            
        if enable_docker:
            environment.docker.enabled = True
            if use_gpu:
                environment.docker.base_image = DEFAULT_GPU_IMAGE
            else:
                environment.docker.base_image = DEFAULT_CPU_IMAGE
        
        environment.register(workspace=ws)
        return environment
    except Exception as error:
        print(error)
        exit(1)

def use_or_create_datastore(ws,
                            datastore_name=None,
                            container_name=None,
                            storage_account_name=None,
                            storage_access_key=None,
                            use_default=True,
                            overwrite=False
                            ):
    try:
        if use_default:
            datastore = ws.get_default_datastore()
            return datastore
            
        assert datastore_name is not None,'datastore name must be provided'
        if datastore_name in ws.datastores:
            datastore = ws.datastores[datastore_name]
        else:
            assert container_name is not None,'when creating new storage container,container_name must be given'
            assert storage_account_name is not None,'when creating new storage container,storage_account_name must be given'
            assert storage_access_key is not None,'when creating new storage container,storage_access_key must be given'
            datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                                datastore_name=datastore_name,
                                                                container_name=container_name,
                                                                account_name=storage_account_name,
                                                                account_key=storage_access_key,
                                                                overwrite=overwrite)
        return datastore
    except Exception as error:
        print(error)
        exit(1)

def get_models(ws,
               model_name,
               model_version=None,
               tags=None):
    if model_version != 'none' and model_version is not None:
        model = Model(workspace=ws,
                      name=model_name,
                      version=model_version,
                      tags=tags)
    else:
        models = Model.list(workspace=ws,
                            latest=True)
        for m in models:
            if m.name == model_name:
                model = m
                
    if model is None:
        raise FileNotFoundError('no matched model found')
    
    return model
#%%
