from azureml.core import Workspace,Experiment
from azureml.pipeline.core import PublishedPipeline
from azureml.core.authentication import ServicePrincipalAuthentication
from env_variables import ENV
import json
from utils import create_or_use_workspace

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
    matched_pipe = []
    published_pipelines = PublishedPipeline.list(ws)
    for pipe in published_pipelines:
        if pipe.name == env.pipeline_name:
            if pipe.version == env.build_id:
                matched_pipe.append(pipe)
    if len(matched_pipe) > 1:
        raise ValueError('only one pipeline should be published')
    elif len(matched_pipe) == 0:
        raise ValueError('no pipeline is published')
    else:
        published_pipeline = matched_pipe[0]
        
        tags = {}
#        pipeline_param = {'model_name':env.model_name,
#                          'epochs':env.max_epoches}
        if env.build_id is not None:
            tags.update({'build_id':env.build_id})
            if env.build_url is not None:
                tags.update({'build_url':env.build_url})
        exp = Experiment(name=env.experiment_name,
                         workspace=ws)
        run = exp.submit(published_pipeline,
                         tags=tags,
                         pipeline_parameters={'model_name':env.model_name,
                                              'model_tag_name':" ",
                                              'model_tag_value':" ",
                                              'epochs':env.max_epochs,
                                              'scoring_file_name':env.scoring_file_name,
                                              'loss_file_name':env.loss_file_name,
                                              'exp_name':env.experiment_name})
        print(f'pipeline {published_pipeline.id} initiated,run id: {run.id}')

if __name__ == '__main__':
    main()
