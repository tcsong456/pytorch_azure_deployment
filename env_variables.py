from dataclasses import dataclass
import os
from typing import Optional

@dataclass
class ENV:
    workspace_name: Optional[str] = os.environ.get('WORKSPACE_NAME')
    subscription_id: Optional[str] = os.environ.get('SUBSCRIPTION_ID')
    resource_group: Optional[str] = os.environ.get('RESOURCE_GROUP')
    build_id: Optional[str] = os.environ.get('BUILD_ID')
    build_url: Optional[str] = os.environ.get('BUILD_URL')
    experiment_name: Optional[str] = os.environ.get('EXPERIMENT_NAME')
    pipeline_name: Optional[str] = os.environ.get('PIPELINE_NAME')    
    conda_dependencies: Optional[str] = os.environ.get('CONDA_DEPENDENCIES')
    vm_size_cpu: Optional[str] = os.environ.get('VM_SIZE_CPU')
    vm_size_gpu: Optional[str] = os.environ.get('VM_SIZE_GPU')
    vm_size_cpu_scoring: Optional[str] = os.environ.get('VM_SIZE_CPU_SCORING')
    vm_size_gpu_scoring: Optional[str] = os.environ.get('VM_SIZE_GPU_SCORING')
    vm_priority: Optional[str] = os.environ.get('VM_PRIORITY')
    max_epochs: Optional[int] = int(os.environ.get('MAX_EPOCHS'))
    auto_trigger_pipeline: Optional[str] = os.environ.get('AUTO-TRIGGER-PIPELINE')
    model_name: Optional[str] = os.environ.get('MODEL_NAME')
    scoring_file_name: Optional[str] = os.environ.get('SCORING_FILE_NAME')
    loss_file_name: Optional[str] = os.environ.get('LOSS_FILE_NAME')
    env_name: Optional[str] = os.environ.get('ENV_NAME')
    datastore_name: Optional[str] = os.environ.get('DATASTORE_NAME')
    create_new_env: Optional[str] = os.environ.get('CREATE_NEW_ENV')
    create_new_env_scoring: Optional[str] = os.environ.get('CREATE_NEW_ENV_SCORING')
    container_name: Optional[str] = os.environ.get('CONTAINER_NAME')
    max_total_runs: Optional[int] = int(os.environ.get('MAX_TOTAL_RUNS'))
    max_concurrent_runs: Optional[int] = int(os.environ.get('MAX_CONCURRENT_RUNS'))
    max_duration_minutes: Optional[int] = int(os.environ.get('MAX_DURATION_MINUTES'))
    hds_pipeline_name: Optional[str] = os.environ.get('HDS_PIPELINE_NAME')
    src_model_name: Optional[str] = os.environ.get('SRC_MODEL_NAME')
    aks_service_name: Optional[str] = os.environ.get('AKS_SERVICE_NAME')
    batch_size: Optional[int] = int(os.environ.get('BATCH_SIZE'))
