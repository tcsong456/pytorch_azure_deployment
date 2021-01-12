echo 'uploading data'
export WORKSPACE_NAME='aml-workspace'
export RESOURCE_GROUP='aml-resource-group'
export SUBSCRIPTION_ID=$(az account show --query id -o tsv)
python upload_data.py
echo 'shell already run' >shell.txt
