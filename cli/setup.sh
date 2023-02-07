GROUP="RG"
LOCATION="eastus"
WORKSPACE="AzureMLWorkspace"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION
az extension add -n ml