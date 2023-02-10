GROUP="RG"
LOCATION="eastus"
WORKSPACE="AzureMLWorkspace"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

az extension remove -n ml
az extension add -n ml