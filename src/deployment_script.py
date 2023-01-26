from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Environment


def trigger_deployment(
    ml_client,
    endpoint_name:str,
    model,
    env,
    deployment_details,
    endpoint_type:str="real-time"
    ):

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Real-time endpoint"
    )

    ml_client.begin_create_or_update(endpoint).wait()

    if ml_client.online_endpoints.get(name=endpoint_name).provisioning_state!="Succeeded":
        pass

    # Deployment script
    code_config = CodeConfiguration(
            code='./src/', scoring_script="deployment.py"
        )

    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_details["name"],
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=code_config,
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    # create the deployment:
    ml_client.begin_create_or_update(deployment).wait()
    
    # blue deployment takes 100 traffic
    endpoint.traffic = {deployment_details["name"]: deployment_details["traffic"]}
    ml_client.begin_create_or_update(endpoint).wait()