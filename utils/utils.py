from azure.ai.ml.entities import Environment, AmlCompute, Data
from azure.ai.ml.constants import AssetTypes
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(filename='../logs.log', encoding='utf-8', level=logging.INFO)

def fetch_env(
    ml_client,
    env_name:str,
    version:str,
    conda_file_path:str=None,
    image:str="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
    ):
    """
    
    Function to fetch/create environment.

    conda_file_path:str = Path to conda file
    env_name:str = Name of the environment
    label:str = Label of the environment

    """

    try:
        env = ml_client.environments.get(name=env_name, version=version)
        return env
    except:
        try:
            env = Environment(
                name=env_name,
                description="Custom environment for creating MLOps project for stock prediction",
                conda_file=conda_file_path,
                image=image,
                version=version,
                auto_increment_version=True
            )
            ml_client.environments.create_or_update(env)

            logging.info(f"{datetime.now()}: Environment with name {env.name} is registered to workspace, the environment version is {env.version}")
        
        except:
            logging.error("Could not create environment.")

def fetch_compute_cluster(
    ml_client,
    target_name:str,
    compute_size="STANDARD_DS2_V2",
    min_instances=1,
    max_instances=2,
    idle_time=180,
    ):

    """
    
    Compute fetching/creation.

    target_name:str = Name of the compute target
    compute_size:str = Compute size, default = STANDARD_DS2_V2
    min_instances:int = Min. no. of computes in the cluster, default=1
    max_instances:int = Max. no. of computes in the cluster, default=2
    idle_time:int = Idle timeout for scale down

    """

    try:
        compute = ml_client.compute.get(target_name)
    except Exception:
        print("Creating a new cpu compute target...")
        # Let's create the Azure ML compute object with the intended parameters
        compute = AmlCompute(
            name=target_name,
            type="amlcompute",
            size=compute_size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time,
        )
        # Now, we pass the object to MLClient's create_or_update method
        ml_client.begin_create_or_update(compute).wait() 
    return compute

def delete_compute_cluster(
    ml_client,
    target_name:str
    ):

    """
    
    Compute deletion.

    target_name:str = Name of the compute target you want to delete.

    """

    ml_client.compute.begin_delete(name=target_name).wait()

def upload_file_to_datastore(
    ml_client,
    file_path:str,
    name:str,
    version:str,
    description:str=None,
    tags:dict=None
    ):

    """
    
    Function to upload files from local directory to Azure ML Studio datastore.

    file_path:str = Local file path
    name:str = File of data asset
    version:str = Version of data asset
    description:str = Description of data asset
    tags:dict = Tags to attach with the data asset

    """

    file = pd.read_csv(file_path)
    len = file.shape[0]

    start = file.loc[0, "Unnamed: 0"]
    end = file.loc[len-1,"Unnamed: 0"]

    if tags is None:
        tags = {'length':len, "start":start,"end":end}

    else:
        tags.update({'length':len, "start":start,"end":end})

    logging.info(f"Uploading dataset to Azure ML Studio of length {len} for data from {start} : {end}")

    upload_data = Data(
    path=file_path,
    name=name,
    version=version,
    description=description,
    type=AssetTypes.URI_FILE,
    tags=tags
    )

    ml_client.data.create_or_update(upload_data)