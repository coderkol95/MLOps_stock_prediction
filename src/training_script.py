from azure.ai.ml import command, Input, Output
from datetime import datetime
import logging
from azure.ai.ml.constants import AssetTypes
logging.basicConfig(filename='../logs.log', encoding='utf-8', level=logging.INFO)


def trigger_train(
    ml_client,
    job_name:str,
    data_asset_uri:str,
    environment,
    compute
    ):
    """
    
    Trigger training pipeline

    job_name:str = Name of the job
    data_asset_uri:str = Data asset URI for training job
    environment: Environment
    compute: Compute cluster
    
    """
    try:
        logging.info("Training")

        local_model_name = f"modelstock_pred_{str(datetime.now().date())}"

        job = command(
            name=job_name,
            inputs={
                "data": Input(type=AssetTypes.URI_FILE, mode="ro_mount", path=data_asset_uri),
                "test_train_ratio": 0.25,
                "local_model_name":local_model_name
                },
            code="./src/",
            command="python train.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --local_model_name ${{inputs.local_model_name}}",
            environment=environment,
            compute=compute.name,
            experiment_name="train_model_stock_price_prediction",
            display_name="stock_price_prediction",
        )
        ml_client.create_or_update(job)

    except:
        logging.error("Could not train.")



## If run as a CLI job, accept inputs via argparse and then get compute and env using ml_client