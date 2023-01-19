import pandas as pd
import numpy as np
import torch
from azureml.core import Run, Model

run = Run.get_context()
ws = run.experiment.workspace

run.log("Starting to train")
....

Summarize conversation  


torch.save(...,path=model_path)
Model.register(workspace=ws, model_name = ,model_path = model_path)

run.complete()