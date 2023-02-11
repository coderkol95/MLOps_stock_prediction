import pandas as pd
from datetime import datetime
import re
from copy import deepcopy

#train_pipeline

with open('../.github/workflows/train_pipeline.yml','r') as f:
    train_pipeline=f.read()

run_id=re.findall('run-[0-9]',train_pipeline)
run_id=run_id[0]
run_n = int(re.findall('[0-9]',run_id)[0])
new_run_id = f"run-{run_n+1}"

model_version=re.findall('--version [0-9]',train_pipeline)
model_version = model_version[0]
version = int(re.findall('[0-9]',model_version)[0])
new_model_version = f"--version {version+1}"

train_pipeline= re.sub(model_version, new_model_version, train_pipeline)
train_pipeline=re.sub(run_id, new_run_id, train_pipeline)

with open('../.github/workflows/train_pipeline.yml','w') as f:
    f.write(train_pipeline)

#train

with open('../jobs/train.yml','r') as f:
    train=f.read()

uri_path=train[train.index('path:'):train.index('@latest')].split(":")
with open('../.github/workflows/data_pipeline.yml','r') as f:
    data_pipeline=f.read()
ticker_string=data_pipeline[data_pipeline.index('ticker'): data_pipeline.index('.NS')]
ticker=ticker_string[ticker_string.index(':'):].strip(":").strip(" ")
new_uri_path = deepcopy(uri_path)
new_uri_path[2]=ticker
uri_path=":".join(uri_path)
new_uri_path=":".join(new_uri_path)

train=re.sub(run_id, new_run_id, train)
train=re.sub(uri_path, new_uri_path, train)

with open('../jobs/train.yml','w') as f:
    f.write(train)