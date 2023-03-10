import pandas as pd
from datetime import datetime
import re
from copy import deepcopy

#model_pipeline

with open('../.github/workflows/model_pipeline.yml','r') as f:
    model_pipeline=f.read()

run_id=re.findall('run-[0-9]',model_pipeline)
run_id=run_id[0]
run_n = int(re.findall('[0-9]',run_id)[0])
new_run_id = f"run-{run_n+1}"

model_version=re.findall('--version [0-9]',model_pipeline)
model_version = model_version[0]
version = int(re.findall('[0-9]',model_version)[0])
new_model_version = f"--version {version+1}"

model_pipeline= re.sub(model_version, new_model_version, model_pipeline)
model_pipeline=re.sub(run_id, new_run_id, model_pipeline)

with open('../.github/workflows/model_pipeline.yml','w') as f:
    f.write(model_pipeline)

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

#deploy

with open('../jobs/deploy.yml','r') as f:
    deploy=f.read()

curr_model=re.findall('GA_model:[0-9]',deploy)[0]
curr_model_version = curr_model.split(':')[1]
curr_model_version  = int(curr_model_version) + 1
deploy=re.sub('GA_model:[0-9]',f'GA_model:{curr_model_version}',deploy)

with open('../jobs/deploy.yml','w') as f:
    f.write(deploy)