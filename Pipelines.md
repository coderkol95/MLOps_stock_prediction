Azure ML SDK v2

# TODO:

1. Print csv and send column data via JSON
2. Accept these and write them out via csv
3. Consume ticker data path
4. Upload to Azure


1. Build pipeline - Code changes to the repository : trigger=push

    - [ ] Pytests with code coverage
    - [ ] github actions during branch merge

2. Training pipeline - To train the model when new data is available / re-training is required : trigger=schedule

    - [X] data: 
        Data pipeline updation based on schedule
        - [X] Batch
        - [X] Data validation, capture the SD [Not applicable in this use case]
        - [X] Data push to github
        - [X] Data upload to Azure
        - [X] Delete local file - not doing, keeping only one file
        - [X] Option to input ticker during cron job

    - [X] train:
        Model re-training and publishing of metrics
        - [X] Model version details - hyperparameters, model structure - will do when improving model - done via source control. Can add tags when specifically improving the model.
        - [X] Publish performance metrics

    - [X] register:
        Model registration with metrics
        - [X] Performance, fairness, bias, equity and other responsible AI metrics to be measured - NA in this case
        - [ ] Model profiling

3. Release pipeline:

    There can be need for re-training & re-deployment under two scenarios:
    1. Data drifts frequently and model is updated on schedule( this is the case in this project )
    2. Data drift needs to be detected and model needs to be subsequently retrained. In that case the below unchecked points need to be looked at.

    - [ ] Compare current model with previous models and deploy if good enough
        - [ ] Compare with live model
        - [X] Create template for provisioning
        - [X] Enable useful logging

    - [ ] Monitoring
        - [ ] Data drift
        - [ ] Infrastructure performance - CI
        - [ ] Model performance on LIVE data - CI