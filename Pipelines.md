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

3. Release pipeline - To deploy : trigger=new model availability

    - [ ] Compare current model with previous models and deploy if good enough
        - [ ] Compare with live model
        - [ ] Create template for provisioning
        - [ ] Enable useful logging

    - [ ] Monitoring
        - [ ] Data drift
        - [ ] Infrastructure performance
        - [ ] Model performance