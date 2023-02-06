Azure ML SDK v2

# TODO:

1. Build pipeline - Code changes to the repository : trigger=push

    - [ ] Pytests with code coverage
    - [ ] github actions during branch merge

2. Training pipeline - To train the model when new data is available / re-training is required : trigger=schedule

    - [X] data: 
        Data pipeline updation based on schedule
        - [X] Batch
        - [X] Data validation, capture the SD [Not applicable in this use case]
        - [ ] Delete local file - will do later

    - [ ] train: 
        Model re-training and publishing of metrics
        - [ ] Model version details - hyperparameters, model structure
        - [ ] Publish performance metrics

    - [ ] register: 
        Model registration and comparison with previous models
        - [ ] Performance, fairness, bias, equity and other responsible AI metrics to be measured
        - [ ] Model profiling

3. Release pipeline - To deploy : trigger=new model availability

    - [ ] Parallely deploy the latest model
        - [ ] Create template for provisioning
        - [ ] Option for rollback
        - [ ] Enable useful logging

    - [ ] Monitoring
        - [ ] Data drift
        - [ ] Infrastructure performance
        - [ ] Model performance