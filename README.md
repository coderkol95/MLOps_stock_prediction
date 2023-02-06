Azure ML SDK v2

# TODO:

1. Build pipeline - Code changes to the repository

    - [ ] Pytests with code coverage
    - [ ] github actions during branch merge

2. Training pipeline - To train the model when new data is available / re-training is required

    - [ ] Data pipeline updation based on schedule
        - [X] Batch
        - [X] Data validation, capture the SD [Not applicable in this use case]

    - [ ] Model re-training and comparison with previous models
        - [ ] Model version details - hyperparameters, model structure
        - [ ] Performance metrics

    - [ ] Model registration if it is good enough
        - [ ] Performance, fairness, bias, equity and other responsible AI practices
        - [ ] Model profiling

3. Release pipeline - To deploy 

    - [ ] Parallely deploy the latest model
        - [ ] Create template for provisioning
        - [ ] Option for rollback
        - [ ] Enable useful logging

    - [ ] Monitoring
        - [ ] Data drift
        - [ ] Infrastructure performance
        - [ ] Model performance