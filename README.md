# Practical MLOps using Azure

<h1> Hey there! I'm Anupam 👋 </h1>
<h2> A data scientist from India 🇮🇳 </h2>

<h3> 👨🏻‍💻 About Me </h3>

- I work as a data scientist in the consulting industry.
- I love solving problems and automating solutions.
- I love music and travelling!

# Project description

This project implements automated MLOps. Every week it fetches data via an API, registers the data as a dataset in Azure datastore. It updates the versions of required pipeline components like run ID, model version etc.. Then it trains the registers a pytorch LSTM model in Azure ML studio. This model is then deployed to an online endpoint. Everything is done through cron jobs.

To read in more detail, the article for this project is published here: 

This project is still in development. :) 

Improvement items:

1. Pytests
2. Implement option of monitoring data drift
3. Model profiling
4. Infrastructure performance
5. Model performance assessment on data slices
6. Model performance on live data

# How to use the project

- Fork this repository
- Create an Azure ML Studio and resource group if you do not have already
- Create a Service Principal([ref.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?tabs=sdk#configure-a-service-principal)) in Azure and save the generated JSON as `AZURE_CREDENTIALS` in Settings> Secrets and variables> Actions>New repository secret
You would need this to access ML Studio
- Create a Personal Access Token from from Settings>Developer settings>Personal access tokens and save it as  `PAT` to your repository secrets. Allow workflow read/write access, actions, code, commit statuses, merge queues, pull requests.
- Update your workspace and resource group details in `cli/setup.sh`

# Data pipeline

The cron job is `.github/workflows/data_pipeline.yml`
- Initial setup
- Downloads the data via an API
- Updates different components' version
- Pushes changes to the repository
- Registers the dataset in Azure ML Studio datastore
Associated files: `jobs/data_upload.yml`, `jobs/data_download.py`, `update_yamls.py`
# Training pipeline

The cron job is `.github/workflows/model_pipeline.yml`
- Initial setup
- Creates a compute
- Trains a LSTM model using pytorch lightning on the latest dataset
- Registers the new model in the workspace
Associated files: `jobs/train.yml`, `jobs/train.py`
# Deployment pipeline

The cron job is `.github/workflows/deployment_pipeline.yml`
- Creates new online endpoint and deployment for the first time
- For the successive training runs, the new model would be deployed to the existing online endpoint
Associated files: `jobs/deploy.yml`, `jobs/deployment.py`