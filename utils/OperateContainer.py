import pandas as pd
import numpy as np
import json
import os
import argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

with open('config/storage_config.json','r') as f:
    storage_details=json.load(f)

storageAccountUrl = storage_details['storageAccountUrl']
storageAccountKey = storage_details['storageAccountKey']
containerName = storage_details['containerName']


blobServiceClient = BlobServiceClient(account_url=storageAccountUrl, credential=storageAccountKey)

# blobServiceClient.create_container(name=containerName)

def downloadBlob(blobName, filePath=None):
    
    blobClientInstance = blobServiceClient.get_blob_client(container=containerName, blob= blobName)
    if filePath is None:
        filePath = f"./{blobName}"
    with open(filePath, 'wb') as myBlob:
        blobData = blobClientInstance.download_blob()
        blobData.readinto(myBlob)

def uploadBlob(filePath):

    blobName = os.path.basename(filePath)
    blobClientInstance = blobServiceClient.get_blob_client(container=containerName, blob=blobName)
    with open(filePath,'rb') as myBlob:
        blobClientInstance.upload_blob(myBlob, overwrite=False)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--action", required=True, help="'upload' or 'download'", type=str)
    parser.add_argument("--filePath", default=None, required=False, help="Path of the file to save to / upload form", type=str)
    parser.add_argument("--blobName", required= False, help="Name of the blob to upload to / download from.", type=str)

    args = parser.parse_args()
    try:
        if args.action=='download':
            downloadBlob(args.blobName, args.filePath)
        
        elif args.action=="upload":
            uploadBlob(args.filePath)
    except:

        raise ProcessLookupError("Your requested action could not be performed.")